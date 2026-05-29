#!/usr/bin/env python3
"""
Forced Alignment Benchmark: TinyAligner vs MFA vs Gentle

Compares character-level boundary predictions against reference TextGrids.

Methods:
  1. TinyAligner  - our lightweight CTC model (~480K params)
  2. MFA          - Montreal Forced Aligner (Kaldi-based, pre-trained)
  3. Gentle       - lowerquality/gentle (Kaldi-based, via HTTP API on localhost:8765)

Metrics:
  - Boundary F1 at ±20ms / ±40ms / ±80ms tolerance
  - Mean Absolute Error (MAE) on matched boundary pairs
  - Real-Time Factor (RTF)
  - Model size

Usage:
    conda activate bfa
    python scripts/benchmark_compare.py                        # all 3 methods
    python scripts/benchmark_compare.py --methods tiny mfa     # skip gentle
    python scripts/benchmark_compare.py --methods tiny         # TinyAligner only

Gentle setup (optional):
    git clone https://github.com/lowerquality/gentle
    cd gentle && bash install.sh
    python serve.py &   # starts HTTP server on :8765
"""

import json
import time
import shutil
import tempfile
import argparse
import subprocess
import sys
import warnings
import re
import requests
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Data paths (same as scripts/train.py) ─────────────────────────────────────
AISHELL_LEXICON  = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
CMUDICT_PATH     = str(Path(__file__).parent.parent / "data" / "lexicons" / "cmudict-0.7b")

# ── MFA paths ──────────────────────────────────────────────────────────────────
MFA_MODELS_DIR   = Path.home() / "Documents" / "MFA" / "pretrained_models"
MFA_ACOUSTIC_EN  = str(MFA_MODELS_DIR / "acoustic"   / "english_mfa.zip")
MFA_ACOUSTIC_ZH  = str(MFA_MODELS_DIR / "acoustic"   / "mandarin_mfa.zip")
MFA_DICT_EN      = str(MFA_MODELS_DIR / "dictionary" / "english_mfa.dict")
MFA_DICT_ZH      = str(MFA_MODELS_DIR / "dictionary" / "mandarin_pinyin.dict")

# ── Gentle ─────────────────────────────────────────────────────────────────────
GENTLE_URL       = "http://localhost:8765/transcriptions"


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    method: str
    f1_20ms: float
    f1_40ms: float
    f1_80ms: float
    precision_20ms: float
    recall_20ms: float
    mae_ms: float
    rtf: float
    model_size_mb: float
    n_files: int
    per_file: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# TextGrid parser (manual – handles overlapping reference intervals)
# ─────────────────────────────────────────────────────────────────────────────

def parse_textgrid_raw(path: str, tier_name: str = "chars") -> List[Tuple[float, float, str]]:
    """
    Raw regex-based TextGrid parser. Tolerates overlapping intervals
    that praatio rejects, since the reference TextGrids use a format
    where interval xmin/xmax can be non-contiguous.
    Returns list of (start_s, end_s, label) for non-empty intervals.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Find all tier blocks
    tier_blocks = re.split(r'item\s*\[\d+\]\s*:', text)[1:]

    target_block = None
    for block in tier_blocks:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        if name_match and name_match.group(1).lower() == tier_name.lower():
            target_block = block
            break

    if target_block is None and tier_blocks:
        target_block = tier_blocks[0]   # fallback to first tier

    if not target_block:
        return []

    intervals = re.findall(
        r'xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+).*?text\s*=\s*"([^"]*)"',
        target_block,
        re.DOTALL,
    )

    SKIP = {"", "sil", "sp", "SIL", "SP"}
    return [
        (float(xmin), float(xmax), label.strip())
        for xmin, xmax, label in intervals
        if label.strip() not in SKIP
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Boundary metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_boundary_metrics(
    pred_intervals: List[Tuple[float, float, str]],
    ref_intervals:  List[Tuple[float, float, str]],
    tolerance_s: float = 0.020,
) -> Tuple[float, float, float, float]:
    """
    Precision / Recall / F1 / MAE on start-boundaries.
    Greedy one-to-one match within ±tolerance_s.
    """
    pred_times = [s for s, e, l in pred_intervals]
    ref_times  = [s for s, e, l in ref_intervals]

    if not pred_times or not ref_times:
        return 0.0, 0.0, 0.0, float("inf")

    ref_matched  = [False] * len(ref_times)
    pred_matched = [False] * len(pred_times)
    matched_errors = []

    for ri, rt in enumerate(ref_times):
        best_dist, best_pi = float("inf"), -1
        for pi, pt in enumerate(pred_times):
            if pred_matched[pi]:
                continue
            dist = abs(rt - pt)
            if dist <= tolerance_s and dist < best_dist:
                best_dist, best_pi = dist, pi
        if best_pi >= 0:
            ref_matched[ri]        = True
            pred_matched[best_pi]  = True
            matched_errors.append(best_dist * 1000)

    tp = sum(ref_matched)
    fp = sum(not m for m in pred_matched)
    fn = sum(not m for m in ref_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mae_ms    = float(np.mean(matched_errors)) if matched_errors else float("inf")

    return precision, recall, f1, mae_ms


def _collect_metrics(pred, ref, tolerances, all_m):
    fm = {}
    for tol in tolerances:
        P, R, F1, mae = compute_boundary_metrics(pred, ref, tol)
        key = int(tol * 1000)
        all_m[tol]["P"].append(P)
        all_m[tol]["R"].append(R)
        all_m[tol]["F1"].append(F1)
        all_m[tol]["MAE"].append(mae)
        fm[f"f1_{key}ms"] = round(F1, 4)
    return fm


def _finalize(all_m, total_time_s, total_audio_s, method, model_size_mb, per_file):
    rtf = total_time_s / max(total_audio_s, 1e-6)
    valid_mae = [m for m in all_m[0.020]["MAE"] if m != float("inf")]
    return MethodResult(
        method=method,
        f1_20ms=float(np.mean(all_m[0.020]["F1"])) if all_m[0.020]["F1"] else 0.0,
        f1_40ms=float(np.mean(all_m[0.040]["F1"])) if all_m[0.040]["F1"] else 0.0,
        f1_80ms=float(np.mean(all_m[0.080]["F1"])) if all_m[0.080]["F1"] else 0.0,
        precision_20ms=float(np.mean(all_m[0.020]["P"])) if all_m[0.020]["P"] else 0.0,
        recall_20ms=float(np.mean(all_m[0.020]["R"])) if all_m[0.020]["R"] else 0.0,
        mae_ms=float(np.mean(valid_mae)) if valid_mae else float("inf"),
        rtf=rtf, model_size_mb=model_size_mb,
        n_files=len(per_file), per_file=per_file,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

def is_chinese(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text.replace(" ", "")), 1) > 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: TinyAligner
# ─────────────────────────────────────────────────────────────────────────────

def load_tinyaligner(checkpoint_path: str, device: str):
    from tiny_aligner.model import TinyAligner
    from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "phone2idx" in ckpt:
        phone2idx = ckpt["phone2idx"]
        idx2phone = {int(k): v for k, v in ckpt["idx2phone"].items()}
        cfg = ckpt.get("config", {})
        model = TinyAligner(n_phones=cfg.get("n_phones", len(phone2idx)),
                            hidden=cfg.get("hidden", 128))
        model.load_state_dict(ckpt["model"])
    else:
        from tiny_aligner.model import build_model
        lexicon = merge_lexicons(AISHELL_LEXICON, CMUDICT_PATH)
        phone2idx, idx2phone = build_phone_vocab(lexicon)
        model = build_model(len(phone2idx))
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.to(device).eval()
    lexicon = merge_lexicons(AISHELL_LEXICON, CMUDICT_PATH)
    return model, lexicon, phone2idx, idx2phone


def benchmark_tinyaligner(test_files, checkpoint_path, device, tolerances, ref_tier="chars"):
    from tiny_aligner.align import align

    print(f"\n{'='*70}")
    print("  Method 1: TinyAligner")
    print(f"{'='*70}")

    model, lexicon, phone2idx, idx2phone = load_tinyaligner(checkpoint_path, device)
    n_params   = sum(p.numel() for p in model.parameters())
    size_mb    = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
    print(f"  Parameters: {n_params:,}  |  Size: {size_mb:.1f} MB")

    # Map ref tier name → TinyAligner result key
    TIER_TO_KEY = {"chars": "char_intervals", "words": "word_intervals", "phones": "phone_intervals"}
    result_key  = TIER_TO_KEY.get(ref_tier, "char_intervals")
    print(f"  Comparing tier: '{ref_tier}'  (result key: '{result_key}')")

    total_time_s = total_audio_s = 0.0
    per_file = []
    all_m = {t: {"P": [], "R": [], "F1": [], "MAE": []} for t in tolerances}

    for item in test_files:
        info      = torchaudio.info(item["wav"])
        audio_dur = info.num_frames / info.sample_rate

        try:
            t0 = time.perf_counter()
            result = align(model=model, wav_path=item["wav"], text=item["text"],
                           lexicon=lexicon, phone2idx=phone2idx, idx2phone=idx2phone,
                           output_textgrid=None, device=torch.device(device))
            elapsed = time.perf_counter() - t0
            pred = result.get(result_key, result.get("char_intervals", []))
        except Exception as e:
            print(f"  ✗ {Path(item['wav']).stem}: {e}"); continue

        # Try requested tier first, then fall back through alternatives
        ref = parse_textgrid_raw(item["textgrid"], ref_tier)
        if not ref:
            for fallback in ("chars", "words", "phones"):
                ref = parse_textgrid_raw(item["textgrid"], fallback)
                if ref:
                    print(f"  ⚠  '{ref_tier}' tier not found, using '{fallback}' tier for {Path(item['wav']).stem}")
                    break
        if not ref:
            print(f"  ✗ No reference intervals for {Path(item['wav']).stem}"); continue

        total_time_s  += elapsed
        total_audio_s += audio_dur
        fm = {"file": Path(item["wav"]).stem, "duration_s": round(audio_dur, 2)}
        fm.update(_collect_metrics(pred, ref, tolerances, all_m))
        fm["rtf"] = round(elapsed / audio_dur, 4)
        per_file.append(fm)

        print(f"  ✓ {Path(item['wav']).stem:25s}  dur={audio_dur:.1f}s  "
              f"F1@20ms={all_m[0.020]['F1'][-1]:.3f}  "
              f"F1@40ms={all_m[0.040]['F1'][-1]:.3f}  "
              f"MAE={all_m[0.020]['MAE'][-1]:.1f}ms  "
              f"RTF={elapsed/audio_dur:.4f}")

    return _finalize(all_m, total_time_s, total_audio_s, "TinyAligner", size_mb, per_file)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Montreal Forced Aligner (MFA)
# ─────────────────────────────────────────────────────────────────────────────

def chars_to_pinyin(text: str) -> str:
    """Convert Chinese characters to tone-numbered pinyin words for MFA."""
    from pypinyin import pinyin, Style
    result = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            py = pinyin(ch, style=Style.TONE3, heteronym=False)
            if py and py[0]:
                syllable = py[0][0]
                # Ensure tone number is present; default to 5 (neutral) if missing
                if not syllable[-1].isdigit():
                    syllable += "5"
                result.append(syllable)
        elif ch.isascii() and ch.isalpha():
            result.append(ch.upper())
        # skip punctuation / spaces
    return " ".join(result)


def prepare_mfa_corpus(test_files: List[Dict], corpus_dir: Path):
    """
    Create MFA corpus directory:
    Each utterance gets:
      corpus_dir/<stem>.wav  (symlink or copy)
      corpus_dir/<stem>.lab  (transcript)
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for item in test_files:
        stem = Path(item["wav"]).stem
        wav_dst = corpus_dir / f"{stem}.wav"
        lab_dst = corpus_dir / f"{stem}.lab"

        # Symlink audio
        if not wav_dst.exists():
            wav_dst.symlink_to(Path(item["wav"]).resolve())

        # Write transcript (.lab)
        text = item["text"]
        if is_chinese(text):
            transcript = chars_to_pinyin(text)
        else:
            # MFA English: just the plain text, strip punctuation
            transcript = re.sub(r"[^\w\s']", "", text).upper()
        lab_dst.write_text(transcript, encoding="utf-8")


def parse_mfa_textgrid(tg_path: str) -> List[Tuple[float, float, str]]:
    """Parse MFA output TextGrid — uses the 'words' tier (best for char comparison)."""
    # Try words tier first, fall back to phones
    for tier in ("words", "phones"):
        intervals = parse_textgrid_raw(tg_path, tier)
        if intervals:
            return intervals
    return []


def benchmark_mfa(test_files: List[Dict], tolerances: List[float]) -> MethodResult:
    print(f"\n{'='*70}")
    print("  Method 2: Montreal Forced Aligner (MFA)")
    print(f"{'='*70}")

    if not shutil.which("mfa"):
        print("  ✗ mfa not in PATH. Install: conda install -c conda-forge montreal-forced-aligner")
        return None

    # Split by language
    zh_files = [f for f in test_files if is_chinese(f["text"])]
    en_files = [f for f in test_files if not is_chinese(f["text"])]

    total_time_s = total_audio_s = 0.0
    per_file = []
    all_m = {t: {"P": [], "R": [], "F1": [], "MAE": []} for t in tolerances}

    def run_mfa_group(files, acoustic, dictionary, label):
        nonlocal total_time_s, total_audio_s
        if not files:
            return
        if not Path(acoustic).exists():
            print(f"  ✗ Acoustic model not found: {acoustic}")
            return
        if not Path(dictionary).exists():
            print(f"  ✗ Dictionary not found: {dictionary}")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            output_dir = Path(tmpdir) / "output"
            prepare_mfa_corpus(files, corpus_dir)

            print(f"  Running MFA align on {len(files)} {label} file(s)...")
            t0 = time.perf_counter()
            result = subprocess.run(
                [
                    "mfa", "align",
                    "--clean",
                    "--overwrite",
                    str(corpus_dir),
                    dictionary,
                    acoustic,
                    str(output_dir),
                ],
                capture_output=True, text=True, timeout=300,
            )
            elapsed_total = time.perf_counter() - t0

            if result.returncode != 0:
                print(f"  ✗ MFA failed:\n{result.stderr[-800:]}")
                return

            # Compute per-file metrics from MFA output TextGrids
            for item in files:
                stem       = Path(item["wav"]).stem
                mfa_tg     = output_dir / f"{stem}.TextGrid"
                info       = torchaudio.info(item["wav"])
                audio_dur  = info.num_frames / info.sample_rate

                if not mfa_tg.exists():
                    print(f"  ✗ No MFA output for {stem}")
                    continue

                pred = parse_mfa_textgrid(str(mfa_tg))
                ref  = parse_textgrid_raw(item["textgrid"], "chars")
                if not ref:
                    print(f"  ✗ No reference for {stem}")
                    continue

                # Attribute time proportionally
                elapsed = elapsed_total * (audio_dur / max(
                    sum(torchaudio.info(f["wav"]).num_frames / torchaudio.info(f["wav"]).sample_rate
                        for f in files), 1e-6))

                total_time_s  += elapsed
                total_audio_s += audio_dur
                fm = {"file": stem, "duration_s": round(audio_dur, 2)}
                fm.update(_collect_metrics(pred, ref, tolerances, all_m))
                fm["rtf"] = round(elapsed / audio_dur, 4)
                per_file.append(fm)

                print(f"  ✓ {stem:25s}  dur={audio_dur:.1f}s  "
                      f"F1@20ms={all_m[0.020]['F1'][-1]:.3f}  "
                      f"F1@40ms={all_m[0.040]['F1'][-1]:.3f}  "
                      f"MAE={all_m[0.020]['MAE'][-1]:.1f}ms")

    run_mfa_group(zh_files, MFA_ACOUSTIC_ZH, MFA_DICT_ZH, "Chinese")
    run_mfa_group(en_files, MFA_ACOUSTIC_EN, MFA_DICT_EN, "English")

    if not per_file:
        return None

    # MFA model sizes
    size_mb = (Path(MFA_ACOUSTIC_EN).stat().st_size + Path(MFA_ACOUSTIC_ZH).stat().st_size) / (1024**2)

    return _finalize(all_m, total_time_s, total_audio_s, "MFA", size_mb, per_file)


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Gentle (via HTTP API)
# ─────────────────────────────────────────────────────────────────────────────

def check_gentle_server() -> bool:
    """Return True if Gentle HTTP server is reachable."""
    try:
        r = requests.get("http://localhost:8765", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def align_with_gentle(wav_path: str, text: str) -> Tuple[List[Tuple[float, float, str]], float]:
    """
    POST to Gentle transcription endpoint.
    Returns (word_intervals, elapsed_s).
    Word-level intervals are used since Gentle does word alignment.
    """
    t0 = time.perf_counter()
    with open(wav_path, "rb") as f:
        response = requests.post(
            GENTLE_URL,
            files={"audio": f},
            data={"transcript": text},
            timeout=120,
        )
    elapsed = time.perf_counter() - t0
    response.raise_for_status()

    data = response.json()
    intervals = []
    for word in data.get("words", []):
        if word.get("case") == "success":
            start = word["start"]
            end   = word["end"]
            w     = word["word"]
            intervals.append((float(start), float(end), w))

    return intervals, elapsed


def benchmark_gentle(test_files: List[Dict], tolerances: List[float]) -> MethodResult:
    print(f"\n{'='*70}")
    print("  Method 3: Gentle")
    print(f"{'='*70}")

    if not check_gentle_server():
        print("  ✗ Gentle server not running on localhost:8765")
        print("    To start: git clone https://github.com/lowerquality/gentle")
        print("              cd gentle && bash install.sh && python serve.py &")
        return None

    print("  ✓ Gentle server is running")

    total_time_s = total_audio_s = 0.0
    per_file = []
    all_m = {t: {"P": [], "R": [], "F1": [], "MAE": []} for t in tolerances}

    for item in test_files:
        if is_chinese(item["text"]):
            print(f"  ⚠ Gentle is English-only, skipping {Path(item['wav']).stem}")
            continue

        info      = torchaudio.info(item["wav"])
        audio_dur = info.num_frames / info.sample_rate

        try:
            pred, elapsed = align_with_gentle(item["wav"], item["text"])
        except Exception as e:
            print(f"  ✗ {Path(item['wav']).stem}: {e}"); continue

        # Compare against word tier in reference (Gentle is word-level)
        ref = parse_textgrid_raw(item["textgrid"], "words")
        if not ref:
            ref = parse_textgrid_raw(item["textgrid"], "chars")
        if not ref:
            print(f"  ✗ No reference for {Path(item['wav']).stem}"); continue

        total_time_s  += elapsed
        total_audio_s += audio_dur
        fm = {"file": Path(item["wav"]).stem, "duration_s": round(audio_dur, 2)}
        fm.update(_collect_metrics(pred, ref, tolerances, all_m))
        fm["rtf"] = round(elapsed / audio_dur, 4)
        per_file.append(fm)

        print(f"  ✓ {Path(item['wav']).stem:25s}  dur={audio_dur:.1f}s  "
              f"F1@20ms={all_m[0.020]['F1'][-1]:.3f}  "
              f"F1@40ms={all_m[0.040]['F1'][-1]:.3f}  "
              f"MAE={all_m[0.020]['MAE'][-1]:.1f}ms  "
              f"RTF={elapsed/audio_dur:.4f}")

    if not per_file:
        return None

    return _finalize(all_m, total_time_s, total_audio_s, "Gentle", 0.0, per_file)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: List[MethodResult]):
    print(f"\n{'='*82}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*82}")
    hdr = (f"{'Method':<20} {'F1@20ms':>8} {'F1@40ms':>8} {'F1@80ms':>8} "
           f"{'MAE(ms)':>8} {'RTF':>8} {'Size(MB)':>9}")
    print(hdr)
    print("-" * 82)
    for r in results:
        mae_s  = f"{r.mae_ms:.1f}"  if r.mae_ms != float("inf") else "N/A"
        size_s = f"{r.model_size_mb:.0f}"
        print(f"{r.method:<20} {r.f1_20ms:>8.4f} {r.f1_40ms:>8.4f} {r.f1_80ms:>8.4f} "
              f"{mae_s:>8} {r.rtf:>8.4f} {size_s:>9}")
    print("-" * 82)

    if len(results) >= 2:
        tiny  = next((r for r in results if r.method == "TinyAligner"), None)
        if tiny:
            for other in results:
                if other.method == "TinyAligner":
                    continue
                f1_diff  = (tiny.f1_20ms - other.f1_20ms) * 100
                mae_diff = tiny.mae_ms - other.mae_ms
                rtf_ratio = other.rtf / tiny.rtf if tiny.rtf > 0 else float("inf")
                print(f"\n  TinyAligner vs {other.method}:")
                print(f"    F1@20ms  : {f1_diff:+.1f} pp  ({'TinyAligner better' if f1_diff>0 else other.method+' better'})")
                print(f"    MAE      : {mae_diff:+.1f} ms  ({'TinyAligner better' if mae_diff<0 else other.method+' better'})")
                print(f"    Speed    : TinyAligner is {rtf_ratio:.1f}x faster than {other.method}")
                if other.model_size_mb > 0:
                    print(f"    Size     : TinyAligner is {other.model_size_mb/tiny.model_size_mb:.0f}x smaller than {other.method}")


def generate_markdown_report(results: List[MethodResult], output_path: str):
    lines = [
        "# Forced Alignment Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Methods",
        "",
        "| Method | Description |",
        "|--------|-------------|",
        "| **TinyAligner** | Lightweight CTC model (~480K params) trained on AISHELL-1 + LibriSpeech |",
        "| **MFA** | Montreal Forced Aligner — Kaldi GMM/HMM, pre-trained `english_mfa` + `mandarin_mfa` |",
        "| **Gentle** | lowerquality/gentle — Kaldi-based, English word-level alignment via HTTP API |",
        "",
        "## Results",
        "",
        "| Method | F1@20ms | F1@40ms | F1@80ms | Prec@20ms | Rec@20ms | MAE(ms) | RTF | Size(MB) |",
        "|--------|---------|---------|---------|-----------|----------|---------|-----|---------|",
    ]
    for r in results:
        mae_s = f"{r.mae_ms:.1f}" if r.mae_ms != float("inf") else "N/A"
        lines.append(
            f"| {r.method} | {r.f1_20ms:.4f} | {r.f1_40ms:.4f} | {r.f1_80ms:.4f} "
            f"| {r.precision_20ms:.4f} | {r.recall_20ms:.4f} | {mae_s} "
            f"| {r.rtf:.4f} | {r.model_size_mb:.0f} |"
        )

    # Per-method comparison vs TinyAligner
    tiny = next((r for r in results if r.method == "TinyAligner"), None)
    if tiny and len(results) >= 2:
        lines += ["", "## Comparison vs TinyAligner", "",
                  "| Metric | " + " | ".join(r.method for r in results if r.method != "TinyAligner") + " |",
                  "|--------|" + "---|" * (len(results) - 1)]
        for metric, getter in [
            ("F1@20ms",  lambda r: f"{r.f1_20ms:.4f}"),
            ("F1@40ms",  lambda r: f"{r.f1_40ms:.4f}"),
            ("MAE (ms)", lambda r: f"{r.mae_ms:.1f}" if r.mae_ms != float("inf") else "N/A"),
            ("RTF",      lambda r: f"{r.rtf:.4f}"),
            ("Size(MB)", lambda r: f"{r.model_size_mb:.0f}"),
        ]:
            others = [r for r in results if r.method != "TinyAligner"]
            lines.append(f"| {metric} (TinyAligner: {getter(tiny)}) | " +
                         " | ".join(getter(r) for r in others) + " |")

    # Per-file
    lines += ["", "## Per-File Results", ""]
    for r in results:
        lines += [f"### {r.method}", "",
                  "| File | Duration | F1@20ms | F1@40ms | F1@80ms | RTF |",
                  "|------|----------|---------|---------|---------|-----|"]
        for f in r.per_file:
            lines.append(
                f"| {f['file']} | {f['duration_s']}s "
                f"| {f.get('f1_20ms','N/A')} | {f.get('f1_40ms','N/A')} "
                f"| {f.get('f1_80ms','N/A')} | {f.get('rtf','N/A')} |"
            )
        lines.append("")

    lines += [
        "## Metric Definitions",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| **F1@Xms** | Boundary F1 with ±X ms match tolerance |",
        "| **MAE** | Mean Absolute Error (ms) on matched boundary pairs |",
        "| **RTF** | Real-Time Factor = inference_time / audio_duration (lower = faster) |",
        "",
        "Boundaries compared are the **start-time** of each predicted segment.",
        "Greedy one-to-one matching within tolerance.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_files(audio_dir, textgrid_dir, text_dir):
    files = []
    for wav in sorted(Path(audio_dir).glob("*.wav")):
        tg  = Path(textgrid_dir) / f"{wav.stem}.TextGrid"
        txt = Path(text_dir) / f"{wav.stem}.txt"
        if not tg.exists():
            print(f"  No TextGrid for {wav.stem}, skipping"); continue
        if not txt.exists():
            print(f"  No transcript for {wav.stem}, skipping"); continue
        files.append({"wav": str(wav), "textgrid": str(tg),
                      "text": txt.read_text(encoding="utf-8").strip()})
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TinyAligner vs MFA vs Gentle"
    )
    parser.add_argument("--audio-dir",    default="test/audio")
    parser.add_argument("--textgrid-dir", default="test/textgrids")
    parser.add_argument("--text-dir",     default="test/text")
    parser.add_argument("--checkpoint",   default="checkpoints/best_model.pt")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--methods",      nargs="+", default=["tiny", "mfa", "gentle"],
                        choices=["tiny", "mfa", "gentle"])
    parser.add_argument("--tolerance",    nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--ref-tier",     default="auto",
                        choices=["auto", "chars", "words", "phones"],
                        help="TextGrid tier to use as reference. 'auto' detects: chars > words > phones")
    parser.add_argument("--output-json",  default="benchmark_results.json")
    parser.add_argument("--output-md",    default="benchmark_report.md")
    args = parser.parse_args()

    root         = Path(__file__).parent.parent
    audio_dir    = root / args.audio_dir
    textgrid_dir = root / args.textgrid_dir
    text_dir     = root / args.text_dir
    checkpoint   = root / args.checkpoint

    print(f"Device       : {args.device}")
    print(f"Audio dir    : {audio_dir}")
    print(f"TextGrid dir : {textgrid_dir}")
    print(f"Checkpoint   : {checkpoint}")

    test_files = collect_test_files(audio_dir, textgrid_dir, text_dir)
    if not test_files:
        print("\nNo test files found"); sys.exit(1)

    print(f"\nFound {len(test_files)} test file(s):")
    for f in test_files:
        lang = "ZH" if is_chinese(f["text"]) else "EN"
        print(f"  [{lang}] {Path(f['wav']).stem}: {f['text'][:65]}")

    # Auto-detect best available reference tier from the first TextGrid
    if args.ref_tier == "auto":
        sample_tg = test_files[0]["textgrid"]
        for candidate in ("chars", "words", "phones"):
            if parse_textgrid_raw(sample_tg, candidate):
                args.ref_tier = candidate
                break
        else:
            args.ref_tier = "words"
    print(f"  Reference tier : '{args.ref_tier}'")

    tolerances = sorted(set([t / 1000 for t in args.tolerance] + [0.020, 0.040, 0.080]))

    results: List[MethodResult] = []

    if "tiny" in args.methods:
        if not checkpoint.exists():
            print(f"\nCheckpoint not found: {checkpoint}\nTrain: python scripts/train.py")
        else:
            r = benchmark_tinyaligner(test_files, str(checkpoint), args.device, tolerances, args.ref_tier)
            if r: results.append(r)

    if "mfa" in args.methods:
        r = benchmark_mfa(test_files, tolerances)
        if r: results.append(r)

    if "gentle" in args.methods:
        r = benchmark_gentle(test_files, tolerances)
        if r: results.append(r)

    if not results:
        print("\nNo results"); sys.exit(1)

    print_summary_table(results)

    json_path = root / args.output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n  JSON : {json_path}")

    generate_markdown_report(results, str(root / args.output_md))


if __name__ == "__main__":
    main()
