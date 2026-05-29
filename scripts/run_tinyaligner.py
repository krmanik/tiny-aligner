#!/usr/bin/env python3
"""
Run TinyAligner forced alignment on a directory of audio+transcript pairs
and save one TextGrid per file.

Each TextGrid has three tiers:
  - phones  : phoneme-level boundaries
  - words   : word-level boundaries
  - chars   : character-level boundaries

Usage:
    conda activate bfa
    python scripts/run_tinyaligner.py                          # test/ files
    python scripts/run_tinyaligner.py \\
        --audio-dir  path/to/audio \\
        --text-dir   path/to/transcripts \\
        --output-dir path/to/output \\
        --checkpoint checkpoints/best_model.pt \\
        --device     cuda
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────────
AISHELL_LEXICON = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
CMUDICT_PATH    = str(Path(__file__).parent.parent / "data" / "lexicons" / "cmudict-0.7b")


def load_model(checkpoint_path: str, device: str):
    from tiny_aligner.model import TinyAligner
    from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "phone2idx" in ckpt:
        phone2idx = ckpt["phone2idx"]
        idx2phone = {int(k): v for k, v in ckpt["idx2phone"].items()}
        cfg   = ckpt.get("config", {})
        model = TinyAligner(
            n_phones=cfg.get("n_phones", len(phone2idx)),
            hidden=cfg.get("hidden", 128),
        )
        model.load_state_dict(ckpt["model"])
    else:
        from tiny_aligner.model import build_model
        lexicon   = merge_lexicons(AISHELL_LEXICON, CMUDICT_PATH)
        phone2idx, idx2phone = build_phone_vocab(lexicon)
        model     = build_model(len(phone2idx))
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.to(device).eval()
    lexicon = merge_lexicons(AISHELL_LEXICON, CMUDICT_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb  = n_params * 4 / (1024 ** 2)
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Parameters: {n_params:,}  |  Size: {size_mb:.1f} MB  |  Device: {device}")
    return model, lexicon, phone2idx, idx2phone


def run(args):
    from tiny_aligner.align import align

    root         = Path(__file__).parent.parent
    audio_dir    = Path(args.audio_dir)  if Path(args.audio_dir).is_absolute()  else root / args.audio_dir
    text_dir     = Path(args.text_dir)   if Path(args.text_dir).is_absolute()   else root / args.text_dir
    output_dir   = Path(args.output_dir) if Path(args.output_dir).is_absolute() else root / args.output_dir
    checkpoint   = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else root / args.checkpoint

    print("=" * 70)
    print("  TinyAligner — batch TextGrid generation")
    print("=" * 70)
    print(f"  Audio dir  : {audio_dir}")
    print(f"  Text dir   : {text_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Checkpoint : {checkpoint}")
    print()

    if not checkpoint.exists():
        print(f"  ✗ Checkpoint not found: {checkpoint}")
        print("    Train first: python scripts/train.py")
        sys.exit(1)

    if not audio_dir.exists():
        print(f"  ✗ Audio dir not found: {audio_dir}"); sys.exit(1)
    if not text_dir.exists():
        print(f"  ✗ Text dir not found: {text_dir}"); sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect files ──────────────────────────────────────────────────────────
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"  ✗ No .wav files found in {audio_dir}"); sys.exit(1)

    items = []
    for wav in wav_files:
        txt = text_dir / f"{wav.stem}.txt"
        if not txt.exists():
            print(f"  ⚠  No transcript for {wav.stem}, skipping")
            continue
        items.append({"wav": str(wav), "text": txt.read_text(encoding="utf-8").strip()})

    if not items:
        print("  ✗ No valid audio+transcript pairs found"); sys.exit(1)

    print(f"  Found {len(items)} file(s):")
    for it in items:
        print(f"    {Path(it['wav']).stem}: {it['text'][:70]}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────────
    device = args.device
    model, lexicon, phone2idx, idx2phone = load_model(str(checkpoint), device)
    print()

    # ── Align each file ────────────────────────────────────────────────────────
    ok = failed = 0
    total_audio_s = total_time_s = 0.0

    for it in items:
        stem     = Path(it["wav"]).stem
        tg_path  = str(output_dir / f"{stem}.TextGrid")
        info     = torchaudio.info(it["wav"])
        duration = info.num_frames / info.sample_rate

        try:
            t0 = time.perf_counter()
            result = align(
                model=model,
                wav_path=it["wav"],
                text=it["text"],
                lexicon=lexicon,
                phone2idx=phone2idx,
                idx2phone=idx2phone,
                output_textgrid=tg_path,
                device=torch.device(device),
            )
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  ✗ {stem}: {e}")
            failed += 1
            continue

        total_audio_s += duration
        total_time_s  += elapsed
        rtf = elapsed / max(duration, 1e-6)

        n_phones = len(result.get("phone_intervals", []))
        n_words  = len(result.get("word_intervals",  []))
        n_chars  = len(result.get("char_intervals",  []))

        print(f"  ✓ {stem:25s}  {duration:.2f}s audio  "
              f"RTF={rtf:.4f}  "
              f"phones={n_phones}  words={n_words}  chars={n_chars}")
        print(f"    → {tg_path}")

        # Print first few char intervals if verbose
        if args.verbose and result.get("char_intervals"):
            for s, e, ch in result["char_intervals"][:8]:
                print(f"       {ch}  {s:.3f}s → {e:.3f}s")
            if len(result["char_intervals"]) > 8:
                print(f"       … ({len(result['char_intervals'])} total)")

        ok += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  Done.  ✓ {ok} succeeded   ✗ {failed} failed")
    if total_audio_s > 0:
        overall_rtf = total_time_s / total_audio_s
        print(f"  Total audio : {total_audio_s:.1f}s")
        print(f"  Total time  : {total_time_s:.2f}s")
        print(f"  Overall RTF : {overall_rtf:.4f}  "
              f"({'%.1f' % (1/overall_rtf)}x real-time)")
    print(f"  TextGrids saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="TinyAligner batch TextGrid generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir",    default="test/audio",
                        help="Directory containing .wav files")
    parser.add_argument("--text-dir",     default="test/text",
                        help="Directory containing .txt transcripts (one per wav)")
    parser.add_argument("--output-dir",   default="test/tiny_textgrids",
                        help="Directory to write TextGrid files")
    parser.add_argument("--checkpoint",   default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="Inference device")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print first few char intervals per file")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
