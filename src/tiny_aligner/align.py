"""
Forced alignment engine + Praat TextGrid generation.

Given:
  - Audio file (WAV)
  - Chinese text (space-separated words or raw characters)
  - Trained TinyAligner model

Produces:
  - Frame-level phoneme alignment (Viterbi on CTC emissions)
  - Praat TextGrid with:
      Tier 1: "phones"     - phoneme boundaries
      Tier 2: "words"      - word boundaries  
      Tier 3: "chars"      - character boundaries

Algorithm: CTC-based forced alignment via log-space Viterbi
  - Build forced path: interleave blanks between phonemes
  - Run forward-backward or Viterbi to get optimal frame→phoneme mapping
  - This gives 20ms resolution (10ms hop × 2 conv stride)
  - Well under 50ms target

Reference: Kürzinger et al. 2020 "CTC-Segmentation"
"""

import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path

from .lexicon import (
    load_lexicon, build_phone_vocab, text_to_phones,
    text_to_chars, phones_to_char_boundaries
)
from .model import TinyAligner
from .dataset import SAMPLE_RATE, N_MELS, WIN_LENGTH, HOP_LENGTH, N_FFT


# ── Frame timing ───────────────────────────────────────────────────────────────
FRAME_SHIFT_MS = 10.0          # mel hop = 10ms
DOWNSAMPLE_FACTOR = 2          # conv stride 2×
OUTPUT_FRAME_MS = FRAME_SHIFT_MS * DOWNSAMPLE_FACTOR  # 20ms per output frame


# ── TextGrid writer ────────────────────────────────────────────────────────────

def write_textgrid(
    path: str,
    tiers: list[dict],
    total_duration: float,
):
    """
    Write a Praat TextGrid file.
    
    Args:
        path: output .TextGrid file path
        tiers: list of dicts with keys:
               - name: tier name
               - intervals: list of (start_s, end_s, label) tuples
        total_duration: total audio duration in seconds
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f'xmin = 0\n')
        f.write(f'xmax = {total_duration:.6f}\n')
        f.write('tiers? <exists>\n')
        f.write(f'size = {len(tiers)}\n')
        f.write('item []:\n')

        for ti, tier in enumerate(tiers, 1):
            intervals = tier["intervals"]
            f.write(f'    item [{ti}]:\n')
            f.write(f'        class = "IntervalTier"\n')
            f.write(f'        name = "{tier["name"]}"\n')
            f.write(f'        xmin = 0\n')
            f.write(f'        xmax = {total_duration:.6f}\n')
            f.write(f'        intervals: size = {len(intervals)}\n')
            for ii, (start, end, label) in enumerate(intervals, 1):
                f.write(f'        intervals [{ii}]:\n')
                f.write(f'            xmin = {start:.6f}\n')
                f.write(f'            xmax = {end:.6f}\n')
                f.write(f'            text = "{label}"\n')


# ── Audio feature extraction ───────────────────────────────────────────────────

def load_audio(wav_path: str) -> tuple[torch.Tensor, float]:
    """Load WAV and return (mel_features [T, n_mels], duration_s)."""
    mel_extractor = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=80.0,
        f_max=7600.0,
    )

    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(0, keepdim=True)

    duration_s = waveform.shape[-1] / SAMPLE_RATE

    mel = mel_extractor(waveform)      # [1, n_mels, T]
    mel = (mel + 1e-6).log()
    mel = mel.squeeze(0).T             # [T, n_mels]
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)

    return mel, duration_s


# ── CTC Forced Alignment (Viterbi) ─────────────────────────────────────────────

def ctc_forced_align(
    log_probs: np.ndarray,   # [T, n_phones]
    phone_ids: list[int],
    blank_idx: int = 0,
) -> list[int]:
    """
    Viterbi forced alignment using CTC blank-interleaved path.
    
    Builds the forced path:
        blank, p[0], blank, p[1], blank, ..., p[N-1], blank
    
    Runs Viterbi to find the optimal frame assignment.
    
    Returns: list of length T, each element is the index into phone_ids
             (or -1 for blank frames).
    """
    T, V = log_probs.shape

    # Build forced CTC path: blank between every pair of phones
    # [blank, p0, blank, p1, blank, ..., pN, blank]
    forced_path = [blank_idx]
    for p in phone_ids:
        forced_path.append(p)
        forced_path.append(blank_idx)
    S = len(forced_path)  # 2N+1

    NEG_INF = -1e9

    # ── Forward pass ────────────────────────────────────────────────────────
    # alpha[s] = log-prob of being at state s at current frame
    alpha = np.full(S, NEG_INF, dtype=np.float64)
    alpha[0] = log_probs[0, blank_idx]
    if S > 1:
        alpha[1] = log_probs[0, forced_path[1]]

    # Track back-pointers
    bp = np.zeros((T, S), dtype=np.int32)

    for t in range(1, T):
        new_alpha = np.full(S, NEG_INF, dtype=np.float64)
        for s in range(S):
            # Can come from: s (stay), s-1, or s-2 (skip blank if same token)
            candidates = [s]
            if s > 0:
                candidates.append(s - 1)
            if s > 1 and forced_path[s] != blank_idx and forced_path[s] != forced_path[s - 2]:
                candidates.append(s - 2)

            best_prev = candidates[0]
            best_val = alpha[candidates[0]]
            for c in candidates[1:]:
                if alpha[c] > best_val:
                    best_val = alpha[c]
                    best_prev = c

            new_alpha[s] = best_val + log_probs[t, forced_path[s]]
            bp[t, s] = best_prev

        alpha = new_alpha

    # ── Backtrace ─────────────────────────────────────────────────────────
    # End at last or second-to-last state
    end_state = S - 1 if alpha[S - 1] > alpha[S - 2] else S - 2

    path = []
    s = end_state
    for t in range(T - 1, -1, -1):
        path.append(s)
        s = bp[t, s]
    path.reverse()

    # Map path states to phone indices in phone_ids list
    # State s: if s is even → blank, if s is odd → phone at (s-1)//2
    frame_phone_idx = []
    for s in path:
        if s % 2 == 0:
            frame_phone_idx.append(-1)    # blank
        else:
            frame_phone_idx.append((s - 1) // 2)  # index into phone_ids

    return frame_phone_idx


def extract_segments(
    frame_phone_idx: list[int],
    phones: list[str],
    frame_shift_s: float,
) -> list[tuple[float, float, str]]:
    """
    Convert frame-level phone assignments to (start_s, end_s, phone) segments.
    Merges consecutive frames with same phone label.
    """
    if not frame_phone_idx:
        return []

    segments = []
    prev_idx = frame_phone_idx[0]
    start_frame = 0

    for t, idx in enumerate(frame_phone_idx[1:], 1):
        if idx != prev_idx:
            label = phones[prev_idx] if prev_idx >= 0 else ""
            segments.append((
                start_frame * frame_shift_s,
                t * frame_shift_s,
                label,
            ))
            start_frame = t
            prev_idx = idx

    # Last segment
    label = phones[prev_idx] if prev_idx >= 0 else ""
    segments.append((
        start_frame * frame_shift_s,
        len(frame_phone_idx) * frame_shift_s,
        label,
    ))

    return segments


# ── Main alignment function ────────────────────────────────────────────────────

def align(
    model: TinyAligner,
    wav_path: str,
    text: str,
    lexicon: dict,
    phone2idx: dict,
    idx2phone: dict,
    output_textgrid: str = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run forced alignment on a single utterance.
    
    Args:
        model: trained TinyAligner
        wav_path: path to WAV file
        text: Chinese text (space-separated words or raw chars)
        lexicon: loaded lexicon
        phone2idx: phoneme → index
        idx2phone: index → phoneme
        output_textgrid: if set, write TextGrid to this path
        device: inference device
    
    Returns dict with:
        phones: list of phones
        phone_intervals: [(start_s, end_s, phone), ...]
        char_intervals: [(start_s, end_s, char), ...]
        word_intervals: [(start_s, end_s, word), ...]
        duration: audio duration in seconds
    """
    blank_idx = phone2idx["<blank>"]

    # 1. Load audio & extract features
    mel, duration_s = load_audio(wav_path)
    mel_tensor = mel.unsqueeze(0).to(device)  # [1, T, C]

    # 2. Get model emissions
    model.eval()
    log_probs_tensor = model.get_emissions(mel_tensor)  # [T', n_phones]
    log_probs = log_probs_tensor.cpu().numpy()

    # 3. Text → phonemes
    phones, phone_ids = text_to_phones(text, lexicon, phone2idx, add_sil=True)
    frame_shift_s = OUTPUT_FRAME_MS / 1000.0

    # 4. Viterbi forced alignment
    frame_phone_idx = ctc_forced_align(log_probs, phone_ids, blank_idx)

    # 5. Extract phone segments
    phone_intervals = extract_segments(frame_phone_idx, phones, frame_shift_s)
    # Remove empty/blank intervals for cleaner output
    phone_intervals_clean = [(s, e, lb) for s, e, lb in phone_intervals if lb]

    # 6. Build word-level segments
    #    Text words → phone count → map phone segments to words
    text_clean = text.strip()
    tokens = text_clean.split() if " " in text_clean else list(text_clean)
    word_intervals = _build_word_intervals(
        tokens, phones, phone_intervals_clean, lexicon
    )

    # 7. Build character-level segments
    char_intervals = _build_char_intervals(
        tokens, phones, phone_intervals_clean, lexicon
    )

    result = {
        "phones": phones,
        "phone_intervals": phone_intervals_clean,
        "word_intervals": word_intervals,
        "char_intervals": char_intervals,
        "duration": duration_s,
    }

    # 8. Write TextGrid
    if output_textgrid:
        # Ensure last interval reaches audio end
        def pad_to_end(intervals, end):
            if not intervals:
                return [(0, end, "")]
            last = intervals[-1]
            if last[1] < end:
                intervals = intervals + [(last[1], end, "")]
            return intervals

        phone_tier = pad_to_end(phone_intervals_clean, duration_s)
        word_tier = pad_to_end(word_intervals, duration_s)
        char_tier = pad_to_end(char_intervals, duration_s)

        write_textgrid(
            output_textgrid,
            tiers=[
                {"name": "phones", "intervals": phone_tier},
                {"name": "words",  "intervals": word_tier},
                {"name": "chars",  "intervals": char_tier},
            ],
            total_duration=duration_s,
        )
        print(f"[Align] TextGrid written: {output_textgrid}")

    return result


def _build_word_intervals(tokens, phones, phone_intervals, lexicon):
    """Map phone intervals back to word-level boundaries."""
    word_intervals = []
    phone_offset = 1  # skip leading SIL

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in lexicon:
            n_ph = len(lexicon[token][0])
        else:
            n_ph = sum(len(lexicon[ch][0]) if ch in lexicon else 1 for ch in token)

        # Find start/end time from phone_intervals
        start_idx = phone_offset - 1
        end_idx = phone_offset + n_ph - 1

        t_start = _get_start_time(phone_intervals, start_idx)
        t_end = _get_end_time(phone_intervals, end_idx)

        word_intervals.append((t_start, t_end, token))
        phone_offset += n_ph

    return word_intervals


def _build_char_intervals(tokens, phones, phone_intervals, lexicon):
    """Map phone intervals back to character-level boundaries."""
    char_intervals = []
    phone_offset = 1  # skip leading SIL

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in lexicon:
            word_phones = lexicon[token][0]
            n_chars = len(token)
            if n_chars == 1:
                t_start = _get_start_time(phone_intervals, phone_offset - 1)
                t_end = _get_end_time(phone_intervals, phone_offset + len(word_phones) - 2)
                char_intervals.append((t_start, t_end, token))
                phone_offset += len(word_phones)
            else:
                # Distribute phones across characters
                phones_per_char = len(word_phones) // n_chars
                remainder = len(word_phones) % n_chars
                offset = phone_offset
                for i, ch in enumerate(token):
                    n = phones_per_char + (1 if i < remainder else 0)
                    t_start = _get_start_time(phone_intervals, offset - 1)
                    t_end = _get_end_time(phone_intervals, offset + n - 2)
                    char_intervals.append((t_start, t_end, ch))
                    offset += n
                phone_offset = offset
        else:
            for ch in token:
                if ch in lexicon:
                    n_ph = len(lexicon[ch][0])
                else:
                    n_ph = 1
                t_start = _get_start_time(phone_intervals, phone_offset - 1)
                t_end = _get_end_time(phone_intervals, phone_offset + n_ph - 2)
                char_intervals.append((t_start, t_end, ch))
                phone_offset += n_ph

    return char_intervals


def _get_start_time(phone_intervals, idx):
    if idx < 0 or idx >= len(phone_intervals):
        return 0.0
    return phone_intervals[idx][0]


def _get_end_time(phone_intervals, idx):
    if idx < 0:
        return 0.0
    idx = min(idx, len(phone_intervals) - 1)
    return phone_intervals[idx][1]


# ── CLI ────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model + vocab from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    phone2idx = ckpt["phone2idx"]
    idx2phone = {int(k): v for k, v in ckpt["idx2phone"].items()}

    model = TinyAligner(n_phones=cfg["n_phones"], hidden=cfg["hidden"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, phone2idx, idx2phone


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run forced alignment")
    parser.add_argument("--wav", required=True, help="Input WAV file")
    parser.add_argument("--text", required=True, help="Chinese transcript")
    parser.add_argument("--model", default="checkpoints/best_model.pt")
    parser.add_argument("--lexicon", default="/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt")
    parser.add_argument("--out", default="output.TextGrid", help="Output TextGrid path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lexicon = load_lexicon(args.lexicon)
    model, phone2idx, idx2phone = load_model(args.model, device)

    result = align(
        model, args.wav, args.text, lexicon, phone2idx, idx2phone,
        output_textgrid=args.out, device=device,
    )

    print(f"\nDuration: {result['duration']:.3f}s")
    print(f"\nPhone intervals ({len(result['phone_intervals'])}):")
    for s, e, lb in result["phone_intervals"][:15]:
        print(f"  {s:.3f}s – {e:.3f}s  {lb}")

    print(f"\nCharacter intervals ({len(result['char_intervals'])}):")
    for s, e, lb in result["char_intervals"][:10]:
        print(f"  {s:.3f}s – {e:.3f}s  {lb}")
