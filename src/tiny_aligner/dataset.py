"""
Dataset loaders for bilingual (Chinese + English) forced alignment training.

Supported datasets:
  - AISHELL-1   (Chinese Mandarin)
  - LibriSpeech (English)

Produces:
  - 40-dim log Mel-filterbank features (16kHz, 25ms window, 10ms hop)
  - Phoneme target sequences (CTC labels)
"""

import os
import random
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from .lexicon import load_lexicon, build_phone_vocab, text_to_phones


# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS = 40          # keep small for iPhone
WIN_LENGTH = 400     # 25 ms
HOP_LENGTH = 160     # 10 ms  →  50ms = 5 frames
N_FFT = 512

# SpecAugment defaults (LB / "lightly augmented" preset, scaled for 40-dim mel)
SPECAUG_FREQ_MASK = 8     # max bins masked per freq mask
SPECAUG_TIME_MASK = 25    # max frames masked per time mask
SPECAUG_N_FREQ = 2
SPECAUG_N_TIME = 2

# Runtime curriculum knob: scales mask widths.
# train.py decays this in the last epochs so the model fine-tunes on cleaner mels.
_SPECAUG_INTENSITY = 1.0


def set_specaugment_intensity(scale: float) -> None:
    """Scale SpecAugment mask widths globally. 1.0 = full, 0.0 = disabled."""
    global _SPECAUG_INTENSITY
    _SPECAUG_INTENSITY = max(0.0, float(scale))


def _apply_specaugment(mel: torch.Tensor) -> torch.Tensor:
    """In-place-ish SpecAugment on a [T, n_mels] log-mel tensor.

    Applies n_freq frequency masks and n_time time masks with uniform widths.
    Masked positions are set to the per-utterance mean (post-CMVN this is 0,
    but we use mean for robustness in case normalisation order changes).
    """
    if _SPECAUG_INTENSITY <= 0.0:
        return mel

    T, F = mel.shape
    fill = mel.mean()
    freq_max = max(0, int(round(SPECAUG_FREQ_MASK * _SPECAUG_INTENSITY)))
    time_max = max(0, int(round(SPECAUG_TIME_MASK * _SPECAUG_INTENSITY)))

    for _ in range(SPECAUG_N_FREQ):
        if freq_max <= 0:
            break
        f = int(torch.randint(0, freq_max + 1, (1,)).item())
        if f and f < F:
            f0 = int(torch.randint(0, F - f + 1, (1,)).item())
            mel[:, f0:f0 + f] = fill

    for _ in range(SPECAUG_N_TIME):
        if time_max <= 0:
            break
        # Cap time mask at 20% of utterance — LibriSpeech-style proportional cap
        max_t = min(time_max, max(1, T // 5))
        t = int(torch.randint(0, max_t + 1, (1,)).item())
        if t and t < T:
            t0 = int(torch.randint(0, T - t + 1, (1,)).item())
            mel[t0:t0 + t, :] = fill

    return mel


class AishellDataset(Dataset):
    """
    Loads AISHELL WAV files + transcript.

    Args:
        data_root: path to data_aishell/ (contains wav/ and transcript/)
        lexicon: loaded lexicon dict
        phone2idx: phoneme → index mapping
        split: 'train' | 'dev' | 'test'
        max_samples: limit dataset size (for quick experiments)
        max_duration_s: skip utterances longer than this (reduce memory)
    """

    def __init__(
        self,
        data_root: str,
        lexicon: dict,
        phone2idx: dict,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_duration_s: float = 15.0,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.lexicon = lexicon
        self.phone2idx = phone2idx
        self.max_duration_s = max_duration_s
        self.augment = augment

        # Mel-filterbank extractor
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=80.0,
            f_max=7600.0,
        )

        # Load transcript
        transcript_file = self.data_root / "transcript" / "aishell_transcript_v0.8.txt"
        self.transcript = self._load_transcript(transcript_file)

        # Build sample list (only utterances with existing wav files)
        wav_dir = self.data_root / "wav" / split
        self.samples = self._collect_samples(wav_dir)

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"[Dataset] {split}: {len(self.samples)} samples loaded")

    def _load_transcript(self, path: Path) -> dict[str, str]:
        trans = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    trans[parts[0]] = parts[1]
        return trans

    def _collect_samples(self, wav_dir: Path) -> list[dict]:
        samples = []
        skipped_too_short = 0

        for wav_path in wav_dir.rglob("*.wav"):
            utt_id = wav_path.stem
            if utt_id not in self.transcript:
                continue
            text = self.transcript[utt_id]
            phones, phone_ids = text_to_phones(text, self.lexicon, self.phone2idx)
            if len(phone_ids) == 0:
                continue
            # CTC feasibility: model downsamples 2×, then needs T' >= L for blank+token interleave.
            # Use file size as cheap upper bound on samples; one mel frame = 160 audio samples.
            try:
                wav_bytes = wav_path.stat().st_size
            except OSError:
                continue
            est_frames = max(1, (wav_bytes - 44) // 2) // HOP_LENGTH
            if est_frames // 2 < len(phone_ids) + 1:
                skipped_too_short += 1
                continue
            samples.append({
                "utt_id": utt_id,
                "wav_path": str(wav_path),
                "text": text,
                "phones": phones,
                "phone_ids": phone_ids,
            })

        if skipped_too_short:
            print(f"[Dataset] AISHELL: skipped {skipped_too_short} CTC-infeasible samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load audio
        waveform, sr = torchaudio.load(sample["wav_path"])
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0, keepdim=True)  # mono

        # Mel features: [1, n_mels, T] → [T, n_mels]
        mel = self.mel(waveform)
        mel = (mel + 1e-6).log()
        mel = mel.squeeze(0).T  # [T, n_mels]

        # Normalize per utterance
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        if self.augment:
            mel = _apply_specaugment(mel)

        phone_ids = torch.tensor(sample["phone_ids"], dtype=torch.long)

        return {
            "mel": mel,               # [T, 40]
            "phone_ids": phone_ids,   # [P]
            "utt_id": sample["utt_id"],
            "text": sample["text"],
            "phones": sample["phones"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad mel and phone sequences for batch training."""
    mels = [b["mel"] for b in batch]
    phone_ids = [b["phone_ids"] for b in batch]

    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    phone_lengths = torch.tensor([p.shape[0] for p in phone_ids], dtype=torch.long)

    # Pad
    mels_padded = pad_sequence(mels, batch_first=True)          # [B, T, 40]
    phones_padded = pad_sequence(phone_ids, batch_first=True)   # [B, P]

    return {
        "mel": mels_padded,
        "mel_lengths": mel_lengths,
        "phone_ids": phones_padded,
        "phone_lengths": phone_lengths,
        "utt_ids": [b["utt_id"] for b in batch],
        "texts": [b["text"] for b in batch],
        "phones": [b["phones"] for b in batch],
    }


def get_dataloaders(
    data_root: str,
    lexicon: dict,
    phone2idx: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    librispeech_root: Optional[str] = None,
    librispeech_splits: list[str] = ("train-clean-100",),
    balanced_sampling: bool = True,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_dl, dev_dl).

    Training:   AISHELL-train + LibriSpeech train splits.
                With balanced_sampling=True a WeightedRandomSampler draws ZH and
                EN with equal expected probability so each batch is ~50/50.
                With augment_train=True SpecAugment is applied to training mels.
    Validation: AISHELL-dev  + LibriSpeech dev-clean (combined, no aug)
    """
    # ── Chinese (AISHELL) ────────────────────────────────────────────────────
    zh_train = AishellDataset(data_root, lexicon, phone2idx, split="train",
                              max_samples=max_train_samples,
                              augment=augment_train)
    zh_dev   = AishellDataset(data_root, lexicon, phone2idx, split="dev",
                              max_samples=500)

    if librispeech_root:
        # ── Training: all splits concatenated ───────────────────────────────
        en_train_sets = [
            LibriSpeechDataset(librispeech_root, lexicon, phone2idx, split=s,
                               augment=augment_train)
            for s in librispeech_splits
        ]
        en_train = ConcatDataset(en_train_sets)
        combined_train = ConcatDataset([zh_train, en_train])
        print(f"[Dataset] Train: {len(zh_train):,} ZH + {len(en_train):,} EN "
              f"= {len(combined_train):,} total")

        # Balanced sampler: weight each sample inversely to its dataset size.
        # Expected per-batch ratio is 50/50 regardless of dataset cardinality.
        n_zh, n_en = len(zh_train), len(en_train)
        if balanced_sampling and n_zh > 0 and n_en > 0:
            w_zh = 0.5 / n_zh
            w_en = 0.5 / n_en
            weights = ([w_zh] * n_zh) + ([w_en] * n_en)
            # One epoch == one pass over the larger half; bilingual coverage
            # comes from the weighted draws.
            num_samples = 2 * max(n_zh, n_en)
            sampler = WeightedRandomSampler(weights, num_samples=num_samples,
                                            replacement=True)
            print(f"[Dataset] Balanced sampler: w_zh={w_zh:.2e} w_en={w_en:.2e} "
                  f"draws/epoch={num_samples:,}")
            train_dl = DataLoader(
                combined_train, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, collate_fn=collate_fn,
                pin_memory=True, drop_last=True,
            )
        else:
            train_dl = DataLoader(
                combined_train, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, collate_fn=collate_fn,
                pin_memory=True, drop_last=True,
            )

        # ── Validation: AISHELL-dev + LibriSpeech dev-clean ─────────────────
        en_dev = LibriSpeechDataset(librispeech_root, lexicon, phone2idx,
                                    split="dev-clean", max_samples=500)
        combined_dev = ConcatDataset([zh_dev, en_dev])
        print(f"[Dataset] Dev:   {len(zh_dev):,} ZH + {len(en_dev):,} EN "
              f"= {len(combined_dev):,} total")

        dev_dl = DataLoader(
            combined_dev, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        train_dl = DataLoader(
            zh_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
        dev_dl = DataLoader(
            zh_dev, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_dl, dev_dl


# ── LibriSpeech ───────────────────────────────────────────────────────────────

class LibriSpeechDataset(Dataset):
    """
    Wraps torchaudio's built-in LibriSpeech loader.

    LibriSpeech transcripts are space-separated UPPERCASE English words, e.g.:
        "HE WAS NOT GOING TO DO IT"
    text_to_phones looks up each token in the merged lexicon (CMUdict keys are
    stored uppercase in the merged lexicon produced by export_onnx.py).

    Args:
        root:       path that contains LibriSpeech/ folder (passed to torchaudio)
        lexicon:    merged bilingual lexicon dict
        phone2idx:  phoneme → index mapping (bilingual)
        split:      e.g. 'train-clean-100', 'train-clean-360', 'dev-clean'
        max_samples: cap dataset size
        max_duration_s: skip long utterances
        download:   pass True to auto-download (not recommended for large splits)
    """

    def __init__(
        self,
        root: str,
        lexicon: dict,
        phone2idx: dict,
        split: str = "train-clean-100",
        max_samples: Optional[int] = None,
        max_duration_s: float = 15.0,
        download: bool = False,
        augment: bool = False,
    ):
        self.lexicon   = lexicon
        self.phone2idx = phone2idx
        self.max_duration_s = max_duration_s
        self.augment = augment

        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=80.0,
            f_max=7600.0,
        )

        raw = self._open_librispeech(root, split, download)
        self._raw = raw
        self.samples = self._build_samples(raw, max_samples)
        print(f"[Dataset] LibriSpeech/{split}: {len(self.samples):,} samples loaded")

    @staticmethod
    def _open_librispeech(root: str, split: str, download: bool):
        """Open the LIBRISPEECH dataset, tolerating two on-disk layouts:

        Standard:    <root>/LibriSpeech/<split>/<speaker>/<chapter>/<utt>.flac
        Nested:      <root>/LibriSpeech/<split>/<split>/<speaker>/<chapter>/<utt>.flac
                     (happens when the tarball was extracted inside its own
                      split folder — yields 0 samples without this fallback)

        In the nested case we repoint the torchaudio dataset's internal
        ``_archive`` to the nested directory and repopulate ``_walker`` with
        the utterance ids it expects (``<speaker>-<chapter>-<utt>``).
        """
        ds = torchaudio.datasets.LIBRISPEECH(root, url=split, download=download)
        if len(ds) > 0:
            return ds
        nested = Path(root) / "LibriSpeech" / split / split
        if not nested.is_dir():
            return ds  # nothing more we can do
        # torchaudio resolves files as <_archive>/<_url>/<speaker>/<chapter>/<file>.
        # Point _archive at the nested dir and clear _url so paths come out right.
        ds._archive = str(nested)
        ds._url = ""
        import glob
        walker = []
        for flac in glob.glob(str(nested / "*/*/*.flac")):
            walker.append(Path(flac).stem)  # "6313-66129-0001"
        ds._walker = sorted(walker)
        return ds

    def _build_samples(self, raw_ds, max_samples):
        samples = []
        skipped_too_short = 0
        max_frames = int(self.max_duration_s * SAMPLE_RATE)
        indices = list(range(len(raw_ds)))
        random.shuffle(indices)
        for i in indices:
            waveform, sr, transcript, *_ = raw_ds[i]
            if waveform.shape[-1] > max_frames:
                continue
            text = transcript.upper()
            phones, phone_ids = text_to_phones(
                text, self.lexicon, self.phone2idx, add_sil=True
            )
            if len(phone_ids) < 2:
                continue
            # CTC feasibility: model downsamples 2×, need T/2 >= len(phones)+1.
            est_frames = waveform.shape[-1] // HOP_LENGTH
            if est_frames // 2 < len(phone_ids) + 1:
                skipped_too_short += 1
                continue
            samples.append({
                "raw_idx":   i,
                "text":      text,
                "phones":    phones,
                "phone_ids": phone_ids,
            })
            if max_samples and len(samples) >= max_samples:
                break
        if skipped_too_short:
            print(f"[Dataset] LibriSpeech: skipped {skipped_too_short} CTC-infeasible samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        waveform, sr, *_ = self._raw[s["raw_idx"]]
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0, keepdim=True)

        mel = self.mel_transform(waveform)
        mel = (mel + 1e-6).log()
        mel = mel.squeeze(0).T  # [T, n_mels]
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        if self.augment:
            mel = _apply_specaugment(mel)

        return {
            "mel":       mel,
            "phone_ids": torch.tensor(s["phone_ids"], dtype=torch.long),
            "utt_id":    f"libri_{s['raw_idx']}",
            "text":      s["text"],
            "phones":    s["phones"],
        }


if __name__ == "__main__":
    from .lexicon import merge_lexicons, build_phone_vocab
    from pathlib import Path
    LEXICON_PATH     = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
    CMUDICT_PATH     = Path(__file__).resolve().parent / "cmudict-0.7b"
    LIBRISPEECH_ROOT = "/home/mani/Documents/forced-aligner/data"

    lexicon = merge_lexicons(LEXICON_PATH, CMUDICT_PATH)
    phone2idx, idx2phone = build_phone_vocab(lexicon)

    print("=== AISHELL sample ===")
    zh_ds = AishellDataset(DATA_ROOT, lexicon, phone2idx, split="train", max_samples=5)
    s = zh_ds[0]
    print(f"utt_id : {s['utt_id']}")
    print(f"text   : {s['text']}")
    print(f"phones : {s['phones'][:10]} ...")
    print(f"mel    : {s['mel'].shape}")

    print("\n=== LibriSpeech sample ===")
    en_ds = LibriSpeechDataset(LIBRISPEECH_ROOT, lexicon, phone2idx,
                               split="train-clean-100", max_samples=5)
    s = en_ds[0]
    print(f"utt_id : {s['utt_id']}")
    print(f"text   : {s['text'][:60]}")
    print(f"phones : {s['phones'][:10]} ...")
    print(f"mel    : {s['mel'].shape}")
