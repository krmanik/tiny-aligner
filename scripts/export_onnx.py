#!/usr/bin/env python3
"""
Export TinyAligner checkpoint → ONNX + lexicon JSON for fully client-side browser inference.

Outputs to static/ directory:
  static/model.onnx       – ONNX model (get_emissions wrapper, input [1,T,40])
  static/lexicon.json     – {lexicon, phone2idx, idx2phone}
  static/manifest.json    – list of test file pairs
  static/test/audio/*.wav – test audio files
  static/test/text/*.txt  – test text files

Then serve with:
  python -m http.server 8080 --directory static
  Open http://localhost:8080
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiny_aligner.model import build_model
from tiny_aligner.lexicon import merge_lexicons

CHECKPOINT   = ROOT.parent / "checkpoints" / "best_model.pt"
LEXICON_PATH = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
CMUDICT_PATH = ROOT.parent / "data" / "lexicons" / "cmudict-0.7b"
STATIC_DIR   = ROOT.parent / "browser/public"
TEST_DIR     = ROOT / "test"


# ── Model export ───────────────────────────────────────────────────────────────

class EmissionsWrapper(nn.Module):
    """
    Wraps get_emissions for clean ONNX export.
    Input:  mel  [1, T, 40]
    Output: log_probs [T', n_phones]
    (No pack_padded_sequence — clean graph for ONNX)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):          # [1, T, 40]
        return self.model.get_emissions(mel)  # [T', n_phones]


def export_model(checkpoint: Path, out_dir: Path) -> tuple[dict, dict]:
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    phone2idx = ckpt["phone2idx"]
    idx2phone = ckpt["idx2phone"]
    cfg = ckpt.get("config", {})
    n_phones = cfg.get("n_phones", len(phone2idx))
    hidden = cfg.get("hidden", 256)

    model = build_model(n_phones, hidden=hidden)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper   = EmissionsWrapper(model)
    dummy_in  = torch.randn(1, 200, 40)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        dummy_in,
        str(onnx_path),
        input_names=["mel"],
        output_names=["log_probs"],
        dynamic_axes={"mel": {1: "time"}, "log_probs": {0: "time"}},
        opset_version=17,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  ✓ model.onnx  ({size_mb:.1f} MB, {n_phones} phones)")
    return phone2idx, idx2phone


# ── Lexicon export ─────────────────────────────────────────────────────────────

def export_lexicon(lexicon_path: str, phone2idx: dict, idx2phone: dict, out_dir: Path):
    print(f"Loading lexicon: {lexicon_path}")
    print(f"Merging with CMUdict: {CMUDICT_PATH}")
    raw = merge_lexicons(lexicon_path, CMUDICT_PATH)
    # Flatten to first pronunciation per entry (same as existing behaviour)
    lexicon = {word: prons[0] for word, prons in raw.items()}
    # Also include uppercase keys so browser lookup is case-insensitive-friendly
    extra = {word.upper(): prons[0] for word, prons in raw.items()
             if word.upper() not in raw}
    lexicon.update(extra)
    print(f"  {len(lexicon):,} words (Chinese + English merged)")

    lex_path = out_dir / "lexicon.json"
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(
            {"lexicon": lexicon, "phone2idx": phone2idx, "idx2phone": idx2phone},
            f, ensure_ascii=False, separators=(",", ":"),
        )
    size_mb = lex_path.stat().st_size / 1e6
    print(f"  ✓ lexicon.json  ({size_mb:.1f} MB)")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    STATIC_DIR.mkdir(exist_ok=True)
    print("=" * 45)
    print(" TinyAligner ONNX Export")
    print("=" * 45)
    print()

    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found: {CHECKPOINT}")
        print("Train the model first:  python train.py --epochs 50")
        sys.exit(1)

    phone2idx, idx2phone = export_model(CHECKPOINT, STATIC_DIR)
    print()
    export_lexicon(LEXICON_PATH, phone2idx, idx2phone, STATIC_DIR)


if __name__ == "__main__":
    main()
