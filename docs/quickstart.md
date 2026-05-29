# Quick Start: Setup & First Run

Get TinyAligner running in <30 minutes.

---

## 1. Environment

```bash
conda create -n tiny-aligner python=3.12
conda activate tiny-aligner
pip install -r requirements.txt
```

Verify:
```bash
python -c "from tiny_aligner import build_model; print('✓ OK')"
```

---

## 2. Data Setup

### Option A: Bilingual (Chinese + English) — Recommended

**AISHELL-1 (Chinese, ~15 GB):**
```bash
wget http://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz
mkdir -p data/aishell
mv data_aishell data/aishell/
```

**LibriSpeech (English, pick one or more):**
```bash
# Training (~6.3 GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C data/

# Validation + test (small, optional)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf dev-clean.tar.gz -C data/
tar -xzf test-clean.tar.gz -C data/
```

**Verify directory structure:**
```bash
ls data/aishell/data_aishell/wav/train/ | head -3
ls data/LibriSpeech/train-clean-100/ | head -3
```

### Option B: Chinese-Only

Skip LibriSpeech; train on AISHELL alone.

```bash
python scripts/train.py --epochs 50 --librispeech-root ""
```

---

## 3. Verify Data Loading

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from tiny_aligner.dataset import AishellDataset, LibriSpeechDataset
from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab

lex = merge_lexicons(
    'data/aishell/resource_aishell/lexicon.txt',
    'data/lexicons/cmudict-0.7b'
)
p2i, _ = build_phone_vocab(lex)

zh = AishellDataset('data/aishell/data_aishell', lex, p2i, split='train', max_samples=1)
en = LibriSpeechDataset('data', lex, p2i, split='dev-clean', max_samples=1)

print(f'✓ AISHELL train:     {len(zh)} samples')
print(f'✓ LibriSpeech test:  {len(en)} samples')
print(f'✓ Vocab:             {len(p2i)} phones')
"
```

Expected output:
```
✓ AISHELL train:     100 samples       (or ~120K for full)
✓ LibriSpeech test:  100 samples       (or ~2.7K for full dev-clean)
✓ Vocab:             258 phones
```

---

## 4. First Training Run (5 min smoke test)

```bash
python scripts/train.py --epochs 2 --max-train 1000 --patience 2
```

This trains on a small subset to verify everything works. Expected: loss decreases, sample decodes appear.

---

## 5. Serious Training

```bash
# Full bilingual (80 epochs, ~6 hours on V100)
python scripts/train.py --epochs 80

# Chinese-only (50 epochs, ~2 hours on V100)
python scripts/train.py --epochs 50 --librispeech-root ""
```

Monitor: each epoch prints dev PER, sample phoneme decodes. Best model saved to `checkpoints/best_model.pt`.

---

## 6. Export for Browser

```bash
python scripts/export_onnx.py
```

Outputs:
- `browser/public/model.onnx` (~3.7 MB)
- `browser/public/lexicon.json` (~1 MB)

---

## 7. Test in Browser

```bash
cd browser
npm install
npm run dev
# → Open http://localhost:5173
```

1. Upload an audio file (WAV, MP3)
2. Enter Chinese text (or English)
3. See word/phoneme alignments in real-time

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install -r requirements.txt` |
| `FileNotFoundError: .../data_aishell/...` | Check `ls data/aishell/data_aishell/wav/train/` |
| `LibriSpeech: 0 samples loaded` | Check `ls data/LibriSpeech/dev-clean/`. Dataset auto-detects nested layout (e.g., `dev-clean/dev-clean/...`). |
| `CUDA out of memory` | `python scripts/train.py --batch-size 8 --hidden 128` |
| No improvement during training | Try `--lr 2e-3` (higher) or `--lr 1e-3` (lower) |

---

## Next Steps

- **Train**: See [training.md](training.md) for parameters & recipes
- **API**: See [api.md](api.md) for Python usage
- **Deploy**: See [browser.md](browser.md) for production deployment
- **Internals**: See [architecture.md](architecture.md) for model design
