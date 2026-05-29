# Training Guide

All training modes and parameters.

---

## Default (Bilingual)

```bash
python scripts/train.py --epochs 80 --lr 2e-3 --batch-size 32
```

- Trains on all AISHELL (~120K) + all LibriSpeech train-clean-100 (~28.5K)
- Balanced 50/50 sampling each batch (ZH:EN)
- SpecAugment on mels, validation on AISHELL-dev + LibriSpeech dev-clean
- Best model → `checkpoints/best_model.pt`
- ~6 hours on V100/A100

---

## Chinese-Only

```bash
python scripts/train.py --epochs 50 --librispeech-root ""
```

- AISHELL only, no English data
- ~2 hours on V100
- Higher LR helps (default `3e-3` is tuned for bilingual; bilingual uses `2e-3`)

---

## Parameters Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 80 | Max training epochs |
| `--batch-size` | 32 | Reduce if OOM |
| `--lr` | 2e-3 | Learning rate (bilingual); use 3e-3 for Chinese-only |
| `--hidden` | 256 | Bi-GRU hidden size (total output = 2×hidden) |
| `--patience` | 15 | Early stopping after N epochs without improvement |
| `--workers` | 4 | DataLoader threads |
| `--max-train` | None | Cap training samples (e.g. 1000 for smoke test) |
| `--resume` | None | Checkpoint path to resume from |
| `--save-dir` | `checkpoints` | Output directory for models |
| `--no-balanced` | False | Disable balanced ZH/EN sampling |
| `--no-augment` | False | Disable SpecAugment on training data |
| `--librispeech-root` | `/path/to/data` | Root dir containing LibriSpeech/; "" to skip |
| `--librispeech-splits` | `train-clean-100` | Space-separated: `train-clean-100 train-clean-360` |
| `--sample-every` | 5 | Print decoded samples every N epochs (0=off) |

---

## Common Recipes

### Quick validation (2 min)
```bash
python scripts/train.py --epochs 2 --max-train 500 --patience 2
```

### Fine-tune from checkpoint
```bash
python scripts/train.py \
  --resume checkpoints/best_model.pt \
  --epochs 20 \
  --lr 5e-4 \
  --max-train 2000  # smaller, focused tuning
```

### Chinese-only from scratch
```bash
python scripts/train.py --epochs 50 --librispeech-root "" --lr 3e-3
```

### English-heavy (more LibriSpeech)
```bash
python scripts/train.py \
  --epochs 80 \
  --librispeech-splits train-clean-100 train-clean-360 \
  --batch-size 24  # larger = need bigger splits
```

### Low-memory training
```bash
python scripts/train.py \
  --batch-size 8 \
  --hidden 128 \
  --workers 2
```

---

## Monitoring

**Per-epoch output:**
```
Epoch 003/80 | train=0.6342 | dev=0.5891 | PER=0.1823 | Acc=81.8% | lr=2.84e-03 | time=169s  [NEW BEST]
```

- `train` = training loss (CTC)
- `dev` = dev loss
- `PER` = phoneme error rate (lower=better)
- `Acc` = 1 - PER (higher=better, target ≥85%)
- `lr` = current learning rate (OneCycleLR schedule)

**Sample decode check** (every 5 epochs by default):
```
[1] text : 这 是 一 个 例 子
    REF  : sil zh e4 sh i4 ... (reference phones)
    HYP  : sil zh e4 sh i2 ... (predicted phones)
    PER  : 0.167  ->  83.3% correct
```

Compares greedy CTC decode vs reference. If HYP has lots of `<unk>`, model isn't converging well.

---

## Tuning Tips

| Symptom | Solution |
|---------|----------|
| Loss stuck high, no improvement | Try `--lr 3e-3` (higher) or `--lr 1e-3` (lower) |
| English PER much worse than Chinese | Check LibriSpeech data loaded; try `--no-balanced` to debug |
| CUDA OOM | `--batch-size 8 --hidden 128` |
| Training too slow | `--workers 8` (if CPU not saturated) |
| Model overfits (train low, dev high) | Increase `--batch-size`, reduce `--lr` slightly, increase patience |

---

## Export

Once training completes:

```bash
python scripts/export_onnx.py
```

Creates:
- `browser/public/model.onnx` — ONNX inference model
- `browser/public/lexicon.json` — merged lexicon + phone mappings
- Checkpoints saved to `checkpoints_smoke/` if `--save-dir` was used

---

## Next Steps

- **Deploy**: [browser.md](browser.md)
- **API**: [api.md](api.md)
- **Architecture**: [architecture.md](architecture.md#training-details)
