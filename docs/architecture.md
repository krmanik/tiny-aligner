# Architecture & Technical Details

Deep dive into model design, training, and implementation.

---

## Model Architecture

### Input

- **Audio**: 16 kHz mono WAV
- **Features**: 40-dim log Mel-filterbank
  - Window: 25 ms (400 samples)
  - Hop: 10 ms (160 samples) → 50 Hz frame rate
  - Freq range: 80–7600 Hz
  - Per-utterance CMVN (mean-variance norm)

### Network

```
Input [B, T, 40]
  ↓
Conv1d × 3 (64 → 128 → 192 channels)
  stride=1, 1, 2
  → [B, 192, T/2]  (2× downsample)
  ↓
Bi-GRU (hidden=256)
  input=192, output=512 (2×hidden)
  → [B, T/2, 512]
  ↓
Linear (512 → 258)
  → [B, T/2, 258]  logits
  ↓
Softmax + Log
  → [T/2, 258]  log probabilities (CTC format)
```

**Parameters**: 930K (~3.7 MB float32, ~900 KB INT8)

### Output

- **Frame rate**: 20 ms (T/2 where T is 10 ms hop frames)
- **Vocab**: 258 tokens
  - Index 0: `<blank>` (CTC blank)
  - Indices 1–217: Chinese phones (pinyin + tones, e.g., `a1`, `ang4`)
  - Indices 218–256: English phones (ARPAbet, stress-stripped, e.g., `AA`, `AH`)
  - Index 257: `<unk>`

---

## Training

### Data

**Bilingual (default):**
- AISHELL-1 (Mandarin): ~120K utterances, ~340 hours
- LibriSpeech train-clean-100 (English): ~28.5K utterances, ~100 hours
- Total: ~440 hours

**Sampling**: Balanced via `WeightedRandomSampler` (50/50 ZH/EN per batch)

### Preprocessing

1. **Mel features**: Per-utterance normalization (zero mean, unit variance)
2. **SpecAugment** (training only):
   - Freq mask: 2 masks, ≤8 bins each
   - Time mask: 2 masks, ≤25 frames each
   - Masking value: per-utterance mean

### Loss

CTC loss with `zero_infinity=True` (ignores blank alignments outside valid phone range).

### Optimization

- **Optimizer**: AdamW (lr=2e-3, weight_decay=1e-4)
- **Scheduler**: OneCycleLR
  - pct_start=0.1 (10% warmup)
  - anneal_strategy=cosine
  - total_steps = epochs × batches/epoch
- **Gradient clip**: 5.0
- **Mixed precision**: AMP (autocast("cuda"))

### Hyperparams

| Param | Value | Notes |
|-------|-------|-------|
| Epochs | 80 | Early stop if PER doesn't improve ×15 epochs |
| Batch size | 32 | Bilingual; reduce to 8 for low-memory |
| LR | 2e-3 | 3e-3 for Chinese-only |
| Hidden | 256 | ~930K params; was 128 (480K) before May 2026 |
| Patience | 15 | Early stopping patience |

---

## Lexicon

### Chinese (AISHELL)

From `resource_aishell/lexicon.txt`:
```
这 zh e4
是 sh i4 / s h i 4
```

- Character-level entries
- Pronunciation: initial + final + tone (0–5)
- Multiple pronunciations separated by `/` (first used)
- ~8K unique characters

### English (CMU Pronouncing Dictionary)

Original CMUdict has stress markers (AO0, AO1, AO2). **Now stripped** (May 2026):
```
DIFFERENT   D IH1 F ER0 AH0 N T   →  D IH F ER AH N T
NATASHA     N AH0 T AA1 SH AH0   →  N AH T AA SH AH
```

Rationale: Stress (0/1/2) is acoustic detail; removing it:
- Reduces EN vocab 75 → 39 phones
- 3× training examples per phone
- Doesn't hurt CTC (CTC ignores state labels anyway)
- Tones intact for Chinese (lowercase: `a1`, `a2`, etc.)

### Merging

`merge_lexicons()` combines:
1. CMUdict (English)
2. AISHELL lexicon (Chinese)
3. Adds uppercase keys for case-insensitive lookup

Result: ~260K entries, 258 unique phones.

---

## Forced Alignment (Viterbi)

### CTC Path Construction

Given phones = [`sil`, `d`, `ih`, `f`, ...]:

```
Forced path = [blank, sil, blank, d, blank, ih, blank, f, ..., blank]
```

Interleaves blanks between phones (CTC requirements).

### Viterbi Decoding

State space: 2N+1 (blank and phone states)

Transitions:
- Stay in state s
- Move to s-1 (go back)
- Move to s-2 (skip blank if token differs)

Forward algorithm (DP):
```
α[t, s] = max(α[t-1, s], α[t-1, s-1], α[t-1, s-2]) + log_probs[t, phone[s]]
```

Backtrace to get state sequence → phone indices per frame.

### Segmentation

Merge consecutive frames with same phone ID → segments (start, end, phone).

Map back to words/characters using lexicon token boundaries.

---

## Training Details

### May 2026 Fixes

**Before:**
- LibriSpeech data silently loaded 0 samples due to nested folder bug (`LibriSpeech/split/split/...`)
- Model trained on Chinese-only, but vocab included ARPAbet (false bilingual)
- English output: `<unk>` spam, useless alignments

**After:**
1. **Dataset loader** detects nested layout, adjusts paths
2. **Stress stripping** in CMUdict loader (AR->AO1→AO)
3. **Bigger model**: 480K → 930K params
4. **Balanced sampling**: WeightedRandomSampler for 50/50 ZH/EN
5. **SpecAugment**: Freq+time masks on training mels

**Result** (5-epoch smoke test):
- PER: 26.3% (combined ZH+EN, all from scratch)
- English now emits real phones (not `<unk>`)
- Model converges faster

### Convergence

Typical:
- Epoch 1–10: Fast improvement (50% → 20% PER)
- Epoch 10–40: Steady (20% → 15%)
- Epoch 40–80: Slow plateau (~15% with balanced training)

Chinese dominates early; English catches up by epoch 20–30.

---

## Implementation Notes

### ONNX Export

`EmissionsWrapper` wraps `model.get_emissions()` for clean ONNX:
- No `pack_padded_sequence` (ONNX-unfriendly)
- Input: [1, T, 40]
- Output: [T', n_phones]
- Opset 17 (compat with ONNX Runtime 1.14+)

### Browser Inference

- ONNX Runtime (JavaScript/WebAssembly)
- Single-threaded (no multi-threading in browsers)
- ~1–2s for 10s audio on modern CPU

### Python API

See [api.md](api.md).

---

## Troubleshooting

### "LibriSpeech: 0 samples"

**Cause**: Nested layout detection failed or files not extracted correctly.

**Fix**:
```bash
ls data/LibriSpeech/dev-clean/
# Should show: BOOKS.TXT, CHAPTERS.TXT, dev-clean/
# If it's dev-clean/dev-clean/, that's the nested case (now auto-detected)
```

### Loss spikes / unstable training

**Cause**: Learning rate too high or batch size too small.

**Fix**:
```bash
python scripts/train.py --lr 1e-3 --batch-size 64
```

### English much worse than Chinese

**Cause**: Check that LibriSpeech samples are actually loading.

**Fix**:
```bash
python scripts/train.py --epochs 1 --sample-every 1 --max-train 1000
# Look for English text in "SAMPLE DECODE" output
```

### OOM on GPU

**Fix**:
```bash
CUDA_VISIBLE_DEVICES="" python scripts/train.py --batch-size 8 --hidden 128  # Force CPU
```

---

## Future Improvements

- [ ] Streaming inference (non-offline RNN stateful processing)
- [ ] Multi-lingual (add Korean, Japanese, etc.)
- [ ] Faster inference via TFLite / CoreML export
- [ ] Attention-based model (currently CTC-only)
- [ ] Character-level decoding (skip phonemes for some languages)

---

## References

- **CTC Loss**: Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" (2006)
- **CTC Forced Alignment**: Kürzinger et al., "CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition" (2020)
- **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (2019)

---

## Next Steps

- **Training**: [training.md](training.md)
- **API**: [api.md](api.md)
- **Browser**: [browser.md](browser.md)
