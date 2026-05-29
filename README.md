# TinyAligner — Bilingual Forced Alignment

Accurate, fast, tiny. A CTC-based acoustic model (930K parameters) for Chinese + English audio forced alignment, running in your browser.

**Key features:**
- 📱 Browser-deployable via ONNX (~3.7 MB float32)
- 🗣️ Bilingual: Chinese (AISHELL) + English (LibriSpeech)

---

## Quick Start (5 min)

### 1. Install

```bash
conda create -n tiny-aligner python=3.12
conda activate tiny-aligner
pip install -r requirements.txt
```

### 2. Download Data (optional, for training)

**AISHELL-1** (Chinese, ~15 GB):
```bash
wget http://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz -C data/aishell/
```

**LibriSpeech** (English, ~6-7 GB per split):
```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C data/
```

See [docs/quickstart.md](docs/quickstart.md) for detailed setup.

### 3. Train a Model

```bash
# Bilingual (Chinese + English)
python scripts/train.py --epochs 80

# Chinese-only
python scripts/train.py --epochs 50 --librispeech-root ""
```

Monitor via sample decodes printed each epoch. See [docs/training.md](docs/training.md) for tuning & parameters.

### 4. Export for Browser

```bash
python scripts/export_onnx.py
# → browser/public/model.onnx + lexicon.json
```

### 5. Run Browser App

```bash
cd browser
npm install
npm run dev
# → http://localhost:5173
```

Upload audio + text, get word/character/phoneme alignments as TextGrid or SRT.

---

## How It Works

**Architecture:**
```
Audio (16kHz WAV) → Mel-filterbank (40 dims) → Conv3 + Bi-GRU → Softmax over ~260 phones
                                                  [930K params]
```

**Alignment:**
- Model outputs per-frame log-probabilities (50 Hz, 20ms)
- CTC forced alignment: Viterbi decoding with a fixed phone sequence
- Maps frames → phones → words → characters

**Languages:**
- **Chinese**: 220 phonemes (pinyin + tones from AISHELL lexicon)
- **English**: 39 phonemes (ARPAbet, stress-normalized; CMUdict)
- Bilingual training (balanced 50/50 sampling) since May 2026

---

## Project Structure

```
forced-alignment/
├── README.md                      # This file
├── requirements.txt
│
├── src/tiny_aligner/              # Python library
│   ├── model.py                   # CTC acoustic model
│   ├── dataset.py                 # AISHELL + LibriSpeech loaders
│   ├── lexicon.py                 # Phoneme utilities
│   └── align.py                   # TextGrid generation
│
├── scripts/
│   ├── train.py                   # Training script
│   └── export_onnx.py             # ONNX export
│
├── browser/                       # SvelteKit web app
│   ├── src/lib/alignment/         # ONNX + CTC alignment (TypeScript)
│   └── public/                    # model.onnx + lexicon.json
│
├── data/
│   ├── aishell/                   # AISHELL-1 (Chinese)
│   ├── LibriSpeech/               # LibriSpeech (English)
│   └── lexicons/cmudict-0.7b      # CMU Pronouncing Dictionary
│
├── checkpoints/                   # Saved models
├── tests/                         # Unit tests
└── docs/                          # Documentation
    ├── quickstart.md              # Install & data setup
    ├── training.md                # Training guide
    ├── api.md                     # Python API
    ├── browser.md                 # Browser deployment
    └── architecture.md            # Model design details
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/quickstart.md](docs/quickstart.md) | Installation, data download, first run |
| [docs/training.md](docs/training.md) | Training modes, parameters, tuning |
| [docs/api.md](docs/api.md) | Python API reference |
| [docs/browser.md](docs/browser.md) | Browser deployment (Vercel, Netlify, static) |
| [docs/architecture.md](docs/architecture.md) | Model design, training recipes, troubleshooting |

---

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model size | 3.7 MB | float32 ONNX |
| Parameters | 930K | ~900K INT8 quantized |
| Inference speed | RTF ~0.02 | CPU; <10ms per sec of audio |
| Frame rate | 50 Hz | 20ms resolution |
| Languages | 2 | Chinese (220 phones) + English (39 phones) |


---

## Citation

If you use TinyAligner in research, please cite:

```bibtex
@software{tinyaligner2026,
  title = {TinyAligner: Bilingual Forced Alignment for Browser},
  author = {krmanik},
  year = {2026},
  url = {https://github.com/krmanik/tiny-aligner}
}
```

---

## License

**Code:** MIT License

**Data:**
- **AISHELL-1**: Apache License 2.0
- **LibriSpeech**: CC BY 4.0
- **CMUdict**: Public Domain

---


## Contributing

Contributions welcome! Areas:
- Non-Latin script support (Korean, Japanese, etc.)
- Streaming inference (non-offline)
- More training recipes & pretrained models
