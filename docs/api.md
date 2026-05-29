# Python API Reference

Use TinyAligner as a library in your Python projects.

---

## Installation

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

Or add to your project:
```python
import sys
sys.path.insert(0, '/path/to/forced-alignment/src')
```

---

## Core Modules

### `tiny_aligner.model`

**Build a model:**
```python
from tiny_aligner.model import build_model

model = build_model(n_phones=258, hidden=256)
# → TinyAligner (930K params)
print(f"Parameters: {model.count_parameters():,}")
```

**Forward pass:**
```python
import torch

# Input: [batch, time, 40]
mel = torch.randn(4, 300, 40)
lengths = torch.tensor([300, 280, 250, 200])

# Output: log probs [time, batch, n_phones] (CTC format)
log_probs, out_lengths = model(mel, lengths)

# Or inference mode (no length info):
log_probs = model.get_emissions(mel)  # → [time, n_phones]
```

**Load checkpoint:**
```python
import torch

ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
phone2idx = ckpt['phone2idx']
idx2phone = {int(k): v for k, v in ckpt['idx2phone'].items()}
config = ckpt['config']

model = build_model(
    n_phones=config['n_phones'],
    hidden=config['hidden']
)
model.load_state_dict(ckpt['model'])
model.eval()
```

---

### `tiny_aligner.lexicon`

**Load lexicons:**
```python
from tiny_aligner.lexicon import (
    load_lexicon,
    load_cmudict,
    merge_lexicons,
    build_phone_vocab,
)

# Chinese only
lex_zh = load_lexicon('data/aishell/resource_aishell/lexicon.txt')

# English only
lex_en = load_cmudict('data/lexicons/cmudict-0.7b')

# Bilingual
lex = merge_lexicons(
    'data/aishell/resource_aishell/lexicon.txt',
    'data/lexicons/cmudict-0.7b'
)

phone2idx, idx2phone = build_phone_vocab(lex)
print(f"Vocab size: {len(phone2idx)}")  # 258
```

**Text to phones:**
```python
from tiny_aligner.lexicon import text_to_phones

# Chinese
phones, ids = text_to_phones('这是一个例子', lex, phone2idx)
# phones: ['sil', 'zh', 'e4', 'sh', 'i4', ...]
# ids: [186, 174, 195, ...]

# English
phones, ids = text_to_phones('HELLO WORLD', lex, phone2idx)
# phones: ['sil', 'H', 'EH', 'L', 'OW', ...]
```

**Lookup word:**
```python
# Get all pronunciations
entry = lex.get('这')       # [['zh', 'e4'], ...]
entry = lex.get('DIFFERENT')  # [['D', 'IH', 'F', 'ER', 'AH', 'N', 'T'], ...]

# First pronunciation (what text_to_phones uses):
phones = entry[0] if entry else ['<unk>']
```

---

### `tiny_aligner.dataset`

**Load AISHELL:**
```python
from tiny_aligner.dataset import AishellDataset

ds = AishellDataset(
    'data/aishell/data_aishell',
    lexicon=lex,
    phone2idx=phone2idx,
    split='train',  # or 'dev', 'test'
    max_samples=100,
    augment=True  # SpecAugment on training
)

sample = ds[0]
# sample['mel']:        [T, 40]
# sample['phone_ids']:  [P]
# sample['text']:       str
# sample['phones']:     [str]
```

**Load LibriSpeech:**
```python
from tiny_aligner.dataset import LibriSpeechDataset

ds = LibriSpeechDataset(
    'data',  # parent of LibriSpeech/
    lexicon=lex,
    phone2idx=phone2idx,
    split='dev-clean',  # or 'train-clean-100', 'test-clean'
    max_samples=100,
    augment=True
)
```

**Create dataloaders:**
```python
from tiny_aligner.dataset import get_dataloaders

train_dl, dev_dl = get_dataloaders(
    'data/aishell/data_aishell',
    lexicon=lex,
    phone2idx=phone2idx,
    batch_size=32,
    num_workers=4,
    librispeech_root='data',  # None for Chinese-only
    balanced_sampling=True,  # 50/50 ZH/EN
    augment_train=True  # SpecAugment
)

for batch in train_dl:
    mel = batch['mel']              # [32, T, 40]
    phone_ids = batch['phone_ids']  # [32, P]
    mel_lengths = batch['mel_lengths']
    phone_lengths = batch['phone_lengths']
    break
```

---

### `tiny_aligner.align`

**Run forced alignment:**
```python
from tiny_aligner.align import align

result = align(
    model=model,
    wav_path='path/to/audio.wav',
    text='这是一个例子',
    lexicon=lex,
    phone2idx=phone2idx,
    idx2phone=idx2phone,
    output_textgrid='output.TextGrid',  # optional
    device=torch.device('cpu')
)

print(result.keys())
# ['phones', 'phone_intervals', 'word_intervals', 'char_intervals', 'duration']

for start, end, label in result['word_intervals']:
    print(f"{label}: {start:.3f}–{end:.3f}s")
```

**Output structure:**
```python
result = {
    'phones': ['sil', 'zh', 'e4', 's', ...],
    'phone_intervals': [(0.00, 0.05, 'sil'), (0.05, 0.15, 'zh'), ...],
    'word_intervals': [(0.00, 0.30, '这'), (0.30, 0.50, '是'), ...],
    'char_intervals': [(0.00, 0.30, '这'), (0.30, 0.50, '是'), ...],
    'duration': 10.25,  # audio duration in seconds
}
```

**TextGrid format:**
```python
# result['output_textgrid'] auto-writes to file with 3 tiers:
# 1. phones   — phoneme intervals
# 2. words    — word intervals
# 3. chars    — character intervals
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from tiny_aligner import build_model
from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab
from tiny_aligner.dataset import get_dataloaders

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lexicon & vocab
lex = merge_lexicons(
    'data/aishell/resource_aishell/lexicon.txt',
    'data/lexicons/cmudict-0.7b'
)
phone2idx, idx2phone = build_phone_vocab(lex)
n_phones = len(phone2idx)

# Model
model = build_model(n_phones=n_phones, hidden=256).to(device)

# Data
train_dl, dev_dl = get_dataloaders(
    'data/aishell/data_aishell',
    lex, phone2idx,
    batch_size=32,
    librispeech_root='data'
)

# Optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
total_steps = 80 * len(train_dl)
scheduler = OneCycleLR(
    optimizer, max_lr=2e-3, total_steps=total_steps,
    pct_start=0.1, anneal_strategy='cos'
)

# Training loop
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
scaler = torch.amp.GradScaler('cuda')

for epoch in range(80):
    model.train()
    for batch in train_dl:
        mel = batch['mel'].to(device)
        phone_ids = batch['phone_ids'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        phone_lengths = batch['phone_lengths'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            log_probs, out_lengths = model(mel, mel_lengths)
            targets_flat = torch.cat([
                phone_ids[i, :phone_lengths[i]]
                for i in range(phone_ids.shape[0])
            ])
            loss = ctc_loss_fn(log_probs, targets_flat, out_lengths, phone_lengths)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in dev_dl:
            mel = batch['mel'].to(device)
            phone_ids = batch['phone_ids'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            phone_lengths = batch['phone_lengths'].to(device)

            log_probs, out_lengths = model(mel, mel_lengths)
            targets_flat = torch.cat([
                phone_ids[i, :phone_lengths[i]]
                for i in range(phone_ids.shape[0])
            ])
            loss = ctc_loss_fn(log_probs, targets_flat, out_lengths, phone_lengths)
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/80 | val_loss={val_loss/len(dev_dl):.4f}')
    torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
```

---

## Common Patterns

### Batch processing multiple files

```python
import glob
from pathlib import Path

audio_files = glob.glob('data/test/*.wav')
texts = {Path(f).stem: Path(f.replace('.wav', '.txt')).read_text()
         for f in audio_files}

for wav_path in audio_files:
    text = texts[Path(wav_path).stem]
    result = align(model, wav_path, text, lex, phone2idx, idx2phone)
    print(f"{Path(wav_path).name}: {len(result['word_intervals'])} words")
```

### Interactive shell

```python
import torch
from tiny_aligner import build_model
from tiny_aligner.lexicon import merge_lexicons, build_phone_vocab
from tiny_aligner.align import align

# Load once
model = build_model(258, 256)
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model'])
lex = merge_lexicons('data/aishell/resource_aishell/lexicon.txt',
                     'data/lexicons/cmudict-0.7b')
p2i, i2p = build_phone_vocab(lex)

# Use interactively
result = align(model, 'test.wav', '这是测试', lex, p2i, i2p)
for s, e, w in result['word_intervals']:
    print(f'{w}: {s:.2f}–{e:.2f}s')
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'tiny_aligner'` | `export PYTHONPATH=$PYTHONPATH:$(pwd)/src` |
| `FileNotFoundError: .../lexicon.txt` | Check paths; verify data/ exists |
| `CUDA out of memory` | `device = torch.device('cpu')` or reduce batch size |
| `AssertionError: blank in phone2idx` | Lexicon missing `<blank>` token; call `build_phone_vocab()` |

---

## Next Steps

- **Training**: [training.md](training.md)
- **Architecture**: [architecture.md](architecture.md)
- **Browser**: [browser.md](browser.md)
