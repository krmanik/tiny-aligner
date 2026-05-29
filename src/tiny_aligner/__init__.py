"""TinyAligner: Bilingual CTC acoustic model for forced alignment.

Modules:
  model - TinyAligner neural network
  dataset - AISHELL + LibriSpeech loaders
  lexicon - Phoneme utilities (Chinese + English)
  align - Alignment inference
"""

__version__ = "0.2.0"

from .model import TinyAligner, build_model
from .lexicon import load_lexicon, merge_lexicons, build_phone_vocab, text_to_phones
from .dataset import AishellDataset, LibriSpeechDataset, get_dataloaders

__all__ = [
    "TinyAligner",
    "build_model",
    "load_lexicon",
    "merge_lexicons",
    "build_phone_vocab",
    "text_to_phones",
    "AishellDataset",
    "LibriSpeechDataset",
    "get_dataloaders",
]
