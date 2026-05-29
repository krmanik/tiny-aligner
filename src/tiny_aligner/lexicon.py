"""
Lexicon utilities for AISHELL forced alignment.

The lexicon maps Chinese words (characters/words) to phoneme sequences.
Format: word  ph1 ph2 ph3 ...
Example: 阿根廷 aa a1 g en1 t ing2

Phoneme-based approach is used because:
- ~200 unique phonemes vs 4000+ Chinese characters → smaller model softmax
- Multiple characters can share same pinyin; lexicon already disambiguates
- Consistent acoustic units regardless of character context
- Tones encoded in phoneme labels (a1, a2, a3, a4, a5)
"""

from collections import defaultdict
from pathlib import Path
import re


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SIL_TOKEN = "sil"


def load_lexicon(lexicon_path: str) -> dict[str, list[str]]:
    """
    Load lexicon file.
    Returns: dict mapping word -> list of phoneme sequences (multiple pronunciations possible)
    We keep all pronunciations and use the first one for alignment.
    """
    lexicon: dict[str, list[list[str]]] = defaultdict(list)
    with open(lexicon_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            phones = parts[1:]
            lexicon[word].append(phones)
    return lexicon


CMUDICT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "lexicons" / "cmudict-0.7b"


_ARPABET_STRESS_RE = re.compile(r"^([A-Z]+)[012]$")


def _strip_stress(ph: str) -> str:
    """AO1 → AO, AH0 → AH; leaves Chinese phones (lowercase) untouched."""
    m = _ARPABET_STRESS_RE.match(ph)
    return m.group(1) if m else ph


def load_cmudict(
    cmudict_path: str | Path = CMUDICT_PATH,
    strip_stress: bool = True,
) -> dict[str, list[list[str]]]:
    """
    Load CMU Pronouncing Dictionary.

    With strip_stress=True (default) ARPAbet phones lose the 0/1/2 stress digit
    (AO0/AO1/AO2 → AO). This cuts the English phone inventory from ~75 → ~39
    so each phone gets 3× more training examples — important for a tiny
    bilingual model. Chinese tones (a1/a2/…) are left alone because they are
    lowercase.

    Lines starting with ';' are comments.
    Alternate pronunciations are suffixed with (2), (3), etc. — we keep all.
    """
    lexicon: dict[str, list[list[str]]] = defaultdict(list)
    with open(cmudict_path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            # Strip alternate-pronunciation suffix: WORD(2) → WORD
            word = re.sub(r"\(\d+\)$", "", word)
            phones = parts[1:]  # e.g. ['S', 'AO1', 'S', 'AH0', 'JH']
            if strip_stress:
                phones = [_strip_stress(p) for p in phones]
            lexicon[word].append(phones)
    return lexicon


def merge_lexicons(
    chinese_path: str | Path,
    cmudict_path: str | Path = CMUDICT_PATH,
) -> dict[str, list[list[str]]]:
    """
    Merge Chinese (AISHELL) and English (CMUdict) lexicons.
    Chinese entries take priority on key collision (rare).
    """
    merged = load_cmudict(cmudict_path)
    chinese = load_lexicon(chinese_path)
    merged.update(chinese)          # Chinese overwrites any CMUdict collision
    return merged


def build_phone_vocab(lexicon: dict) -> tuple[dict, dict]:
    """
    Build phoneme vocabulary from lexicon.
    Returns: (phone2idx, idx2phone)
    Index 0 is always BLANK (required for CTC).
    """
    phones = set()
    for pronunciations in lexicon.values():
        for phone_seq in pronunciations:
            for ph in phone_seq:
                phones.add(ph)

    # Sort for reproducibility
    sorted_phones = sorted(phones)

    phone2idx = {BLANK_TOKEN: 0}
    idx2phone = {0: BLANK_TOKEN}

    for i, ph in enumerate(sorted_phones, start=1):
        phone2idx[ph] = i
        idx2phone[i] = ph

    # Add UNK at end
    unk_idx = len(phone2idx)
    phone2idx[UNK_TOKEN] = unk_idx
    idx2phone[unk_idx] = UNK_TOKEN

    return phone2idx, idx2phone


def text_to_phones(
    text: str,
    lexicon: dict,
    phone2idx: dict,
    add_sil: bool = True
) -> tuple[list[str], list[int]]:
    """
    Convert Chinese text (possibly space-separated words) to phoneme sequence.
    
    AISHELL transcripts already have space-separated words, e.g.:
        "而 对 楼市 成交 抑制 作用 最 大 的 限 购"
    Each token is looked up in the lexicon.
    
    Chinese text has no natural word boundaries, but AISHELL provides them.
    We handle both spaced and un-spaced input:
    - If spaces present: split on spaces → look up each word
    - If no spaces: split into individual characters → look up each char
    
    Returns:
        phones: list of phoneme strings
        phone_ids: list of phoneme indices
    """
    text = text.strip()

    # Determine tokens
    if " " in text:
        tokens = text.split()
    else:
        # Split into individual characters (each Chinese char is one token)
        tokens = list(text)

    phones = []
    if add_sil and SIL_TOKEN in phone2idx:
        phones.append(SIL_TOKEN)

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Case-insensitive lookup: try as-is → UPPER → lower
        # Handles both Chinese chars and LibriSpeech ALL-CAPS English words
        entry = (
            lexicon.get(token) or
            lexicon.get(token.upper()) or
            lexicon.get(token.lower())
        )
        if entry:
            phones.extend(entry[0])
        else:
            # Try splitting multi-char token into individual chars
            found = False
            char_phones = []
            for ch in token:
                if ch in lexicon:
                    char_phones.extend(lexicon[ch][0])
                    found = True
                else:
                    char_phones.append(UNK_TOKEN)
            if found:
                phones.extend(char_phones)
            else:
                phones.append(UNK_TOKEN)

    if add_sil and SIL_TOKEN in phone2idx:
        phones.append(SIL_TOKEN)

    phone_ids = [phone2idx.get(ph, phone2idx[UNK_TOKEN]) for ph in phones]
    return phones, phone_ids


def text_to_chars(text: str) -> list[str]:
    """
    Extract individual Chinese characters from text for TextGrid labeling.
    Handles space-separated word transcripts by stripping spaces.
    """
    # Remove spaces to get character list
    text = text.strip()
    chars = []
    for ch in text:
        if ch.strip():  # skip spaces
            chars.append(ch)
    return chars


def phones_to_char_boundaries(
    phones: list[str],
    lexicon: dict,
    text: str
) -> list[tuple[str, int, int]]:
    """
    Map phoneme-level alignment back to character-level.
    Returns list of (char, start_phone_idx, end_phone_idx) tuples.
    
    This is used after forced alignment to produce character timestamps.
    """
    text = text.strip()
    if " " in text:
        tokens = text.split()
    else:
        tokens = list(text)

    # Build per-token phone counts (how many phones each token has)
    char_boundaries = []
    phone_offset = 1  # start after leading SIL

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token in lexicon:
            n_phones = len(lexicon[token][0])
            for ch in token:
                # Distribute phones evenly across characters in multi-char tokens
                pass
            # For multi-char tokens, assign phones proportionally
            if len(token) == 1:
                char_boundaries.append((token, phone_offset, phone_offset + n_phones))
                phone_offset += n_phones
            else:
                # Split phones roughly evenly among characters
                n_chars = len(token)
                # Get phone sequence for this word
                word_phones = lexicon[token][0]
                # Try to assign 1-2 phones per char (Chinese syllable = initial+final)
                phones_per_char = len(word_phones) // n_chars
                remainder = len(word_phones) % n_chars
                offset = phone_offset
                for i, ch in enumerate(token):
                    n = phones_per_char + (1 if i < remainder else 0)
                    char_boundaries.append((ch, offset, offset + n))
                    offset += n
                phone_offset = offset
        else:
            # Try individual chars
            for ch in token:
                ch = ch.strip()
                if not ch:
                    continue
                if ch in lexicon:
                    n_phones = len(lexicon[ch][0])
                else:
                    n_phones = 1
                char_boundaries.append((ch, phone_offset, phone_offset + n_phones))
                phone_offset += n_phones

    return char_boundaries


if __name__ == "__main__":
    import sys

    LEXICON_PATH = "/home/mani/Documents/forced-aligner/data/aishell/resource_aishell/lexicon.txt"
    lexicon = load_lexicon(LEXICON_PATH)
    phone2idx, idx2phone = build_phone_vocab(lexicon)

    print(f"Lexicon size: {len(lexicon)} words")
    print(f"Phoneme vocab size: {len(phone2idx)} (including blank & unk)")
    print(f"Sample phonemes: {list(phone2idx.keys())[:20]}")

    sample = "而 对 楼市 成交 抑制 作用 最 大 的 限 购"
    phones, ids = text_to_phones(sample, lexicon, phone2idx)
    print(f"\nSample text: {sample}")
    print(f"Phones ({len(phones)}): {phones}")
    print(f"IDs: {ids}")
