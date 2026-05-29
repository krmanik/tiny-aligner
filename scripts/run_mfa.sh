#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_mfa.sh  —  Run MFA forced alignment on audio + transcript files
#
# Produces one TextGrid per audio file in OUTPUT_DIR.
#
# Usage:
#   bash scripts/run_mfa.sh                          # default: test/ files
#   bash scripts/run_mfa.sh --audio-dir path/to/wav \
#                            --text-dir  path/to/txt \
#                            --output-dir path/to/out
#
# Language detection:
#   Files with ≥30% CJK characters → Mandarin (pypinyin converts to pinyin)
#   Otherwise                       → English
#
# Requirements:
#   conda activate bfa
#   (MFA already installed: mfa --version → 3.x)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

AUDIO_DIR="${PROJECT_ROOT}/test/audio"
TEXT_DIR="${PROJECT_ROOT}/test/text"
OUTPUT_DIR="${PROJECT_ROOT}/test/mfa_textgrids"

MFA_MODELS_DIR="${HOME}/Documents/MFA/pretrained_models"
ACOUSTIC_EN="${MFA_MODELS_DIR}/acoustic/english_mfa.zip"
ACOUSTIC_ZH="${MFA_MODELS_DIR}/acoustic/mandarin_mfa.zip"
DICT_EN="${MFA_MODELS_DIR}/dictionary/english_mfa.dict"
DICT_ZH="${MFA_MODELS_DIR}/dictionary/mandarin_pinyin.dict"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-dir)  AUDIO_DIR="$2";  shift 2 ;;
    --text-dir)   TEXT_DIR="$2";   shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --acoustic-en) ACOUSTIC_EN="$2"; shift 2 ;;
    --acoustic-zh) ACOUSTIC_ZH="$2"; shift 2 ;;
    --dict-en)    DICT_EN="$2";    shift 2 ;;
    --dict-zh)    DICT_ZH="$2";    shift 2 ;;
    -h|--help)
      sed -n '3,20p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[MFA] $*"; }
die()  { echo "[MFA] ERROR: $*" >&2; exit 1; }
hr()   { printf '%0.s-' {1..70}; echo; }

is_chinese() {
  # Returns 0 (true) if ≥30% of non-space chars are CJK
  python3 - "$1" <<'PYEOF'
import sys, unicodedata
text = open(sys.argv[1], encoding="utf-8").read()
chars = [c for c in text if not c.isspace()]
cjk   = sum(1 for c in chars if '\u4e00' <= c <= '\u9fff')
sys.exit(0 if chars and cjk / len(chars) >= 0.3 else 1)
PYEOF
}

chars_to_pinyin() {
  # Convert a Chinese text file to space-separated tone-numbered pinyin.
  # Non-CJK characters (English words, punctuation, digits) are skipped so
  # MFA never sees OOV tokens against the Mandarin dictionary.
  python3 - "$1" <<'PYEOF'
import sys
from pypinyin import pinyin, Style
text = open(sys.argv[1], encoding="utf-8").read()
result = []
for ch in text:
    if '\u4e00' <= ch <= '\u9fff':
        py = pinyin(ch, style=Style.TONE3, heteronym=False)
        if py and py[0]:
            syl = py[0][0]
            if not syl[-1].isdigit():
                syl += "5"
            result.append(syl)
    # Non-CJK chars (English, digits, punctuation) are intentionally skipped
    # to avoid OOV tokens that cause MFA to mark everything as <unk>
print(" ".join(result))
PYEOF
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
hr
log "Pre-flight checks"

command -v mfa >/dev/null 2>&1   || die "mfa not in PATH. Run: conda activate bfa"
command -v python3 >/dev/null    || die "python3 not found"
python3 -c "import pypinyin"     2>/dev/null || die "pypinyin not installed: pip install pypinyin"

[[ -d "$AUDIO_DIR" ]] || die "Audio dir not found: $AUDIO_DIR"
[[ -d "$TEXT_DIR"  ]] || die "Text dir not found:  $TEXT_DIR"

log "Audio dir  : $AUDIO_DIR"
log "Text dir   : $TEXT_DIR"
log "Output dir : $OUTPUT_DIR"
log "MFA models : $MFA_MODELS_DIR"
hr

# ── Collect files and split by language ───────────────────────────────────────
ZH_FILES=()
EN_FILES=()
MISSING=()

for wav in "$AUDIO_DIR"/*.wav; do
  [[ -f "$wav" ]] || continue
  stem="$(basename "$wav" .wav)"
  txt="${TEXT_DIR}/${stem}.txt"

  if [[ ! -f "$txt" ]]; then
    log "  ⚠  No transcript for $stem — skipping"
    MISSING+=("$stem")
    continue
  fi

  if is_chinese "$txt"; then
    ZH_FILES+=("$stem")
    log "  [ZH] $stem"
  else
    EN_FILES+=("$stem")
    log "  [EN] $stem"
  fi
done

if [[ ${#ZH_FILES[@]} -eq 0 && ${#EN_FILES[@]} -eq 0 ]]; then
  die "No valid audio+transcript pairs found in $AUDIO_DIR / $TEXT_DIR"
fi

log "Found ${#ZH_FILES[@]} Chinese, ${#EN_FILES[@]} English file(s)"
hr

mkdir -p "$OUTPUT_DIR"

# ── Function: run MFA on one language group ───────────────────────────────────
run_mfa_group() {
  local label="$1"
  local acoustic="$2"
  local dictionary="$3"
  shift 3
  local files=("$@")

  [[ ${#files[@]} -eq 0 ]] && return

  log "Running MFA [$label] on ${#files[@]} file(s) (one at a time)..."

  [[ -f "$acoustic"   ]] || die "Acoustic model not found: $acoustic"
  [[ -f "$dictionary" ]] || die "Dictionary not found: $dictionary"

  # Write a config file to widen the beam (flags not available as CLI args in MFA 3.x)
  local mfa_config
  mfa_config="$(mktemp /tmp/mfa_cfg_XXXXXX.yaml)"
  cat > "$mfa_config" <<'YAMLEOF'
beam: 100
retry_beam: 400
YAMLEOF

  local total_ok=0

  for stem in "${files[@]}"; do
    local wav="${AUDIO_DIR}/${stem}.wav"
    local txt="${TEXT_DIR}/${stem}.txt"

    # Build a single-file temp corpus
    local corpus_dir out_dir
    corpus_dir="$(mktemp -d /tmp/mfa_corpus_XXXXXX)"
    out_dir="$(mktemp -d /tmp/mfa_output_XXXXXX)"

    # Resample audio to 16 kHz (MFA's mandarin_mfa model expects 16 kHz)
    python3 -c "
import torchaudio
wav, sr = torchaudio.load('$wav')
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000
torchaudio.save('${corpus_dir}/${stem}.wav', wav, sr)
" 2>/dev/null || ln -sf "$(realpath "$wav")" "${corpus_dir}/${stem}.wav"

    # Write .lab transcript
    if [[ "$label" == "Chinese" ]]; then
      local lab
      lab="$(chars_to_pinyin "$txt")"
      if [[ -z "$lab" ]]; then
        log "  ⚠  $stem: pinyin conversion produced empty output — skipping"
        rm -rf "$corpus_dir" "$out_dir"
        continue
      fi
      echo "$lab" > "${corpus_dir}/${stem}.lab"
    else
      python3 -c "
import re, sys
t = open('$txt', encoding='utf-8').read()
print(re.sub(r\"[^\\w\\s']\", '', t).upper())
" > "${corpus_dir}/${stem}.lab"
    fi

    log "  [$stem] lab: $(cat "${corpus_dir}/${stem}.lab")"

    # Capture MFA output to a temp log; don't pipe through grep so MFA never
    # gets SIGPIPE from an early-exiting grep (tqdm progress bars can trigger this)
    local mfa_log
    mfa_log="$(mktemp /tmp/mfa_log_XXXXXX)"
    local mfa_ok=0
    mfa align \
      --clean \
      --overwrite \
      -c "$mfa_config" \
      "$corpus_dir" \
      "$dictionary" \
      "$acoustic" \
      "$out_dir" >"$mfa_log" 2>&1 || mfa_ok=$?

    # If first pass failed, retry with --fine_tune (extra tuning pass)
    if [[ $mfa_ok -ne 0 || ! -f "${out_dir}/${stem}.TextGrid" ]]; then
      log "  [${stem}] First pass failed, retrying with --fine_tune..."
      mfa align \
        --clean \
        --overwrite \
        --fine_tune \
        -c "$mfa_config" \
        "$corpus_dir" \
        "$dictionary" \
        "$acoustic" \
        "$out_dir" >"$mfa_log" 2>&1 || mfa_ok=$?
    fi

    # Show non-empty log lines
    grep -v '^[[:space:]]*$' "$mfa_log" || true
    rm -f "$mfa_log"

    local tg="${out_dir}/${stem}.TextGrid"
    if [[ -f "$tg" ]]; then
      cp "$tg" "${OUTPUT_DIR}/${stem}.TextGrid"
      log "  ✓ ${stem}.TextGrid saved"
      (( total_ok++ )) || true
    else
      log "  ✗ No TextGrid produced for $stem"
    fi

    rm -rf "$corpus_dir" "$out_dir"
  done

  log "  Finished [$label]: $total_ok/${#files[@]} succeeded"
  rm -f "$mfa_config"
}

# ── Run MFA ───────────────────────────────────────────────────────────────────
run_mfa_group "Chinese" "$ACOUSTIC_ZH" "$DICT_ZH" "${ZH_FILES[@]+"${ZH_FILES[@]}"}"
run_mfa_group "English" "$ACOUSTIC_EN" "$DICT_EN" "${EN_FILES[@]+"${EN_FILES[@]}"}"

# ── Summary ───────────────────────────────────────────────────────────────────
hr
log "Output TextGrids:"
for tg in "$OUTPUT_DIR"/*.TextGrid; do
  [[ -f "$tg" ]] && log "  $(basename "$tg")"
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  log "Skipped (no transcript): ${MISSING[*]}"
fi

hr
log "Done. TextGrids saved to: $OUTPUT_DIR"
