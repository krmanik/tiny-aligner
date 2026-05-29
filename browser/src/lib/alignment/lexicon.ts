import initJieba, { cut } from 'jieba-wasm';
import type { LexiconData, TextToPhonesResult } from './types';

let lexiconCache: LexiconData | null = null;
let jiebaReady = false;

function normalizePhoneMap(raw: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  Object.entries(raw).forEach(([k, v]) => {
    out[k] = Number(v);
  });
  return out;
}

export async function loadLexicon(): Promise<LexiconData> {
  if (lexiconCache) return lexiconCache;

  const res = await fetch(`${import.meta.env.BASE_URL}lexicon.json`);
  if (!res.ok) {
    throw new Error(`Failed to load lexicon.json: ${res.status}`);
  }
  const raw = (await res.json()) as LexiconData;

  lexiconCache = {
    lexicon: raw.lexicon,
    phone2idx: normalizePhoneMap(raw.phone2idx),
    idx2phone: raw.idx2phone,
  };
  return lexiconCache;
}

async function ensureJieba(): Promise<void> {
  if (jiebaReady) return;
  try {
    await initJieba();
    jiebaReady = true;
  } catch {
    jiebaReady = false;
  }
}

/** True if the text contains no CJK characters (pure Latin/punctuation). */
function isEnglishOnly(text: string): boolean {
  return !/[\u3000-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F]/.test(text);
}

export async function tokenizeChinese(text: string): Promise<string[]> {
  const clean = text.trim();
  if (!clean) return [];

  // Pre-split: keep Chinese runs and English words as separate tokens
  // e.g. "这是一个word例子" → ["这是一个", "word", "例子"]
  const segments: string[] = [];
  const RE = /([A-Za-z0-9'\-]+)|([^A-Za-z0-9'\-]+)/g;
  let m: RegExpExecArray | null;
  while ((m = RE.exec(clean)) !== null) {
    const s = m[0].trim();
    if (s) segments.push(s);
  }

  const tokens: string[] = [];
  for (const seg of segments) {
    if (/^[A-Za-z0-9'\-]+$/.test(seg)) {
      // English word — emit as-is (lexicon lookup is case-insensitive)
      tokens.push(seg);
    } else if (/\s/.test(seg)) {
      tokens.push(...seg.split(/\s+/).filter(Boolean));
    } else {
      // Strip pure punctuation before feeding to jieba so "word。" → "word"
      const noPunct = seg.replace(/[\p{P}\p{S}]/gu, '').trim();
      if (!noPunct) continue; // skip punctuation-only segments entirely
      // Chinese segment — run jieba
      await ensureJieba();
      if (jiebaReady) {
        const segmented = cut(noPunct, true).map((x) => x.trim()).filter(Boolean);
        if (segmented.length > 0) {
          tokens.push(...segmented);
          continue;
        }
      }
      tokens.push(...Array.from(noPunct));
    }
  }

  return tokens;
}

export async function textToPhones(text: string, data: LexiconData): Promise<TextToPhonesResult> {
  const tokens = await tokenizeChinese(text);
  const phones: string[] = [];
  const phoneIds: number[] = [];
  const tokenPhoneCounts: number[] = [];

  const silence = data.phone2idx.sil !== undefined ? 'sil' : null;
  const unkToken = '<unk>';
  const unkId = data.phone2idx[unkToken] ?? 0;

  if (silence) {
    phones.push(silence);
    phoneIds.push(data.phone2idx[silence]);
  }

  for (const token of tokens) {
    const p = tokenToPhones(token, data.lexicon, unkToken, data.phone2idx);
    tokenPhoneCounts.push(p.length);
    phones.push(...p);
    phoneIds.push(...p.map((ph) => data.phone2idx[ph] ?? unkId));
  }

  if (silence) {
    phones.push(silence);
    phoneIds.push(data.phone2idx[silence]);
  }

  return { tokens, phones, phoneIds, tokenPhoneCounts };
}

export function tokenToPhones(
  token: string,
  lexicon: Record<string, string[]>,
  unkToken = '<unk>',
  phone2idx?: Record<string, number>,
): string[] {
  if (!token) return [];
  // Case-insensitive lookup: try as-is, uppercase, lowercase
  const entry = lexicon[token] ?? lexicon[token.toUpperCase()] ?? lexicon[token.toLowerCase()];
  if (entry) {
    // If phone2idx is provided, check whether the model actually knows these phones.
    // If ALL phones are unknown (e.g. ARPAbet phones on a Chinese-only model), fall
    // through to the character-based <unk> fallback so CTC offsets stay consistent.
    if (!phone2idx || entry.some((ph) => phone2idx[ph] !== undefined && ph !== unkToken)) {
      return [...entry];
    }
  }

  const out: string[] = [];
  let hasAnyKnown = false;
  for (const ch of Array.from(token)) {
    if (lexicon[ch]) {
      out.push(...lexicon[ch]);
      hasAnyKnown = true;
    } else {
      out.push(unkToken);
    }
  }

  // No chars found in lexicon → this is a fully foreign token (e.g. English word
  // on Chinese model). Return exactly 1 <unk> so CTC offsets stay proportional
  // to word count, not character count.
  if (!hasAnyKnown) return [unkToken];
  return out;
}
