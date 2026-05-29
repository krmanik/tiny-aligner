import type { TimedToken } from './types';

export function ctcForcedAlign(
  logProbs: Float32Array,
  timeSteps: number,
  vocabSize: number,
  phoneIds: number[],
  blankIdx = 0,
): number[] {
  const forcedPath: number[] = [blankIdx];
  for (const p of phoneIds) {
    forcedPath.push(p);
    forcedPath.push(blankIdx);
  }
  const S = forcedPath.length;
  const NEG_INF = -1e12;

  const alpha = new Float64Array(S);
  const newAlpha = new Float64Array(S);
  alpha.fill(NEG_INF);
  alpha[0] = getLP(logProbs, 0, blankIdx, vocabSize);
  if (S > 1) alpha[1] = getLP(logProbs, 0, forcedPath[1], vocabSize);

  const bp = new Int32Array(timeSteps * S);

  for (let t = 1; t < timeSteps; t++) {
    newAlpha.fill(NEG_INF);
    for (let s = 0; s < S; s++) {
      let bestPrev = s;
      let bestVal = alpha[s];

      if (s > 0 && alpha[s - 1] > bestVal) {
        bestVal = alpha[s - 1];
        bestPrev = s - 1;
      }

      if (
        s > 1 &&
        forcedPath[s] !== blankIdx &&
        forcedPath[s] !== forcedPath[s - 2] &&
        alpha[s - 2] > bestVal
      ) {
        bestVal = alpha[s - 2];
        bestPrev = s - 2;
      }

      newAlpha[s] = bestVal + getLP(logProbs, t, forcedPath[s], vocabSize);
      bp[t * S + s] = bestPrev;
    }
    alpha.set(newAlpha);
  }

  let endState = S - 1;
  if (S > 1 && alpha[S - 2] > alpha[S - 1]) endState = S - 2;

  const path = new Int32Array(timeSteps);
  let s = endState;
  for (let t = timeSteps - 1; t >= 0; t--) {
    path[t] = s;
    s = bp[t * S + s];
  }

  const framePhoneIdx: number[] = new Array(timeSteps);
  for (let t = 0; t < timeSteps; t++) {
    const state = path[t];
    framePhoneIdx[t] = state % 2 === 0 ? -1 : (state - 1) >> 1;
  }

  return framePhoneIdx;
}

function getLP(logProbs: Float32Array, t: number, v: number, vocab: number): number {
  return logProbs[t * vocab + v] ?? -1e12;
}

export function extractSegments(
  framePhoneIdx: number[],
  phones: string[],
  frameShiftS: number,
): TimedToken[] {
  if (framePhoneIdx.length === 0) return [];

  const out: TimedToken[] = [];
  let prev = framePhoneIdx[0];
  let startFrame = 0;

  for (let t = 1; t < framePhoneIdx.length; t++) {
    const idx = framePhoneIdx[t];
    if (idx !== prev) {
      out.push({
        word: prev >= 0 ? phones[prev] : '',
        start: startFrame * frameShiftS,
        end: t * frameShiftS,
      });
      startFrame = t;
      prev = idx;
    }
  }

  out.push({
    word: prev >= 0 ? phones[prev] : '',
    start: startFrame * frameShiftS,
    end: framePhoneIdx.length * frameShiftS,
  });

  return out;
}

export function buildWordIntervals(
  tokens: string[],
  phoneIntervals: TimedToken[],
  lexicon: Record<string, string[]>,
  tokenPhoneCounts: number[],
): TimedToken[] {
  const out: TimedToken[] = [];
  let phoneOffset = 1;

  for (let i = 0; i < tokens.length; i++) {
    const nPh = tokenPhoneCounts[i] ?? countTokenPhones(tokens[i], lexicon);
    const startIdx = phoneOffset - 1;
    const endIdx = phoneOffset + nPh - 1;

    out.push({
      word: tokens[i],
      start: getStartTime(phoneIntervals, startIdx),
      end: getEndTime(phoneIntervals, endIdx),
    });
    phoneOffset += nPh;
  }

  return out;
}

export function buildCharIntervals(
  tokens: string[],
  phoneIntervals: TimedToken[],
  lexicon: Record<string, string[]>,
  tokenPhoneCounts: number[],
): TimedToken[] {
  const out: TimedToken[] = [];
  let phoneOffset = 1;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nPh = tokenPhoneCounts[i] ?? countTokenPhones(token, lexicon);

    if (lexicon[token]) {
      const wordPhones = lexicon[token];
      const chars = Array.from(token);
      if (chars.length === 1) {
        out.push({
          word: chars[0],
          start: getStartTime(phoneIntervals, phoneOffset - 1),
          end: getEndTime(phoneIntervals, phoneOffset + wordPhones.length - 2),
        });
        phoneOffset += wordPhones.length;
      } else {
        const base = Math.floor(wordPhones.length / chars.length);
        const rem = wordPhones.length % chars.length;

        let offset = phoneOffset;
        chars.forEach((ch, i) => {
          const n = base + (i < rem ? 1 : 0);
          out.push({
            word: ch,
            start: getStartTime(phoneIntervals, offset - 1),
            end: getEndTime(phoneIntervals, offset + n - 2),
          });
          offset += n;
        });
        phoneOffset = offset;
      }
      continue;
    }

    // Token not in lexicon — use the pre-computed nPh slots and divide
    // time evenly across characters (covers both Chinese char fallback and
    // fully-unknown foreign words that got 1 <unk> slot).
    {
      const chars = Array.from(token);
      const tokenStart = getStartTime(phoneIntervals, phoneOffset - 1);
      const tokenEnd   = getEndTime(phoneIntervals, phoneOffset + nPh - 2);
      const dur = (tokenEnd - tokenStart) / chars.length;
      chars.forEach((ch, i) => {
        out.push({
          word: ch,
          start: tokenStart + i * dur,
          end:   tokenStart + (i + 1) * dur,
        });
      });
      phoneOffset += nPh;
    }
  }

  return out;
}

function countTokenPhones(token: string, lexicon: Record<string, string[]>): number {
  if (lexicon[token]) return lexicon[token].length;
  let count = 0;
  for (const ch of Array.from(token)) {
    count += lexicon[ch] ? lexicon[ch].length : 1;
  }
  return Math.max(count, 1);
}

function getStartTime(intervals: TimedToken[], idx: number): number {
  if (idx < 0 || idx >= intervals.length) return 0;
  return intervals[idx].start;
}

function getEndTime(intervals: TimedToken[], idx: number): number {
  if (idx < 0) return 0;
  if (idx >= intervals.length) return intervals.length ? intervals[intervals.length - 1].end : 0;
  return intervals[idx].end;
}
