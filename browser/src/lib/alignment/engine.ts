import * as ort from 'onnxruntime-web';
import { extractLogMel, decodeToMono16k } from './audio';
import { OUTPUT_FRAME_S } from './config';
import { ctcForcedAlign, extractSegments, buildWordIntervals, buildCharIntervals } from './ctc';
import { loadLexicon, textToPhones } from './lexicon';
import { buildSRT } from './srt';
import type { AlignmentResult } from './types';

let sessionPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (sessionPromise) return sessionPromise;

  ort.env.wasm.numThreads = 1;

  sessionPromise = ort.InferenceSession.create(`${import.meta.env.BASE_URL}model.onnx`, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return sessionPromise;
}

export async function runForcedAlignment(audioSource: File | string, text: string): Promise<AlignmentResult> {
  const cleanText = text.trim();
  if (!cleanText) {
    throw new Error('Text is empty.');
  }

  const lex = await loadLexicon();
  const { tokens, phones, phoneIds, tokenPhoneCounts } = await textToPhones(cleanText, lex);
  if (!tokens.length || !phoneIds.length) {
    throw new Error('Unable to tokenize text into phonemes.');
  }

  const waveform = await decodeToMono16k(audioSource);
  const features = extractLogMel(waveform);

  const session = await getSession();
  const input = new ort.Tensor('float32', features.mel, [1, features.frames, features.nMels]);
  const outputs = await session.run({ mel: input });

  const tensor = outputs.log_probs;
  if (!tensor) {
    throw new Error('Model output log_probs missing.');
  }

  const logProbs = tensor.data as Float32Array;
  const [timeSteps, vocabSize] = tensor.dims as [number, number];

  const blankIdx = lex.phone2idx['<blank>'] ?? 0;
  const framePhoneIdx = ctcForcedAlign(logProbs, timeSteps, vocabSize, phoneIds, blankIdx);

  const phoneIntervals = extractSegments(framePhoneIdx, phones, OUTPUT_FRAME_S).filter((x) => x.word);
  const words = buildWordIntervals(tokens, phoneIntervals, lex.lexicon, tokenPhoneCounts);
  const chars = buildCharIntervals(tokens, phoneIntervals, lex.lexicon, tokenPhoneCounts);

  return {
    duration: features.duration,
    phones,
    words,
    chars,
    srt: buildSRT(words),
  };
}

export async function warmupModel(): Promise<void> {
  await Promise.all([getSession(), loadLexicon()]);
}
