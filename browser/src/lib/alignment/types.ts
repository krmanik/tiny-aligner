export interface LexiconData {
  lexicon: Record<string, string[]>;
  phone2idx: Record<string, number>;
  idx2phone: Record<string, string>;
}

export interface TimedToken {
  word: string;
  start: number;
  end: number;
}

export interface AlignmentResult {
  duration: number;
  phones: string[];
  words: TimedToken[];
  chars: TimedToken[];
  srt: string;
}

export interface TextToPhonesResult {
  tokens: string[];
  phones: string[];
  phoneIds: number[];
  /** Number of phones contributed by each token (parallel to tokens[]).
   *  Used by ctc.ts to advance the phone offset — must stay in sync with
   *  what tokenToPhones actually returned for each token. */
  tokenPhoneCounts: number[];
}

export interface ExampleItem {
  id: string;
  audio: string;
  text: string;
}
