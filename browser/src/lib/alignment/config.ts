export const SAMPLE_RATE = 16_000;
export const N_MELS = 40;
export const WIN_LENGTH = 400;
export const HOP_LENGTH = 160;
export const N_FFT = 512;
export const F_MIN = 80;
export const F_MAX = 7600;

export const FRAME_SHIFT_MS = 10;
export const DOWNSAMPLE_FACTOR = 2;
export const OUTPUT_FRAME_S = (FRAME_SHIFT_MS * DOWNSAMPLE_FACTOR) / 1000;
