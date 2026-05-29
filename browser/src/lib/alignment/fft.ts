export function hannWindow(length: number): Float32Array {
  const out = new Float32Array(length);
  if (length <= 1) return out;
  for (let n = 0; n < length; n++) {
    out[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (length - 1));
  }
  return out;
}

function reverseBits(x: number, bits: number): number {
  let y = 0;
  for (let i = 0; i < bits; i++) {
    y = (y << 1) | (x & 1);
    x >>>= 1;
  }
  return y;
}

export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  const bits = Math.log2(n);
  if (!Number.isInteger(bits)) {
    throw new Error(`FFT input length must be power of 2, got ${n}`);
  }

  for (let i = 0; i < n; i++) {
    const j = reverseBits(i, bits);
    if (j > i) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >>> 1;
    const theta = (-2 * Math.PI) / size;
    const wMulR = Math.cos(theta);
    const wMulI = Math.sin(theta);

    for (let start = 0; start < n; start += size) {
      let wR = 1;
      let wI = 0;
      for (let k = 0; k < half; k++) {
        const i = start + k;
        const j = i + half;

        const tR = wR * real[j] - wI * imag[j];
        const tI = wR * imag[j] + wI * real[j];

        real[j] = real[i] - tR;
        imag[j] = imag[i] - tI;
        real[i] = real[i] + tR;
        imag[i] = imag[i] + tI;

        const nextWR = wR * wMulR - wI * wMulI;
        const nextWI = wR * wMulI + wI * wMulR;
        wR = nextWR;
        wI = nextWI;
      }
    }
  }
}
