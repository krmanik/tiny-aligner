declare module 'onnxruntime-web' {
  export namespace env {
    namespace wasm {
      let numThreads: number;
      let wasmPaths: string | Record<string, string>;
      let simd: boolean;
    }
  }

  export type TensorType = 'float32' | 'float64' | 'int32' | 'int64' | 'bool';

  export class Tensor {
    constructor(type: TensorType, data: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array, dims: number[]);
    readonly data: Float32Array & Float64Array & Int32Array & BigInt64Array & Uint8Array;
    readonly dims: ReadonlyArray<number>;
  }

  export interface SessionOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
    intraOpNumThreads?: number;
    interOpNumThreads?: number;
  }

  export class InferenceSession {
    static create(path: string, options?: SessionOptions): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }
}

