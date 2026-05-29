declare module 'jieba-wasm/web' {
  export default function init(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module): Promise<void>;
  export function cut(text: string, hmm?: boolean): string[];
}
