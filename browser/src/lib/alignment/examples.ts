import type { ExampleItem } from './types';

export async function loadExamples(): Promise<ExampleItem[]> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}examples/manifest.json`);
    if (!res.ok) return [];
    const data = (await res.json()) as ExampleItem[];
    const base = import.meta.env.BASE_URL.replace(/\/$/, '');
    return data.map((x) => ({
      id: x.id,
      audio: `${base}${x.audio}`,
      text: x.text,
    }));
  } catch {
    return [];
  }
}

export async function loadExampleText(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load text file: ${path}`);
  }
  return (await res.text()).trim();
}
