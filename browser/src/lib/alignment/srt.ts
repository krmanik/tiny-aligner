import type { TimedToken } from './types';

export function buildSRT(tokens: TimedToken[]): string {
  const lines: Array<{ start: number; end: number; text: string }> = [];

  let group: TimedToken[] = [];
  let groupStart: number | null = null;

  tokens.forEach((t) => {
    if (groupStart === null) groupStart = t.start;
    group.push(t);
    const elapsed = t.end - groupStart;
    if (group.length >= 10 || elapsed >= 4.0) {
      lines.push({
        start: groupStart,
        end: t.end,
        text: group.map((x) => x.word).join(''),
      });
      group = [];
      groupStart = null;
    }
  });

  if (group.length) {
    lines.push({
      start: groupStart ?? 0,
      end: group[group.length - 1].end,
      text: group.map((x) => x.word).join(''),
    });
  }

  return lines
    .map((line, i) => `${i + 1}\n${srtTime(line.start)} --> ${srtTime(line.end)}\n${line.text}`)
    .join('\n\n');
}

function srtTime(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = Math.floor(s % 60);
  const ms = Math.round((s % 1) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(ss)},${ms.toString().padStart(3, '0')}`;
}

function pad(n: number): string {
  return String(n).padStart(2, '0');
}
