<script lang="ts">
  import { onMount } from 'svelte';
  import { loadExamples } from './lib/alignment/examples';
  import { runForcedAlignment, warmupModel } from './lib/alignment/engine';
  import type { AlignmentResult, ExampleItem, TimedToken } from './lib/alignment/types';

  // ── State ────────────────────────────────────────────────────────────────────
  let examples: ExampleItem[] = $state([]);
  let selectedExample: ExampleItem | null = $state(null);

  // Custom upload
  let customAudioFile: File | null = $state(null);
  let customAudioUrl: string | null = $state(null);
  let customText: string = $state('');

  // Active audio source (either from example or custom)
  let audioSrc: string = $state('');
  let activeText: string = $state('');

  let result: AlignmentResult | null = $state(null);
  let statusType: 'idle' | 'running' | 'done' | 'error' = $state('idle');
  let statusMsg: string = $state('');

  let activeTab: 'words' | 'chars' | 'srt' | 'timeline' = $state('words');
  let showContent: boolean = $state(false);

  // Audio element ref
  let audioEl: HTMLAudioElement | undefined = $state(undefined);
  let rafId: number | null = null;

  // ── Init ─────────────────────────────────────────────────────────────────────
  onMount(async () => {
    examples = await loadExamples();
    // Warm up ONNX session and lexicon in background
    warmupModel().catch(() => {});
  });

  // ── Example selection ─────────────────────────────────────────────────────────
  function selectExample(ex: ExampleItem) {
    selectedExample = ex;
    customAudioFile = null;
    customAudioUrl = null;
    customText = '';
    audioSrc = ex.audio;
    activeText = ex.text;
    showContent = true;
    result = null;
    statusType = 'idle';
    statusMsg = '';
    stopRAF();
  }

  // ── Custom upload ─────────────────────────────────────────────────────────────
  function handleAudioFile(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    customAudioFile = file;
    if (customAudioUrl) URL.revokeObjectURL(customAudioUrl);
    customAudioUrl = URL.createObjectURL(file);
    selectedExample = null;
    audioSrc = customAudioUrl;
    showContent = true;
    result = null;
    statusType = 'idle';
    statusMsg = '';
    stopRAF();
  }

  function handleTextPaste(e: Event) {
    customText = (e.target as HTMLTextAreaElement).value;
    if (customAudioFile) activeText = customText;
  }

  function activateCustom() {
    if (!customAudioFile || !customText.trim()) return;
    selectedExample = null;
    audioSrc = customAudioUrl!;
    activeText = customText;
    showContent = true;
    result = null;
    statusType = 'idle';
    statusMsg = '';
    stopRAF();
  }

  // ── Alignment ─────────────────────────────────────────────────────────────────
  async function runAlign() {
    if (!audioSrc || !activeText.trim()) return;
    statusType = 'running';
    statusMsg = 'Aligning…';
    result = null;
    stopRAF();

    try {
      const source = customAudioFile && audioSrc === customAudioUrl
        ? customAudioFile
        : audioSrc;
      result = await runForcedAlignment(source, activeText);
      statusType = 'done';
      statusMsg = `Done — ${result.words.length} words, ${result.duration.toFixed(2)}s`;
      if (audioEl && !audioEl.paused) startRAF();
    } catch (err: unknown) {
      statusType = 'error';
      statusMsg = (err instanceof Error ? err.message : String(err));
    }
  }

  // ── Playback sync ─────────────────────────────────────────────────────────────
  function startRAF() {
    stopRAF();
    rafId = requestAnimationFrame(tick);
  }
  function stopRAF() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  }
  function tick() {
    if (audioEl) syncHighlight(audioEl.currentTime);
    rafId = requestAnimationFrame(tick);
  }

  // Highlight tokens — store as $state for reactivity
  let activeWordIdx: number = $state(-1);
  let activeCharIdx: number = $state(-1);
  let timelineActiveIdx: number = $state(-1);

  function syncHighlight(t: number) {
    if (!result) return;

    let wi = -1;
    for (let i = 0; i < result.words.length; i++) {
      if (t >= result.words[i].start && t < result.words[i].end) { wi = i; break; }
    }
    activeWordIdx = wi;

    let ci = -1;
    for (let i = 0; i < result.chars.length; i++) {
      if (t >= result.chars[i].start && t < result.chars[i].end) { ci = i; break; }
    }
    activeCharIdx = ci;
    timelineActiveIdx = wi;
  }

  function seekTo(token: TimedToken) {
    if (!audioEl) return;
    audioEl.currentTime = token.start;
    audioEl.play();
  }

  // ── SRT helpers ───────────────────────────────────────────────────────────────
  function copySRT() {
    if (!result) return;
    navigator.clipboard.writeText(result.srt).then(() => {
      statusType = 'done';
      statusMsg = 'SRT copied!';
      setTimeout(() => { statusType = 'idle'; statusMsg = ''; }, 2000);
    });
  }
  function downloadSRT() {
    if (!result) return;
    const a = document.createElement('a');
    a.href = 'data:text/srt;charset=utf-8,' + encodeURIComponent(result.srt);
    a.download = (selectedExample?.id ?? 'output') + '.srt';
    a.click();
  }

  // ── Audio events ──────────────────────────────────────────────────────────────
  function onPlay() { if (result) startRAF(); }
  function onPause() { stopRAF(); }
  function onSeeked() {
    if (result && audioEl) syncHighlight(audioEl.currentTime);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────
  function fmt(s: number): string { return s.toFixed(3) + 's'; }
  function pct(tok: TimedToken, dur: number): number {
    return ((tok.end - tok.start) / dur) * 100;
  }
  function leftPct(tok: TimedToken, dur: number): number {
    return (tok.start / dur) * 100;
  }
</script>

<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2">
    <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
  </svg>
  <h1>TinyAligner</h1>
  <span class="badge">Chinese Forced Alignment</span>
</header>

<div class="layout">

  <!-- Sidebar -->
  <aside class="sidebar">
    <h2>Examples</h2>
    {#each examples as ex}
      <div
        class="file-card"
        class:active={selectedExample?.id === ex.id}
        onclick={() => selectExample(ex)}
        role="button"
        tabindex="0"
        onkeydown={(e) => e.key === 'Enter' && selectExample(ex)}
      >
        <div class="fc-id">{ex.id}</div>
        <div class="fc-text">{ex.text}</div>
      </div>
    {/each}

    <h2 style="margin-top:18px">Custom Upload</h2>
    <div class="upload-area">
      <div>
        <label for="audio-upload">Audio file (WAV)</label>
        <input id="audio-upload" type="file" accept="audio/*" onchange={handleAudioFile} />
      </div>
      <div>
        <label for="text-input">Chinese text</label>
        <textarea
          id="text-input"
          placeholder="输入要对齐的文本…"
          oninput={handleTextPaste}
          value={customText}
        ></textarea>
      </div>
      <button
        class="btn"
        style="align-self:flex-start"
        disabled={!customAudioFile || !customText.trim()}
        onclick={activateCustom}
      >Use Custom</button>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">

    {#if !showContent}
      <div class="placeholder">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" opacity=".3">
          <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        <span>Select an example or upload a file to begin</span>
      </div>
    {:else}

      <!-- Player row -->
      <div class="player-row">
        <!-- svelte-ignore a11y_media_has_caption -->
        <audio
          bind:this={audioEl}
          src={audioSrc}
          controls
          preload="auto"
          onplay={onPlay}
          onpause={onPause}
          onseeked={onSeeked}
        ></audio>
        <button class="btn" disabled={statusType === 'running'} onclick={runAlign}>
          Generate TextGrid
        </button>
        <span class="status {statusType}">
          {#if statusType === 'running'}<span class="spinner"></span>{/if}
          {statusMsg}
        </span>
      </div>

      <!-- Results -->
      {#if result}
        <div>
          <div class="tabs">
            {#each (['words', 'chars', 'srt', 'timeline'] as const) as tab}
              <button
                class="tab-btn"
                class:active={activeTab === tab}
                onclick={() => (activeTab = tab)}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            {/each}
          </div>

          <!-- Words -->
          {#if activeTab === 'words'}
            <div class="tab-pane active">
              <div class="subtitle-box">
                {#each result.words as tok, i}
                  <span
                    class="word-token"
                    class:highlight={activeWordIdx === i}
                    class:past={audioEl && audioEl.currentTime >= tok.end}
                    onclick={() => seekTo(tok)}
                    role="button"
                    tabindex="0"
                    onkeydown={(e) => e.key === 'Enter' && seekTo(tok)}
                  >{tok.word}</span>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Chars -->
          {#if activeTab === 'chars'}
            <div class="tab-pane active">
              <div class="subtitle-box">
                {#each result.chars as tok, i}
                  <span
                    class="word-token"
                    class:highlight={activeCharIdx === i}
                    class:past={audioEl && audioEl.currentTime >= tok.end}
                    onclick={() => seekTo(tok)}
                    role="button"
                    tabindex="0"
                    onkeydown={(e) => e.key === 'Enter' && seekTo(tok)}
                  >{tok.word}</span>
                {/each}
              </div>
            </div>
          {/if}

          <!-- SRT -->
          {#if activeTab === 'srt'}
            <div class="tab-pane active">
              <div class="srt-box">{result.srt}</div>
              <div class="action-row">
                <button class="btn btn-ghost" onclick={copySRT}>Copy SRT</button>
                <button class="btn btn-ghost" onclick={downloadSRT}>Download .srt</button>
              </div>
            </div>
          {/if}

          <!-- Timeline -->
          {#if activeTab === 'timeline'}
            <div class="tab-pane active">
              <div class="timeline-wrap">
                <h3>Word Timestamps</h3>
                {#each result.words as tok, i}
                  <div
                    class="t-row"
                    class:thl={timelineActiveIdx === i}
                    onclick={() => seekTo(tok)}
                    role="button"
                    tabindex="0"
                    onkeydown={(e) => e.key === 'Enter' && seekTo(tok)}
                  >
                    <span class="t-word">{tok.word}</span>
                    <span class="t-times">{fmt(tok.start)} → {fmt(tok.end)}</span>
                    <div class="t-bar-wrap">
                      <div
                        class="t-bar"
                        style="width:{pct(tok, result.duration).toFixed(1)}%; margin-left:{leftPct(tok, result.duration).toFixed(1)}%"
                      ></div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {:else if statusType === 'idle' && showContent}
        <div class="placeholder" style="flex:0;padding:24px 0">
          <span style="color:var(--muted);font-size:.85rem">Click "Generate TextGrid" to align audio with text</span>
        </div>
      {/if}

    {/if}

  </main>
</div>
