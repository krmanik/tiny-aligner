# Forced Alignment Benchmark Report

Generated: 2026-05-12 15:22:30

## Methods

| Method | Description |
|--------|-------------|
| **TinyAligner** | Lightweight CTC model (~480K params) trained on AISHELL-1 + LibriSpeech |
| **MFA** | Montreal Forced Aligner — Kaldi GMM/HMM, pre-trained `english_mfa` + `mandarin_mfa` |
| **Gentle** | lowerquality/gentle — Kaldi-based, English word-level alignment via HTTP API |

## Results

| Method | F1@20ms | F1@40ms | F1@80ms | Prec@20ms | Rec@20ms | MAE(ms) | RTF | Size(MB) |
|--------|---------|---------|---------|-----------|----------|---------|-----|---------|
| TinyAligner | 0.2672 | 0.3731 | 0.6022 | 0.2698 | 0.2689 | 9.2 | 0.0214 | 1 |

## Per-File Results

### TinyAligner

| File | Duration | F1@20ms | F1@40ms | F1@80ms | RTF |
|------|----------|---------|---------|---------|-----|
| test_audio_000 | 4.24s | 0.0789 | 0.2105 | 0.5526 | 0.048 |
| test_audio_004 | 25.76s | 0.4554 | 0.5357 | 0.6518 | 0.017 |

## Metric Definitions

| Metric | Definition |
|--------|------------|
| **F1@Xms** | Boundary F1 with ±X ms match tolerance |
| **MAE** | Mean Absolute Error (ms) on matched boundary pairs |
| **RTF** | Real-Time Factor = inference_time / audio_duration (lower = faster) |

Boundaries compared are the **start-time** of each predicted segment.
Greedy one-to-one matching within tolerance.