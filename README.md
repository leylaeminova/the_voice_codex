# The Voice Codex

A deterministic voice-identity pipeline that extracts acoustic fingerprints from speech,
aggregates them incrementally into a collective state, and verifies whether a given voice
was part of that collective — all without storing raw audio.

---

## Pipeline Overview

```
Audio file(s)
    │
    ▼
extract.py          →  82-dim Voice Essence vector per file
    │
    ▼
aggregate.py        →  Welford online aggregation (O(1)/voice)
    │                   outputs: features.npz · features.csv · features_aggregate.npz
    ▼
verify.py           →  membership scoring + accuracy experiment + dataset QC
    │
    ▼
analysis.ipynb      →  EDA, PCA, t-SNE, correlation heatmap, full written report
```

---

## File Structure

```
the_voice_codex/
├── extract.py               # Trial I  — per-file Voice Essence extraction (82 dims)
├── aggregate.py             # Trial II — Welford online aggregation + cluster-then-aggregate
├── verify.py                # Trial III — membership verification + accuracy experiment
├── test_determinism.py      # Trial IV — 21-test bit-exact reproducibility suite
├── analysis.ipynb           # Report   — EDA, visualisations, design reasoning, limitations
├── requirements.txt         # Pinned Python dependencies
├── README.md
├── essences.npz             # Pre-built essence matrix (45 × 82) with speaker labels
├── aggregate_state.npz      # Pre-built Welford state for the full 45-voice corpus
├── verification_accuracy.png
├── roc_curves.png
└── voice_codex_dataset/     # ⚠ NOT included in repo — see Dataset Setup below
    └── data/
        ├── manifest.json
        └── samples/
            ├── speaker_01/  # 3 .wav clips each, 15 speakers total
            ...
            └── speaker_15/
```

---

## Dataset Setup

> **The `voice_codex_dataset/` folder is not included in this repository.**
> Place it at the project root (next to `extract.py`) before running any pipeline commands.

Expected layout after placement:

```
the_voice_codex/
├── extract.py
├── aggregate.py
...
└── voice_codex_dataset/
    └── data/
        ├── manifest.json
        └── samples/
            ├── speaker_01/
            │   ├── clip_01.wav
            │   ├── clip_02.wav
            │   └── clip_03.wav
            ├── speaker_02/
            ...
            └── speaker_15/
```

The pipeline works with **any directory of `.wav` files** — you can substitute
`voice_codex_dataset/data/samples/` with your own folder in every command below.
The dataset integration tests in `test_determinism.py` are automatically skipped
if the folder is absent; all other tests use synthetic tones and run without it.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
```

### 2. Trial I — Extract a single voice essence

```bash
python extract.py voice_codex_dataset/data/samples/speaker_01/clip_01.wav
```

Prints the 82-dim vector segmented by feature group.

### 3. Trial II — Aggregate the full dataset (Welford, O(1)/voice)

```bash
python aggregate.py voice_codex_dataset/data/samples/ --output features.npz
```

Optional flags:
- `--cluster` — runs K-Means and HDBSCAN cluster-then-aggregate comparison
- `--n-clusters N` — number of K-Means clusters (default 5)
- `--o1-check` — prints per-update timings to empirically verify O(1) behaviour

### 4. Trial III — Membership verification

Score a single voice against the aggregate:
```bash
python verify.py score features_aggregate.npz essences.npz --index 0 --mode mahalanobis
```

Run the full accuracy-vs-chorus-size experiment (100 random trials per size):
```bash
python verify.py experiment essences.npz --trials 100 --chart verification_accuracy.png --roc roc_curves.png
```

Run dataset QC (NaN / bounds / duplicates / outliers):
```bash
python verify.py qc features.npz
```

### 5. Trial IV — Determinism gate

```bash
pytest test_determinism.py -v          # full 21-test suite
pytest test_determinism.py -v -k pipeline   # just the core gate
python test_determinism.py             # standalone, no pytest needed
```

### 6. Analysis notebook

```bash
jupyter lab analysis.ipynb
```

---

## Features Extracted (ESSENCE_DIM = 82)

| Slice | Segment | Dims | Tool | Description |
|-------|---------|------|------|-------------|
| [0:5] | F0 stats | 5 | Parselmouth | Pitch mean, std, min, max (Hz); voiced fraction |
| [5:8] | F0 trajectory | 3 | Parselmouth | Slope (Hz/s); vibrato rate (Hz); vibrato depth (cents RMS) |
| [8:21] | MFCC mean | 13 | librosa | Vocal tract shape — per-coefficient temporal means |
| [21:34] | MFCC std | 13 | librosa | Vocal tract shape — per-coefficient temporal stds |
| [34:47] | Delta-MFCC mean | 13 | librosa | Rate of vocal tract change — means |
| [47:60] | Delta-MFCC std | 13 | librosa | Rate of vocal tract change — stds |
| [60:66] | Formants F1–F3 | 6 | Parselmouth | Resonance frequencies — mean & std per formant (Hz) |
| [66:68] | HNR | 2 | Parselmouth | Harmonics-to-noise ratio — mean & std (dB) |
| [68:72] | Amplitude | 4 | librosa/scipy | RMS envelope — mean, std, skewness, excess kurtosis |
| [72:78] | Spectral | 6 | librosa | Centroid mean/std; bandwidth mean; rolloff mean; ZCR mean/std |
| [78:82] | Onset | 4 | librosa | Onset rate (evt/s); strength mean/std; mean inter-onset interval (s) |

**Total: 82 dimensions** across 11 segments covering pitch, vocal tract shape, formants,
voice quality, loudness contour, spectral brightness, and articulation rhythm.

---

## Aggregation Algorithm

Welford's online algorithm (Welford 1962) maintains a running `(count, mean, M2)` triple:

```
delta  = x_new − mean_old
mean  += delta / count
delta2 = x_new − mean_new
M2    += delta × delta2
```

Each update is **O(1)** — no previously-seen voices are re-processed. Adding voice
number 1,000,001 takes the same time as adding voice number 2 (~1.5 µs empirically).
The state is sufficient to recover mean and per-dimension variance at any point, which
feeds directly into the Mahalanobis membership scorer.

Parallel shards can be merged exactly using Chan et al.'s formula (1979) without
re-processing any individual voices.

---

## Membership Verification

Given an `AggregateState` and a candidate essence vector, three scoring modes are available:

| Mode | Formula | Notes |
|------|---------|-------|
| `mahalanobis` | `−√Σ((xᵢ−μᵢ)²/σᵢ²)` | Default; uses both mean and variance; most discriminative |
| `cosine` | `(x·μ)/(‖x‖‖μ‖)` | Direction-only; ignores per-dimension variance |
| `log_likelihood` | `−½Σ((xᵢ−μᵢ)²/σᵢ² + log(2πσᵢ²))` | Gaussian model; equivalent ranking to Mahalanobis |

A chi-squared p-value (`p_value()`) calibrates the Mahalanobis distance under a diagonal
Gaussian model: `d² ~ χ²(82)` for a voice drawn from the aggregate distribution.

**Key result:** AUC-ROC degrades from ~0.99 at N=3 to ~0.59 at N=40. The aggregate
mean dilutes each individual contribution by 1/N — the crystal's memory blurs as the
chorus grows. See `analysis.ipynb` §Report for full interpretation.

---

## Reproducibility Guarantees

| Source of non-determinism | Fix |
|---|---|
| K-Means random centroid init | `random_state=42` hard-coded in `cluster_aggregate()` |
| Welford FP order-dependence | Files always processed in sorted path order |
| forge_update in-place arrays | `forge_init()` allocates fresh arrays; no shared references |
| librosa audio loading | Fixed `sr=16000, mono=True` via soundfile backend |
| Parselmouth/Praat state | New `Sound` object per call; no shared Praat state |
| BLAS matmul (cosine mode) | Arrays are tiny (45×82); BLAS uses sequential code at this size |

The 21-test suite in `test_determinism.py` verifies bit-exact equality across in-process
runs, subprocess boundaries, and save/load round-trips.

---

## Dataset

15 synthetic speakers, 3 clips each (45 total), generated with `espeak` (speakers 1–9)
and `flite` (speakers 10–15) across 11 distinct voice profiles. Each clip contains a
different English tongue-twister or literary sentence (~3–5 seconds).

**Note:** TTS-synthesized voices share more acoustic properties than real human speech
(especially within the same engine). Verification accuracy on real human voices would
differ — likely harder to distinguish at the same chorus sizes.

---

## Design Decisions

**Why 82 dimensions, not fewer?**
Each segment targets a different acoustic facet of speaker identity. MFCCs (52 dims)
are the backbone — they capture vocal tract geometry and dominate speaker-recognition
literature. Formants (6 dims) are expensive but irreplaceable for vocal tract shape.
F0 trajectory (8 dims) captures pitch dynamics. HNR, amplitude, spectral, and onset
features add complementary axes. Dropping any segment degrades separability; the
inter/intra speaker distance ratio in the corpus is ~1.4× in standardised space.

**Why diagonal Mahalanobis, not full covariance?**
Welford's online algorithm naturally produces per-dimension variance in O(D) space.
Full covariance would require O(D²) = 6,724 extra floats per aggregate state and an
online update formula that is significantly more complex. The diagonal approximation
is still substantially better than raw Euclidean distance because it down-weights
high-variance (noisy) dimensions and amplifies stable ones.

**Why not a neural embedding (x-vectors, ECAPA-TDNN)?**
Classical features are interpretable, fast, and require no GPU or pre-trained model.
The quest explicitly discourages GPU-trained deep learning. The pipeline is designed
to be transparent — every dimension has a documented acoustic interpretation.

---

## Known Limitations

- **Chorus size ceiling:** With 45 clips, the "50+ voices" experiment point is not
  reachable with the provided dataset alone. Results are reported up to N=40.
- **TTS homogeneity:** All voices are synthesized. Real human voices with recording
  condition variation would yield lower AUC at equivalent chorus sizes.
- **Vibrato depth QC:** `speaker_13` (flite `kal16` voice) has an unusually wide F0
  range (~70–80 Hz std vs ~7 Hz for espeak voices), causing the vibrato depth feature
  to exceed the 500-cent QC bound. This is a data characteristic, not a code error.
- **Diagonal covariance:** Features are treated as independent. Real MFCC coefficients
  are correlated; full Mahalanobis would improve AUC at all chorus sizes.

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for pinned versions.

## License

MIT
