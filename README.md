# The Voice Codex

Deterministic acoustic feature extraction pipeline for speech and voice analysis.

## Overview

The Voice Codex extracts a reproducible, fixed-dimension feature vector from any audio
file, covering fundamental frequency (F0), harmonics-to-noise ratio (HNR), MFCCs,
spectral shape, and energy features.

```
Audio file(s)
    │
    ▼
extract.py          ←  per-file FeatureVector  (F0 · HNR · MFCCs · spectral)
    │
    ▼
aggregate.py        ←  batch N files  →  features.npz  +  features.csv
    │
    ▼
verify.py           ←  QC checks  (NaN / Inf · bounds · duplicates · outliers)
    │
    ▼
analysis.ipynb      ←  EDA, plots, PCA, correlation heatmap
```

## File Structure

```
the_voice_codex/
├── extract.py           # Core per-file feature extraction
├── aggregate.py         # Batch runner → .npz / .csv
├── verify.py            # QC and sanity checks
├── test_determinism.py  # Pytest suite: bit-exact reproducibility
├── analysis.ipynb       # Exploratory analysis notebook
├── requirements.txt     # Pinned dependencies
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract features from a single file

```bash
python extract.py path/to/audio.wav
```

### 3. Batch-aggregate a directory

```bash
python aggregate.py path/to/audio_dir/ --output features.npz
```

### 4. Verify the feature archive

```bash
python verify.py features.npz
```

### 5. Run determinism tests

```bash
pytest test_determinism.py -v
```

### 6. Explore in the notebook

```bash
jupyter lab analysis.ipynb
```

## Features Extracted

| Feature | Tool | Description |
|---------|------|-------------|
| F0 mean / std / min / max | Parselmouth (Praat) | Fundamental frequency statistics over voiced frames |
| Voiced fraction | Parselmouth | Fraction of frames with detected voicing |
| HNR mean | Parselmouth (cc) | Harmonics-to-noise ratio in dB |
| RMS mean / std | librosa | Short-time root-mean-square energy |
| Spectral centroid | librosa | Brightness of the spectrum |
| Spectral bandwidth | librosa | Spread of the spectrum |
| Spectral rolloff | librosa | Frequency below 85 % of energy |
| ZCR mean | librosa | Zero-crossing rate |
| MFCCs (×13) mean + std | librosa | Mel-frequency cepstral coefficients |
| Delta-MFCCs (×13) mean + std | librosa | First-order MFCC derivatives |

**Total feature dimension:** 13 scalar + 4 × 13 MFCC = **65**

## Reproducibility Guarantees

- `librosa.load` is called with a fixed `sr` and `mono=True` — no random resampling.
- Parselmouth/Praat pitch and harmonicity algorithms are deterministic given identical input.
- NumPy random state is not touched by any extraction code.
- `test_determinism.py` verifies bit-exact equality across two sequential calls and across
  a subprocess boundary.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for pinned versions.

## License

MIT
