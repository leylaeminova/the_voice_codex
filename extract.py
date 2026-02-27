"""
extract.py — Voice Essence extraction pipeline.

Primary API
-----------
    extract_essence(wav_path: str) -> np.ndarray   # shape (82,), dtype float64

Vector layout — ESSENCE_DIM = 82
---------------------------------
Segment              Slice    Dims  Tool              Description
─────────────────────────────────────────────────────────────────────────────
F0_stats             [0:5]      5   Parselmouth       mean, std, min, max Hz;
                                                      voiced_fraction (0–1)
F0_trajectory        [5:8]      3   Parselmouth       linear slope (Hz/s);
                                                      vibrato rate (Hz, 4–8 Hz band);
                                                      vibrato depth (cents RMS)
MFCC_mean           [8:21]     13   librosa           per-coefficient temporal mean
MFCC_std           [21:34]     13   librosa           per-coefficient temporal std
DeltaMFCC_mean     [34:47]     13   librosa           delta-MFCC temporal mean
DeltaMFCC_std      [47:60]     13   librosa           delta-MFCC temporal std
Formants           [60:66]      6   Parselmouth/Praat F1 mean, F1 std,
                                                      F2 mean, F2 std,
                                                      F3 mean, F3 std  (Hz)
HNR                [66:68]      2   Parselmouth       harmonicity mean, std (dB)
Amplitude          [68:72]      4   librosa/scipy     RMS mean, std, skewness,
                                                      excess kurtosis
Spectral           [72:78]      6   librosa           centroid mean, centroid std,
                                                      bandwidth mean,
                                                      rolloff mean,
                                                      ZCR mean, ZCR std  (Hz/ratio)
Onset              [78:82]      4   librosa           onset rate (evt/s);
                                                      onset strength mean, std;
                                                      mean inter-onset interval (s)
─────────────────────────────────────────────────────────────────────────────
TOTAL                            82
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import parselmouth
import scipy.stats
from parselmouth.praat import call

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SAMPLE_RATE: int    = 16_000
HOP_LENGTH: int     = 256
N_FFT: int          = 1024
N_MFCC: int         = 13
F0_FLOOR: float     = 75.0
F0_CEILING: float   = 600.0
FORMANT_MAX: float  = 5500.0   # Praat Formant ceiling (Hz)
VIBRATO_LO: float   = 4.0      # vibrato band low  (Hz)
VIBRATO_HI: float   = 8.0      # vibrato band high (Hz)

ESSENCE_DIM: int    = 82

# Documented slice map — used by the notebook for axis labelling
ESSENCE_LAYOUT: dict[str, tuple[slice, str]] = {
    "F0_stats":       (slice( 0,  5), "F0 mean/std/min/max (Hz); voiced fraction"),
    "F0_trajectory":  (slice( 5,  8), "F0 slope (Hz/s); vibrato rate (Hz); vibrato depth (cents RMS)"),
    "MFCC_mean":      (slice( 8, 21), "MFCC C0-C12 temporal means"),
    "MFCC_std":       (slice(21, 34), "MFCC C0-C12 temporal stds"),
    "DeltaMFCC_mean": (slice(34, 47), "Delta-MFCC C0-C12 temporal means"),
    "DeltaMFCC_std":  (slice(47, 60), "Delta-MFCC C0-C12 temporal stds"),
    "Formants":       (slice(60, 66), "F1, F2, F3 — mean and std per formant (Hz)"),
    "HNR":            (slice(66, 68), "Harmonics-to-noise ratio — mean and std (dB)"),
    "Amplitude":      (slice(68, 72), "RMS envelope — mean, std, skewness, excess kurtosis"),
    "Spectral":       (slice(72, 78), "Centroid mean/std; bandwidth mean; rolloff mean; ZCR mean/std"),
    "Onset":          (slice(78, 82), "Onset rate; onset strength mean/std; mean IOI (s)"),
}


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────

def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load and resample audio to a fixed sample rate (mono)."""
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, sr_out


# ──────────────────────────────────────────────
# Block A — F0 stats + trajectory  [0:8]
# ──────────────────────────────────────────────

def _extract_f0_block(path: str | Path) -> np.ndarray:
    """
    Return 8 values: F0 stats (5) + trajectory features (3).

    F0 stats  [0:5]: mean, std, min, max (Hz), voiced_fraction
    Trajectory [5:8]: linear slope (Hz/s), vibrato rate (Hz), vibrato depth (cents RMS)

    Vibrato: detected as the dominant spectral peak in the 4–8 Hz band of the
    detrended F0 contour; depth is the RMS of cents-deviation from the trend.
    Both are 0 when the voiced segment is too short (< 20 frames).
    """
    snd = parselmouth.Sound(str(path))
    pitch = call(snd, "To Pitch", 0.0, F0_FLOOR, F0_CEILING)
    f0_all   = pitch.selected_array["frequency"]   # 0 for unvoiced
    times    = pitch.xs()
    voiced_m = f0_all > 0
    voiced   = f0_all[voiced_m]
    t_voiced = times[voiced_m]

    if len(voiced) == 0:
        return np.zeros(8)

    f0_mean  = float(np.mean(voiced))
    f0_std   = float(np.std(voiced))
    f0_min   = float(np.min(voiced))
    f0_max   = float(np.max(voiced))
    voiced_frac = len(voiced) / max(len(f0_all), 1)

    # --- F0 trajectory slope (linear Hz/s) ---
    if len(t_voiced) >= 2:
        slope = float(np.polyfit(t_voiced, voiced, 1)[0])
    else:
        slope = 0.0

    # --- Vibrato (requires ≥ 20 voiced frames) ---
    vibrato_rate  = 0.0
    vibrato_depth = 0.0
    if len(voiced) >= 20:
        # Convert to cents relative to mean, then detrend
        cents = 1200.0 * np.log2(np.maximum(voiced / f0_mean, 1e-9))
        cents_detrend = cents - np.polyval(np.polyfit(t_voiced, cents, 1), t_voiced)

        duration   = t_voiced[-1] - t_voiced[0]
        frame_rate = len(voiced) / max(duration, 1e-6)   # frames/s

        fft_mag  = np.abs(np.fft.rfft(cents_detrend))
        freqs    = np.fft.rfftfreq(len(cents_detrend), d=1.0 / frame_rate)

        mask = (freqs >= VIBRATO_LO) & (freqs <= VIBRATO_HI)
        if mask.any():
            peak_idx     = np.argmax(fft_mag[mask])
            vibrato_rate = float(freqs[mask][peak_idx])
        vibrato_depth = float(np.sqrt(np.mean(cents_detrend ** 2)))   # RMS cents

    return np.array([
        f0_mean, f0_std, f0_min, f0_max, voiced_frac,
        slope, vibrato_rate, vibrato_depth,
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Block B — MFCCs  [8:60]
# ──────────────────────────────────────────────

def _extract_mfcc_block(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """
    Return 52 values: MFCC mean (13), std (13), delta mean (13), delta std (13).
    """
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    dmfcc = librosa.feature.delta(mfcc)
    return np.concatenate([
        mfcc.mean(axis=1),  mfcc.std(axis=1),
        dmfcc.mean(axis=1), dmfcc.std(axis=1),
    ])


# ──────────────────────────────────────────────
# Block C — Formants F1/F2/F3  [60:66]
# ──────────────────────────────────────────────

def _extract_formant_block(path: str | Path) -> np.ndarray:
    """
    Return 6 values: F1 mean, F1 std, F2 mean, F2 std, F3 mean, F3 std (Hz).

    Uses Praat Formant (burg) with 5 expected formants and a 5500 Hz ceiling,
    suitable for both male and female speech at 16 kHz.
    Undefined formant values (0 / NaN) are excluded before computing stats.
    """
    snd     = parselmouth.Sound(str(path))
    formant = call(snd, "To Formant (burg)", 0.0, 5, FORMANT_MAX, 0.025, 50)

    results = []
    for fn in (1, 2, 3):
        vals = []
        for t in formant.ts():
            v = call(formant, "Get value at time", fn, t, "Hertz", "Linear")
            if v and not np.isnan(v) and v > 0:
                vals.append(v)
        if vals:
            results.extend([float(np.mean(vals)), float(np.std(vals))])
        else:
            results.extend([0.0, 0.0])

    return np.array(results, dtype=np.float64)


# ──────────────────────────────────────────────
# Block D — HNR  [66:68]
# ──────────────────────────────────────────────

def _extract_hnr_block(path: str | Path) -> np.ndarray:
    """Return 2 values: HNR mean and std (dB)."""
    snd         = parselmouth.Sound(str(path))
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, F0_FLOOR, 0.1, 1.0)
    hnr_vals    = harmonicity.values.flatten()
    valid       = hnr_vals[hnr_vals != -200.0]
    if len(valid) == 0:
        return np.zeros(2)
    return np.array([float(np.mean(valid)), float(np.std(valid))], dtype=np.float64)


# ──────────────────────────────────────────────
# Block E — Amplitude envelope  [68:72]
# ──────────────────────────────────────────────

def _extract_amplitude_block(y: np.ndarray) -> np.ndarray:
    """
    Return 4 values from the short-time RMS envelope:
    mean, std, skewness (scipy), excess kurtosis (scipy, Fisher definition).
    """
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    return np.array([
        float(np.mean(rms)),
        float(np.std(rms)),
        float(scipy.stats.skew(rms)),
        float(scipy.stats.kurtosis(rms)),   # excess (Fisher), 0 for Gaussian
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Block F — Spectral  [72:78]
# ──────────────────────────────────────────────

def _extract_spectral_block(y: np.ndarray, sr: int) -> np.ndarray:
    """Return 6 values: centroid mean/std, bandwidth mean, rolloff mean, ZCR mean/std."""
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff   = librosa.feature.spectral_rolloff( y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr       = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    return np.array([
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.mean(rolloff)),
        float(np.mean(zcr)),      float(np.std(zcr)),
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Block G — Onset / Attack  [78:82]
# ──────────────────────────────────────────────

def _extract_onset_block(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Return 4 values capturing voice onset / attack characteristics:
      [0] onset_rate       — detected onsets per second
      [1] onset_str_mean   — mean onset-strength envelope (spectral flux proxy)
      [2] onset_str_std    — std of onset-strength envelope
      [3] mean_ioi         — mean inter-onset interval in seconds (0 if < 2 onsets)

    onset_str_mean/std reflect how sharply and consistently the voice
    builds energy (high values = crisp attacks; low = gradual/breathy onset).
    """
    duration = len(y) / sr
    o_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    o_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=HOP_LENGTH)
    o_times  = librosa.frames_to_time(o_frames, sr=sr, hop_length=HOP_LENGTH)

    onset_rate = len(o_times) / max(duration, 1e-6)

    if len(o_times) >= 2:
        ioi      = float(np.mean(np.diff(o_times)))
    else:
        ioi      = 0.0

    return np.array([
        onset_rate,
        float(np.mean(o_env)),
        float(np.std(o_env)),
        ioi,
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Primary public API
# ──────────────────────────────────────────────

def extract_essence(wav_path: str) -> np.ndarray:
    """
    Extract the Voice Essence from a .wav file.

    Returns a fixed-length vector capturing speaker acoustic identity.

    Shape  : (82,), dtype float64
    Layout : see ESSENCE_LAYOUT at the top of this module

    Segment breakdown
    -----------------
    [0:5]   F0 statistics     — pitch height and range (speaker register)
    [5:8]   F0 trajectory     — pitch dynamics, vibrato
    [8:34]  MFCCs × 2         — vocal tract shape (the strongest speaker ID signal)
    [34:60] Delta-MFCCs × 2   — how quickly the vocal tract changes
    [60:66] Formants F1–F3    — vowel space, articulatory posture
    [66:68] HNR               — voice quality (breathy vs. modal vs. pressed)
    [68:72] Amplitude env.    — loudness contour shape
    [72:78] Spectral           — brightness, bandwidth, noise ratio
    [78:82] Onset/attack       — speaking rate, articulation sharpness
    """
    path = Path(wav_path)
    logger.info("extract_essence: %s", path.name)

    y, sr = load_audio(path)

    blocks = [
        _extract_f0_block(path),           # [0:8]   8 dims
        _extract_mfcc_block(y, sr),        # [8:60]  52 dims
        _extract_formant_block(path),      # [60:66]  6 dims
        _extract_hnr_block(path),          # [66:68]  2 dims
        _extract_amplitude_block(y),       # [68:72]  4 dims
        _extract_spectral_block(y, sr),    # [72:78]  6 dims
        _extract_onset_block(y, sr),       # [78:82]  4 dims
    ]

    essence = np.concatenate(blocks).astype(np.float64)
    assert essence.shape == (ESSENCE_DIM,), f"Unexpected shape {essence.shape}"
    return essence


# ──────────────────────────────────────────────
# Legacy API (used by aggregate.py / verify.py)
# ──────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    Thin wrapper around the 82-dim essence vector for backward compatibility
    with aggregate.py and verify.py.
    """
    file: str = ""
    _vec: np.ndarray = field(default_factory=lambda: np.zeros(ESSENCE_DIM))

    # Scalar aliases for verify.py bounds checks
    @property
    def f0_mean(self) -> float:       return float(self._vec[0])
    @property
    def f0_std(self) -> float:        return float(self._vec[1])
    @property
    def f0_min(self) -> float:        return float(self._vec[2])
    @property
    def f0_max(self) -> float:        return float(self._vec[3])
    @property
    def f0_voiced_fraction(self) -> float: return float(self._vec[4])
    @property
    def hnr_mean(self) -> float:      return float(self._vec[66])
    @property
    def rms_mean(self) -> float:      return float(self._vec[68])

    def to_flat_array(self) -> np.ndarray:
        return self._vec.copy()

    def __repr__(self) -> str:
        return (
            f"FeatureVector(file={self.file!r}, dim={ESSENCE_DIM}, "
            f"f0_mean={self.f0_mean:.1f} Hz, hnr={self.hnr_mean:.1f} dB)"
        )


def extract(path: str | Path, sr: int = SAMPLE_RATE) -> FeatureVector:
    """
    Extract features from a single audio file.
    Returns a FeatureVector wrapping the 82-dim essence vector.
    """
    path = Path(path)
    vec  = extract_essence(str(path))
    return FeatureVector(file=str(path), _vec=vec)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract Voice Essence from an audio file.")
    parser.add_argument("audio", help="Path to .wav file")
    args = parser.parse_args()

    vec = extract_essence(args.audio)

    print(f"\nEssence vector — shape {vec.shape}, dtype {vec.dtype}\n")
    for name, (sl, desc) in ESSENCE_LAYOUT.items():
        vals = vec[sl]
        print(f"  {name:<18} [{sl.start}:{sl.stop}]  {desc}")
        print(f"               → {np.round(vals, 3)}\n")

    # Emit JSON for subprocess comparison in test_determinism
    output = {
        "file":         args.audio,
        "essence_dim":  int(ESSENCE_DIM),
        "f0_mean":      float(vec[0]),
        "f0_std":       float(vec[1]),
        "rms_mean":     float(vec[68]),
        "hnr_mean":     float(vec[66]),
        "mfcc_mean":    vec[8:21].tolist(),
        "vector":       vec.tolist(),
    }
    import sys
    # Only print JSON if stdout is being captured (pipe) to avoid polluting terminal
    if not sys.stdout.isatty():
        print(json.dumps(output))
