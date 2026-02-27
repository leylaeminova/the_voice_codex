"""
extract.py — acoustic feature extraction from audio files.

Extracts a deterministic feature vector per utterance, including:
  - Fundamental frequency (F0) statistics via Parselmouth/Praat
  - MFCCs, delta-MFCCs via librosa
  - Spectral features: centroid, bandwidth, rolloff, ZCR
  - RMS energy and harmonic-to-noise ratio (HNR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SAMPLE_RATE: int = 16_000          # resample target (Hz)
HOP_LENGTH: int = 256              # librosa hop length
N_FFT: int = 1024                  # FFT window size
N_MFCC: int = 13                   # number of MFCC coefficients
F0_FLOOR: float = 75.0             # Praat F0 floor (Hz)
F0_CEILING: float = 600.0          # Praat F0 ceiling (Hz)


# ──────────────────────────────────────────────
# Data structure
# ──────────────────────────────────────────────

@dataclass
class FeatureVector:
    """Flat feature vector for a single audio utterance."""

    file: str = ""

    # F0
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_voiced_fraction: float = 0.0

    # HNR
    hnr_mean: float = 0.0

    # Energy
    rms_mean: float = 0.0
    rms_std: float = 0.0

    # Spectral
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zcr_mean: float = 0.0

    # MFCCs (mean + std per coefficient → 2 × N_MFCC values)
    mfcc_mean: list[float] = field(default_factory=list)
    mfcc_std: list[float] = field(default_factory=list)

    # Delta-MFCCs
    delta_mfcc_mean: list[float] = field(default_factory=list)
    delta_mfcc_std: list[float] = field(default_factory=list)

    def to_flat_array(self) -> np.ndarray:
        """Return a 1-D float64 array of all scalar features."""
        scalars = [
            self.f0_mean, self.f0_std, self.f0_min, self.f0_max,
            self.f0_voiced_fraction, self.hnr_mean,
            self.rms_mean, self.rms_std,
            self.spectral_centroid_mean, self.spectral_centroid_std,
            self.spectral_bandwidth_mean, self.spectral_rolloff_mean,
            self.zcr_mean,
        ]
        return np.array(
            scalars
            + list(self.mfcc_mean)
            + list(self.mfcc_std)
            + list(self.delta_mfcc_mean)
            + list(self.delta_mfcc_std),
            dtype=np.float64,
        )


# ──────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────

def load_audio(path: Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load and resample audio to a fixed sample rate."""
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, sr_out


def extract_f0_hnr(
    path: Path,
    floor: float = F0_FLOOR,
    ceiling: float = F0_CEILING,
) -> tuple[dict[str, float], float]:
    """
    Use Parselmouth/Praat to extract F0 statistics and HNR.
    Returns (f0_stats dict, hnr_mean).
    """
    snd = parselmouth.Sound(str(path))

    # F0 via autocorrelation
    pitch = call(snd, "To Pitch", 0.0, floor, ceiling)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    f0_stats = {
        "f0_mean": float(np.mean(voiced)) if len(voiced) else 0.0,
        "f0_std": float(np.std(voiced)) if len(voiced) else 0.0,
        "f0_min": float(np.min(voiced)) if len(voiced) else 0.0,
        "f0_max": float(np.max(voiced)) if len(voiced) else 0.0,
        "f0_voiced_fraction": len(voiced) / max(len(f0_values), 1),
    }

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
    hnr_values = harmonicity.values.flatten()
    valid_hnr = hnr_values[hnr_values != -200.0]
    hnr_mean = float(np.mean(valid_hnr)) if len(valid_hnr) else 0.0

    return f0_stats, hnr_mean


def extract_librosa_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> dict[str, object]:
    """Extract MFCCs, spectral, and energy features via librosa."""
    # RMS
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta_mfcc = librosa.feature.delta(mfcc)

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "mfcc_std": mfcc.std(axis=1).tolist(),
        "delta_mfcc_mean": delta_mfcc.mean(axis=1).tolist(),
        "delta_mfcc_std": delta_mfcc.std(axis=1).tolist(),
    }


def extract(path: str | Path, sr: int = SAMPLE_RATE) -> FeatureVector:
    """
    Full extraction pipeline for a single audio file.

    Parameters
    ----------
    path : path-like
        Audio file (wav, mp3, flac, …).
    sr : int
        Target sample rate.

    Returns
    -------
    FeatureVector
    """
    path = Path(path)
    logger.info("Extracting: %s", path.name)

    y, sr_out = load_audio(path, sr=sr)
    f0_stats, hnr_mean = extract_f0_hnr(path)
    lib_feats = extract_librosa_features(y, sr_out)

    fv = FeatureVector(
        file=str(path),
        f0_mean=f0_stats["f0_mean"],
        f0_std=f0_stats["f0_std"],
        f0_min=f0_stats["f0_min"],
        f0_max=f0_stats["f0_max"],
        f0_voiced_fraction=f0_stats["f0_voiced_fraction"],
        hnr_mean=hnr_mean,
        **lib_feats,
    )
    return fv


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Extract acoustic features from an audio file.")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate (Hz)")
    args = parser.parse_args()

    fv = extract(args.audio, sr=args.sr)
    print(json.dumps(fv.__dict__, indent=2))
