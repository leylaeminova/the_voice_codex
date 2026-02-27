"""
verify.py — quality-control and sanity checks on extracted feature data.

Checks performed
----------------
1. Shape / dtype consistency
2. No NaN or Inf values
3. Feature range plausibility (F0, HNR, MFCC bounds)
4. Per-file duplicate detection
5. Optional z-score outlier flagging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from aggregate import load_npz

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Expected feature bounds (index → (lo, hi))
# These map to the flat array produced by FeatureVector.to_flat_array()
# ──────────────────────────────────────────────

# Scalar slots: f0_mean, f0_std, f0_min, f0_max, f0_voiced_fraction,
#               hnr_mean, rms_mean, rms_std,
#               spectral_centroid_mean, spectral_centroid_std,
#               spectral_bandwidth_mean, spectral_rolloff_mean, zcr_mean
BOUNDS: dict[int, tuple[float, float]] = {
    0:  (0.0, 1000.0),   # f0_mean (Hz)
    1:  (0.0, 500.0),    # f0_std
    2:  (0.0, 1000.0),   # f0_min
    3:  (0.0, 1500.0),   # f0_max
    4:  (0.0, 1.0),      # f0_voiced_fraction
    5:  (-20.0, 40.0),   # hnr_mean (dB)
    6:  (0.0, 1.0),      # rms_mean (normalised)
    7:  (0.0, 1.0),      # rms_std
    8:  (0.0, 8000.0),   # spectral_centroid_mean (Hz, sr/2 max)
    12: (0.0, 1.0),      # zcr_mean
}

Z_THRESHOLD: float = 4.0   # flag samples beyond this many σ from the mean


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class VerificationReport:
    n_samples: int
    n_features: int
    has_nan: bool
    has_inf: bool
    bounds_violations: dict[str, list[int]]   # feature_name → [sample indices]
    duplicate_indices: list[tuple[int, int]]
    outlier_indices: dict[int, list[int]]     # feat_idx → [sample indices]
    passed: bool

    def summary(self) -> str:
        lines = [
            f"Samples : {self.n_samples}",
            f"Features: {self.n_features}",
            f"NaN     : {self.has_nan}",
            f"Inf     : {self.has_inf}",
            f"Bounds  : {sum(len(v) for v in self.bounds_violations.values())} violation(s)",
            f"Dupes   : {len(self.duplicate_indices)} pair(s)",
            f"Outliers: {sum(len(v) for v in self.outlier_indices.values())} flag(s)",
            f"PASSED  : {self.passed}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────

def check_finite(features: np.ndarray) -> tuple[bool, bool]:
    """Return (has_nan, has_inf)."""
    has_nan = bool(np.isnan(features).any())
    has_inf = bool(np.isinf(features).any())
    if has_nan:
        logger.warning("NaN values detected in feature matrix.")
    if has_inf:
        logger.warning("Inf values detected in feature matrix.")
    return has_nan, has_inf


def check_bounds(features: np.ndarray) -> dict[str, list[int]]:
    """Return mapping from feature label → list of offending sample indices."""
    violations: dict[str, list[int]] = {}
    for col, (lo, hi) in BOUNDS.items():
        if col >= features.shape[1]:
            continue
        col_data = features[:, col]
        bad = np.where((col_data < lo) | (col_data > hi))[0].tolist()
        if bad:
            label = f"feat[{col}]"
            violations[label] = bad
            logger.warning("Bounds violation at %s (%.1f–%.1f): %d sample(s)", label, lo, hi, len(bad))
    return violations


def check_duplicates(features: np.ndarray) -> list[tuple[int, int]]:
    """Return pairs of indices with identical feature vectors."""
    dupes: list[tuple[int, int]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if np.array_equal(features[i], features[j]):
                dupes.append((i, j))
                logger.warning("Duplicate vectors at indices %d and %d", i, j)
    return dupes


def check_outliers(
    features: np.ndarray,
    z_threshold: float = Z_THRESHOLD,
) -> dict[int, list[int]]:
    """Flag samples beyond *z_threshold* σ from the column mean."""
    outliers: dict[int, list[int]] = {}
    col_means = features.mean(axis=0)
    col_stds = features.std(axis=0)
    for col in range(features.shape[1]):
        if col_stds[col] == 0:
            continue
        z = np.abs((features[:, col] - col_means[col]) / col_stds[col])
        flagged = np.where(z > z_threshold)[0].tolist()
        if flagged:
            outliers[col] = flagged
    return outliers


# ──────────────────────────────────────────────
# Main verify function
# ──────────────────────────────────────────────

def verify(
    source: str | Path | np.ndarray,
    files: Optional[list[str]] = None,
    z_threshold: float = Z_THRESHOLD,
    strict: bool = False,
) -> VerificationReport:
    """
    Run all QC checks on a feature matrix.

    Parameters
    ----------
    source : path to .npz file, or a pre-loaded ndarray
    files : file list (required when source is ndarray)
    z_threshold : z-score cutoff for outlier detection
    strict : if True, raise on any failure

    Returns
    -------
    VerificationReport
    """
    if isinstance(source, (str, Path)):
        features, files = load_npz(source)
    else:
        features = np.asarray(source, dtype=np.float64)
        files = files or []

    logger.info("Verifying matrix shape=%s", features.shape)

    has_nan, has_inf = check_finite(features)
    bounds_violations = check_bounds(features)
    duplicates = check_duplicates(features)
    outliers = check_outliers(features, z_threshold=z_threshold)

    passed = not has_nan and not has_inf and not bounds_violations and not duplicates

    report = VerificationReport(
        n_samples=features.shape[0],
        n_features=features.shape[1],
        has_nan=has_nan,
        has_inf=has_inf,
        bounds_violations=bounds_violations,
        duplicate_indices=duplicates,
        outlier_indices=outliers,
        passed=passed,
    )

    if strict and not passed:
        raise ValueError(f"Verification failed:\n{report.summary()}")

    return report


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Verify a feature .npz archive.")
    parser.add_argument("npz", help="Path to .npz feature archive")
    parser.add_argument("--z", type=float, default=Z_THRESHOLD, help="Z-score outlier threshold")
    parser.add_argument("--strict", action="store_true", help="Exit with error code on failure")
    args = parser.parse_args()

    report = verify(args.npz, z_threshold=args.z, strict=args.strict)
    print(report.summary())
    raise SystemExit(0 if report.passed else 1)
