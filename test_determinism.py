"""
test_determinism.py — assert that feature extraction is bit-exact across runs.

Strategy
--------
1. Synthesize a short deterministic test signal (pure tones + white noise with
   fixed seed) and save it as a temporary WAV.
2. Run extract() twice on the same file.
3. Assert the two FeatureVector arrays are identical (np.array_equal).
4. Also verify that results are stable across re-imports by running extract()
   in a subprocess and comparing against the in-process result.

Run with pytest or directly:
    pytest test_determinism.py -v
    python test_determinism.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav

from extract import extract, FeatureVector, SAMPLE_RATE


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

DURATION_S = 2.0    # seconds
SEED = 42


def _make_test_wav(path: Path, sr: int = SAMPLE_RATE, seed: int = SEED) -> None:
    """Write a deterministic test WAV: 200 Hz sine + fixed-seed white noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION_S, int(sr * DURATION_S), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 200 * t)   # 200 Hz fundamental
    signal += 0.05 * rng.standard_normal(len(t))   # low-level noise
    signal_int16 = (signal * 32767).astype(np.int16)
    wav.write(str(path), sr, signal_int16)


@pytest.fixture(scope="module")
def test_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("audio") / "test_tone.wav"
    _make_test_wav(p)
    return p


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

class TestInProcessDeterminism:
    """Two calls to extract() in the same process must return identical vectors."""

    def test_vectors_are_identical(self, test_wav: Path) -> None:
        fv1: FeatureVector = extract(test_wav)
        fv2: FeatureVector = extract(test_wav)

        arr1 = fv1.to_flat_array()
        arr2 = fv2.to_flat_array()

        assert arr1.shape == arr2.shape, "Feature vector shapes differ between runs."
        assert np.array_equal(arr1, arr2), (
            f"Feature vectors differ.\nMax abs diff: {np.max(np.abs(arr1 - arr2)):.2e}"
        )

    def test_vector_is_finite(self, test_wav: Path) -> None:
        fv = extract(test_wav)
        arr = fv.to_flat_array()
        assert np.all(np.isfinite(arr)), "Feature vector contains NaN or Inf."

    def test_vector_dimension(self, test_wav: Path) -> None:
        """Dimension must be fixed: 13 scalars + 4 × N_MFCC MFCC values."""
        from extract import N_MFCC
        expected_dim = 13 + 4 * N_MFCC
        fv = extract(test_wav)
        arr = fv.to_flat_array()
        assert arr.shape[0] == expected_dim, (
            f"Expected dim {expected_dim}, got {arr.shape[0]}."
        )

    def test_f0_plausible(self, test_wav: Path) -> None:
        """F0 mean should be near 200 Hz for the test tone."""
        fv = extract(test_wav)
        # Allow generous tolerance — Praat may read slight freq deviations
        assert 150.0 < fv.f0_mean < 300.0, (
            f"F0 mean {fv.f0_mean:.1f} Hz is outside expected range for 200 Hz tone."
        )


class TestCrossProcessDeterminism:
    """
    Run extract() in a subprocess via the CLI and compare with in-process result.
    This catches hidden global state and import-order side effects.
    """

    def test_subprocess_matches_inprocess(self, test_wav: Path) -> None:
        result = subprocess.run(
            [sys.executable, "extract.py", str(test_wav)],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        subprocess_dict = json.loads(result.stdout)

        # Re-extract in-process
        fv = extract(test_wav)
        inprocess_dict = fv.__dict__

        for key in ("f0_mean", "f0_std", "rms_mean", "hnr_mean"):
            sp_val = subprocess_dict[key]
            ip_val = inprocess_dict[key]
            assert abs(sp_val - ip_val) < 1e-9, (
                f"Mismatch for {key}: subprocess={sp_val}, in-process={ip_val}"
            )

        sp_mfcc = np.array(subprocess_dict["mfcc_mean"])
        ip_mfcc = np.array(inprocess_dict["mfcc_mean"])
        assert np.allclose(sp_mfcc, ip_mfcc, atol=1e-9), (
            f"MFCC mean mismatch.\nMax diff: {np.max(np.abs(sp_mfcc - ip_mfcc)):.2e}"
        )


class TestAggregateConsistency:
    """aggregate() on a single file must match direct extract() output."""

    def test_aggregate_matches_extract(self, test_wav: Path, tmp_path: Path) -> None:
        from aggregate import aggregate, load_npz

        out_npz = tmp_path / "features.npz"
        matrix, files = aggregate(
            test_wav.parent,
            output=out_npz,
            skip_errors=False,
        )

        fv = extract(test_wav)
        expected = fv.to_flat_array()

        # Find the row corresponding to our test file
        idx = next(
            (i for i, f in enumerate(files) if Path(f).name == test_wav.name),
            None,
        )
        assert idx is not None, "Test file not found in aggregate output."
        assert np.array_equal(matrix[idx], expected), (
            "Aggregate row does not match direct extract() output."
        )


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test_tone.wav"
        _make_test_wav(wav_path)

        print("Running determinism tests directly …")

        fv1 = extract(wav_path)
        fv2 = extract(wav_path)
        arr1 = fv1.to_flat_array()
        arr2 = fv2.to_flat_array()

        if np.array_equal(arr1, arr2):
            print("PASS  in-process determinism")
        else:
            print(f"FAIL  max diff = {np.max(np.abs(arr1 - arr2)):.2e}")

        if np.all(np.isfinite(arr1)):
            print("PASS  all features finite")
        else:
            print("FAIL  NaN or Inf in feature vector")

        print(f"INFO  feature dim = {arr1.shape[0]}")
        print(f"INFO  F0 mean = {fv1.f0_mean:.2f} Hz")
