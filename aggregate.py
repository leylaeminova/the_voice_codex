"""
aggregate.py — batch processing and aggregation of feature vectors.

Takes a directory (or manifest CSV) of audio files, runs extract.py on each,
and writes results to a structured NumPy archive (.npz) or CSV.
"""

from __future__ import annotations

import csv
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from extract import FeatureVector, extract, SAMPLE_RATE

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ──────────────────────────────────────────────
# Manifest helpers
# ──────────────────────────────────────────────

def collect_audio_files(root: Path, recursive: bool = True) -> list[Path]:
    """Return a sorted list of audio files under *root*."""
    glob = root.rglob("*") if recursive else root.glob("*")
    files = sorted(p for p in glob if p.suffix.lower() in AUDIO_EXTENSIONS)
    logger.info("Found %d audio file(s) under %s", len(files), root)
    return files


def load_manifest(csv_path: Path) -> list[Path]:
    """
    Load a two-column CSV manifest: ``file,label`` (header optional).
    Returns only the file paths.
    """
    paths: list[Path] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower() in ("file", "path", "audio"):
                continue
            paths.append(Path(row[0].strip()))
    return paths


# ──────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────

def aggregate(
    source: str | Path,
    output: str | Path = "features.npz",
    sr: int = SAMPLE_RATE,
    recursive: bool = True,
    skip_errors: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract features for all audio files found under *source* (directory)
    or listed in *source* (CSV manifest), then save to *output*.

    Parameters
    ----------
    source : path-like
        Directory of audio files **or** a CSV manifest.
    output : path-like
        Destination .npz file (also writes a sidecar .csv).
    sr : int
        Target sample rate for librosa loading.
    recursive : bool
        Whether to recurse into subdirectories (ignored for CSV manifests).
    skip_errors : bool
        If True, log extraction errors and continue; else re-raise.

    Returns
    -------
    features : np.ndarray, shape (N, D)
    file_list : list[str]
    """
    source = Path(source)
    output = Path(output)

    if source.is_file() and source.suffix.lower() == ".csv":
        paths = load_manifest(source)
    elif source.is_dir():
        paths = collect_audio_files(source, recursive=recursive)
    else:
        raise ValueError(f"source must be a directory or a .csv manifest, got: {source}")

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    file_list: list[str] = []

    for p in paths:
        try:
            fv: FeatureVector = extract(p, sr=sr)
            vec = fv.to_flat_array()
            vectors.append(vec)
            file_list.append(str(p))
            rows.append(asdict(fv))
            logger.info("  OK  %s  (dim=%d)", p.name, vec.shape[0])
        except Exception as exc:  # noqa: BLE001
            if skip_errors:
                logger.warning("SKIP %s — %s", p.name, exc)
                logger.debug(traceback.format_exc())
            else:
                raise

    if not vectors:
        raise RuntimeError("No features extracted — check source path and audio files.")

    matrix = np.vstack(vectors)  # (N, D)

    # Save .npz
    np.savez_compressed(output, features=matrix, files=np.array(file_list))
    logger.info("Saved %s  shape=%s", output, matrix.shape)

    # Save sidecar CSV
    csv_out = output.with_suffix(".csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %s", csv_out)

    return matrix, file_list


def load_npz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a previously saved .npz feature archive."""
    data = np.load(str(path), allow_pickle=False)
    return data["features"], data["files"].tolist()


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch-extract and aggregate acoustic features.")
    parser.add_argument("source", help="Audio directory or CSV manifest")
    parser.add_argument("--output", default="features.npz", help="Output .npz path")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate (Hz)")
    parser.add_argument("--no-recurse", action="store_true", help="Disable recursive directory walk")
    parser.add_argument("--strict", action="store_true", help="Abort on first extraction error")
    args = parser.parse_args()

    matrix, files = aggregate(
        args.source,
        output=args.output,
        sr=args.sr,
        recursive=not args.no_recurse,
        skip_errors=not args.strict,
    )
    print(f"Aggregated {len(files)} file(s) → matrix shape {matrix.shape}")
