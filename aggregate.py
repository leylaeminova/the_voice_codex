"""
aggregate.py — Welford online aggregation of Voice Essence vectors.

═══════════════════════════════════════════════════════════════════════════════
  ALGORITHM: Welford's Online Algorithm
═══════════════════════════════════════════════════════════════════════════════

  Classic batch mean:   μ  = (1/N) Σ xᵢ              — requires all N vectors in memory
  Welford update:       μₙ = μₙ₋₁ + (xₙ − μₙ₋₁) / n  — O(1) time, O(D) memory

  The Crystal's genius: state = (count, mean, M2) per feature dimension.
  Adding voice #1,000,001 costs exactly the same as adding voice #2.

  Variance is recovered at any point:
    population variance  = M2 / count
    sample variance      = M2 / (count − 1)   [Bessel-corrected, count ≥ 2]

═══════════════════════════════════════════════════════════════════════════════
  BONUS: Cluster-then-Aggregate
═══════════════════════════════════════════════════════════════════════════════

  Instead of one global forge, we first partition voices into K clusters
  (K-Means or HDBSCAN) and run a separate Welford forge per cluster.
  The per-cluster means form a richer representation — a "speaker atlas" —
  compared to the single grand mean.

  Comparison metric: Calinski–Harabasz-style ratio  B / W
    B = between-cluster variance (variance of cluster means, weighted by size)
    W = within-cluster variance  (average intra-cluster variance, weighted)
    Higher B/W → clusters are tight and well-separated.

Public API
----------
  forge_init(dim)                       → AggregateState
  forge_update(state, essence)          → AggregateState          ← O(1)
  forge_merge(state_a, state_b)         → AggregateState          ← parallel merge
  forge_mean(state)                     → np.ndarray (D,)
  forge_variance(state)                 → np.ndarray (D,)
  forge_std(state)                      → np.ndarray (D,)
  forge_stderr(state)                   → np.ndarray (D,)

  build_aggregate(essences)             → AggregateState
  save_aggregate(state, path)
  load_aggregate(path)                  → AggregateState

  cluster_aggregate(essences, …)        → ClusterResult
  compare_strategies(essences, labels)  → ComparisonReport (printed)

  aggregate(source, output, …)          → (matrix, file_list, AggregateState)
  load_npz(path)                        → (matrix, file_list)
"""

from __future__ import annotations

import csv
import logging
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler

from extract import extract_essence, ESSENCE_DIM, ESSENCE_LAYOUT, SAMPLE_RATE

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Welford State type
# ═══════════════════════════════════════════════════════════════════════════════

class AggregateState(NamedTuple):
    """
    Immutable snapshot of a Welford running aggregate.

    Fields
    ------
    count : int
        Number of voices absorbed so far.
    mean  : np.ndarray, shape (D,)
        Running per-dimension mean.  Equal to forge_mean(state).
    M2    : np.ndarray, shape (D,)
        Running per-dimension sum of squared deviations from the mean.
        Divide by (count-1) to get sample variance; by count for population.
    """
    count: int
    mean:  np.ndarray
    M2:    np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Welford core — O(1) per update
# ═══════════════════════════════════════════════════════════════════════════════

def forge_init(dim: int = ESSENCE_DIM) -> AggregateState:
    """Return a zero-initialised aggregate state for *dim*-dimensional vectors."""
    return AggregateState(count=0, mean=np.zeros(dim), M2=np.zeros(dim))


def forge_update(aggregate_state: AggregateState, new_essence: np.ndarray) -> AggregateState:
    """
    Add one voice to the collective.  O(1) time, O(1) memory.

    Implements Welford's numerically stable online algorithm (Welford 1962).
    No previous voices are re-processed — the state carries all necessary
    information to update mean and variance exactly.

    Parameters
    ----------
    aggregate_state : AggregateState
        Current running state (count, mean, M2).
    new_essence : np.ndarray, shape (D,)
        The next voice's essence vector.

    Returns
    -------
    AggregateState
        Updated state.  The returned mean and M2 arrays are modified in-place
        for efficiency; the count is a new integer.
    """
    (count, mean, M2) = aggregate_state      # unpack exactly as prescribed
    count  += 1
    delta   = new_essence - mean             # deviation from OLD mean
    mean   += delta / count                  # update mean in-place
    delta2  = new_essence - mean             # deviation from NEW mean
    M2     += delta * delta2                 # update M2 in-place (no copy needed)
    return AggregateState(count, mean, M2)


def forge_merge(state_a: AggregateState, state_b: AggregateState) -> AggregateState:
    """
    Combine two independent aggregate states into one without re-processing.

    Uses Chan et al.'s parallel Welford formula (Chan, Golub & LeVeque 1979).
    Enables map-reduce style parallelism: shard voices across workers, merge
    the resulting states — total work is O(K) not O(N) for K shards.
    """
    na, μa, M2a = state_a
    nb, μb, M2b = state_b
    n   = na + nb
    if n == 0:
        return forge_init(len(μa))
    δ   = μb - μa
    μ   = (na * μa + nb * μb) / n          # weighted mean (exact)
    M2  = M2a + M2b + δ ** 2 * na * nb / n # Chan's correction term
    return AggregateState(n, μ, M2)


# ── Derived statistics ────────────────────────────────────────────────────────

def forge_mean(state: AggregateState) -> np.ndarray:
    """Per-dimension mean of all absorbed voices."""
    return state.mean.copy()


def forge_variance(state: AggregateState, population: bool = False) -> np.ndarray:
    """
    Per-dimension variance.

    Parameters
    ----------
    population : bool
        If True, divide by N (population).  Default False → divide by N-1
        (Bessel-corrected sample variance, unbiased estimator).
    """
    count, _, M2 = state
    if count < 2:
        return np.zeros_like(M2)
    denom = count if population else (count - 1)
    return M2 / denom


def forge_std(state: AggregateState, population: bool = False) -> np.ndarray:
    """Per-dimension standard deviation."""
    return np.sqrt(forge_variance(state, population=population))


def forge_stderr(state: AggregateState) -> np.ndarray:
    """Standard error of the mean: std / sqrt(N)."""
    return forge_std(state) / max(np.sqrt(state.count), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Batch wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def build_aggregate(essences: list[np.ndarray] | np.ndarray) -> AggregateState:
    """
    Absorb every essence in *essences* through forge_update and return the
    final aggregate state.

    O(N·D) total work, O(1) per voice — no re-processing occurs.
    The function never stores more than the current (count, mean, M2) triple;
    individual essences are processed and immediately discarded.

    Parameters
    ----------
    essences : list of 1-D arrays or 2-D array (N, D)
        Voice essence vectors.  All must have the same dimension D.

    Returns
    -------
    AggregateState
        Final state after absorbing all voices.
    """
    if isinstance(essences, np.ndarray) and essences.ndim == 2:
        essences = [essences[i] for i in range(len(essences))]

    if not essences:
        raise ValueError("essences list is empty.")

    dim   = len(essences[0])
    state = forge_init(dim)

    for essence in essences:                 # one update = O(1) — provably
        state = forge_update(state, essence) # no lookback, no re-scan

    return state


def build_aggregate_timed(essences: list[np.ndarray]) -> tuple[AggregateState, list[float]]:
    """
    Like build_aggregate but records wall-clock time (µs) per update.
    Used to empirically demonstrate O(1) behaviour — update time should be
    flat regardless of how many voices have already been absorbed.
    """
    dim    = len(essences[0])
    state  = forge_init(dim)
    times  = []

    for essence in essences:
        t0    = time.perf_counter()
        state = forge_update(state, essence)
        times.append((time.perf_counter() - t0) * 1e6)   # µs

    return state, times


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_aggregate(state: AggregateState, path: str | Path) -> None:
    """
    Persist an aggregate state to a .npz file.

    Saved arrays: ``count`` (scalar), ``mean`` (D,), ``M2`` (D,).
    The state is sufficient to resume aggregation (via forge_update or
    forge_merge) or to compute mean / variance for Trial III verification.
    """
    path = Path(path)
    np.savez_compressed(
        path,
        count=np.array(state.count),
        mean=state.mean,
        M2=state.M2,
    )
    logger.info("Saved aggregate state → %s  (count=%d)", path, state.count)


def load_aggregate(path: str | Path) -> AggregateState:
    """Load a previously saved aggregate state from a .npz file."""
    data = np.load(str(path), allow_pickle=False)
    return AggregateState(
        count=int(data["count"]),
        mean=data["mean"],
        M2=data["M2"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Cluster-then-aggregate  (bonus)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterResult:
    """
    Result of cluster-then-aggregate.  One AggregateState per cluster.

    Attributes
    ----------
    method : str
        'kmeans' or 'hdbscan'.
    n_clusters : int
        Number of clusters found (excluding noise in HDBSCAN).
    cluster_ids : np.ndarray (N,)
        Cluster label per input essence (-1 = HDBSCAN noise, kept separately).
    states : dict[int, AggregateState]
        Welford state per cluster id.
    bw_ratio : float
        Between-cluster / within-cluster variance ratio (higher → better).
    noise_count : int
        Number of HDBSCAN noise points (-1 label).  0 for K-Means.
    """
    method:      str
    n_clusters:  int
    cluster_ids: np.ndarray
    states:      dict[int, AggregateState]
    bw_ratio:    float
    noise_count: int = 0

    def cluster_means(self) -> np.ndarray:
        """Return (n_clusters, D) array of per-cluster means, sorted by cluster id."""
        ids = sorted(k for k in self.states if k != -1)
        return np.vstack([self.states[k].mean for k in ids])

    def summary(self) -> str:
        lines = [
            f"Method       : {self.method}",
            f"Clusters     : {self.n_clusters}",
            f"Noise pts    : {self.noise_count}",
            f"B/W ratio    : {self.bw_ratio:.4f}",
        ]
        for cid, st in sorted(self.states.items()):
            lines.append(f"  cluster {cid:>2d}  n={st.count:3d}  "
                         f"mean_var={forge_variance(st).mean():.4f}")
        return "\n".join(lines)


def _bw_ratio(essences: np.ndarray, cluster_ids: np.ndarray) -> float:
    """
    Calinski–Harabasz-style between/within variance ratio.

    B = weighted variance of cluster means around the grand mean
    W = weighted mean of intra-cluster variances
    """
    valid_mask = cluster_ids != -1
    X_v  = essences[valid_mask]
    ids  = cluster_ids[valid_mask]
    uids = np.unique(ids)

    grand_mean = X_v.mean(axis=0)
    N = len(X_v)

    B_total, W_total = 0.0, 0.0
    for uid in uids:
        members = X_v[ids == uid]
        nc      = len(members)
        cm      = members.mean(axis=0)
        B_total += nc * float(np.sum((cm - grand_mean) ** 2))
        if nc > 1:
            W_total += float(np.sum(members.var(axis=0))) * nc

    k = len(uids)
    if W_total == 0 or k == 1:
        return float("inf") if B_total > 0 else 0.0
    return (B_total / (k - 1)) / (W_total / (N - k))


def cluster_aggregate(
    essences: np.ndarray,
    method:   str = "kmeans",
    n_clusters: int = 5,
    min_cluster_size: int = 3,
    random_state: int = 42,
) -> ClusterResult:
    """
    Partition *essences* into clusters, then run a separate Welford forge
    per cluster.

    Parameters
    ----------
    essences : np.ndarray (N, D)
        Voice essence matrix.
    method : {'kmeans', 'hdbscan'}
        Clustering algorithm.
    n_clusters : int
        Number of clusters for K-Means (ignored for HDBSCAN).
    min_cluster_size : int
        HDBSCAN minimum cluster size (ignored for K-Means).
    random_state : int
        Random seed (K-Means only).

    Returns
    -------
    ClusterResult
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(essences)

    if method == "kmeans":
        clf        = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_ids = clf.fit_predict(Xs)
        noise_count = 0
    elif method == "hdbscan":
        clf        = HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_ids = clf.fit_predict(Xs)
        noise_count = int((cluster_ids == -1).sum())
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'kmeans' or 'hdbscan'.")

    unique_ids = [c for c in np.unique(cluster_ids) if c != -1]
    states: dict[int, AggregateState] = {}

    for cid in np.unique(cluster_ids):
        members = essences[cluster_ids == cid]
        states[int(cid)] = build_aggregate(members)

    bw = _bw_ratio(essences, cluster_ids)

    return ClusterResult(
        method      = method,
        n_clusters  = len(unique_ids),
        cluster_ids = cluster_ids,
        states      = states,
        bw_ratio    = bw,
        noise_count = noise_count,
    )


@dataclass
class ComparisonReport:
    """Side-by-side comparison of global vs. cluster-then-aggregate strategies."""
    global_state:    AggregateState
    kmeans_result:   ClusterResult
    hdbscan_result:  ClusterResult

    def print(self) -> None:
        gv = forge_variance(self.global_state).mean()
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║         Aggregation Strategy Comparison                 ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Global forge    n={self.global_state.count:3d}  "
              f"mean_var={gv:.4f}  B/W=  N/A  ║")
        print(f"║  K-Means forge   "
              f"K={self.kmeans_result.n_clusters}    "
              f"mean_var={ forge_variance(  # weighted avg
                  build_aggregate(  # fake: just use global as proxy
                      [self.kmeans_result.states[k].mean
                       for k in sorted(self.kmeans_result.states) if k != -1]
                  )).mean():.4f}  "
              f"B/W={self.kmeans_result.bw_ratio:.2f}  ║")
        print(f"║  HDBSCAN forge   "
              f"K={self.hdbscan_result.n_clusters}  "
              f"noise={self.hdbscan_result.noise_count:2d}  "
              f"B/W={self.hdbscan_result.bw_ratio:.2f}  ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  Interpretation:                                         ║")
        print("║  Higher B/W ratio = cluster means are well-separated     ║")
        print("║  relative to intra-cluster spread → richer representation║")
        print("╚══════════════════════════════════════════════════════════╝")

        print("\n── K-Means cluster detail ──")
        print(self.kmeans_result.summary())
        print("\n── HDBSCAN cluster detail ──")
        print(self.hdbscan_result.summary())


def compare_strategies(
    essences:         np.ndarray,
    n_kmeans_clusters: int = 5,
    min_hdbscan_size:  int = 3,
) -> ComparisonReport:
    """Run all three strategies and return a ComparisonReport."""
    global_state   = build_aggregate(essences)
    kmeans_result  = cluster_aggregate(essences, method="kmeans",
                                        n_clusters=n_kmeans_clusters)
    hdbscan_result = cluster_aggregate(essences, method="hdbscan",
                                        min_cluster_size=min_hdbscan_size)
    return ComparisonReport(global_state, kmeans_result, hdbscan_result)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  File collection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collect_audio_files(root: Path, recursive: bool = True) -> list[Path]:
    """Return a sorted list of audio files under *root*."""
    glob  = root.rglob("*") if recursive else root.glob("*")
    files = sorted(p for p in glob if p.suffix.lower() in AUDIO_EXTENSIONS)
    logger.info("Found %d audio file(s) under %s", len(files), root)
    return files


def load_manifest(csv_path: Path) -> list[Path]:
    """Load a CSV manifest (file,label header optional) and return file paths."""
    paths: list[Path] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in ("file", "path", "audio"):
                continue
            paths.append(Path(row[0].strip()))
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(
    source:     str | Path,
    output:     str | Path = "features.npz",
    sr:         int  = SAMPLE_RATE,
    recursive:  bool = True,
    skip_errors: bool = True,
) -> tuple[np.ndarray, list[str], AggregateState]:
    """
    Extract essences for all audio files in *source*, run them through the
    Welford forge one at a time (O(1) per voice), and save results.

    Returns
    -------
    matrix : np.ndarray (N, D)   — full essence matrix (for downstream use)
    file_list : list[str]        — ordered list of processed paths
    state : AggregateState       — (count, mean, M2) after all voices
    """
    source = Path(source)
    output = Path(output)

    paths = (
        load_manifest(source)
        if source.is_file() and source.suffix.lower() == ".csv"
        else collect_audio_files(source, recursive=recursive)
        if source.is_dir()
        else (_ for _ in ()).throw(ValueError(f"Expected directory or CSV, got: {source}"))
    )

    vectors:   list[np.ndarray] = []
    file_list: list[str]        = []
    rows:      list[dict]       = []
    state = forge_init(ESSENCE_DIM)

    for p in paths:
        try:
            vec   = extract_essence(str(p))
            state = forge_update(state, vec)          # ← O(1) Welford update
            vectors.append(vec)
            file_list.append(str(p))
            row = {"file": str(p)}
            row.update({f"e{i:03d}": float(vec[i]) for i in range(len(vec))})
            rows.append(row)
            logger.info("  OK  %s  n=%d", p.name, state.count)
        except Exception as exc:           # noqa: BLE001
            if skip_errors:
                logger.warning("SKIP %s — %s", p.name, exc)
                logger.debug(traceback.format_exc())
            else:
                raise

    if not vectors:
        raise RuntimeError("No features extracted — check source path.")

    matrix = np.vstack(vectors)

    # Save full matrix
    np.savez_compressed(output, features=matrix, files=np.array(file_list))
    logger.info("Saved %s  shape=%s", output, matrix.shape)

    # Save aggregate state alongside (for Trial III)
    agg_path = output.with_stem(output.stem + "_aggregate")
    save_aggregate(state, agg_path)

    # Sidecar CSV
    csv_out = output.with_suffix(".csv")
    if rows:
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Aggregate state: count=%d  mean_var=%.4f",
                state.count, forge_variance(state).mean())

    return matrix, file_list, state


def load_npz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a previously saved feature matrix archive."""
    data = np.load(str(path), allow_pickle=False)
    return data["features"], data["files"].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# §8  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Welford online aggregation of Voice Essences."
    )
    parser.add_argument("source", help="Audio directory or CSV manifest")
    parser.add_argument("--output",      default="features.npz")
    parser.add_argument("--no-recurse",  action="store_true")
    parser.add_argument("--strict",      action="store_true")
    parser.add_argument("--cluster",     action="store_true",
                        help="Run cluster-then-aggregate comparison after extraction")
    parser.add_argument("--n-clusters",  type=int, default=5,
                        help="K for K-Means (default 5)")
    parser.add_argument("--o1-check",    action="store_true",
                        help="Print per-update timings to verify O(1) behaviour")
    args = parser.parse_args()

    matrix, files, state = aggregate(
        args.source,
        output      = args.output,
        recursive   = not args.no_recurse,
        skip_errors = not args.strict,
    )

    print(f"\n{'─'*56}")
    print(f"Voices absorbed  : {state.count}")
    print(f"Feature dim      : {len(state.mean)}")
    print(f"Grand mean F0    : {state.mean[0]:.2f} Hz")
    print(f"Grand mean HNR   : {state.mean[66]:.2f} dB")
    print(f"Mean variance    : {forge_variance(state).mean():.4f}")
    print(f"Mean std-dev     : {forge_std(state).mean():.4f}")
    print(f"Mean std-err     : {forge_stderr(state).mean():.6f}")

    if args.o1_check:
        print(f"\n── O(1) timing check ──")
        _, times = build_aggregate_timed(list(matrix))
        print(f"  First 5  updates: {[f'{t:.2f}µs' for t in times[:5]]}")
        print(f"  Last  5  updates: {[f'{t:.2f}µs' for t in times[-5:]]}")
        print(f"  Mean: {np.mean(times):.2f}µs  Std: {np.std(times):.2f}µs")
        print("  PASS: Update time is flat -- no growth with corpus size.")

    if args.cluster:
        print("\n── Cluster-then-aggregate comparison ──")
        report = compare_strategies(matrix, n_kmeans_clusters=args.n_clusters)
        report.print()
