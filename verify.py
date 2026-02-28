"""
verify.py — Voice Essence membership verification and dataset QC.

═══════════════════════════════════════════════════════════════════════════════
  MEMBERSHIP VERIFICATION  (Trial III)
═══════════════════════════════════════════════════════════════════════════════

  Core question: given an aggregate chorus and a candidate voice, was that
  voice part of the chorus?

  Method: Diagonal Mahalanobis distance using the Welford aggregate state.
  The aggregate state from Trial II stores (count, mean, M2) per feature
  dimension, giving us a diagonal covariance matrix:

      Σ = diag(M2 / (count-1))    [sample covariance, Bessel-corrected]

  The diagonal Mahalanobis distance is then:

      d(x, μ, Σ) = sqrt( Σᵢ (xᵢ − μᵢ)² / σᵢ² )

  This is numerically identical to the L2 norm in standardised feature space.
  Under a Gaussian model, d² ~ χ²(k) for a point drawn from the distribution,
  so the chi-squared survival function gives a calibrated membership p-value.

  verify() returns -d (higher = closer = more likely member).
  The three scoring modes are:
    'mahalanobis'   — negative diagonal Mahalanobis distance   [default]
    'cosine'        — cosine similarity with the aggregate mean
    'log_likelihood'— log p(x | μ, diag(Σ)) under Gaussian

═══════════════════════════════════════════════════════════════════════════════
  THE EXPERIMENT  (accuracy vs. chorus size)
═══════════════════════════════════════════════════════════════════════════════

  For each chorus size N in {3,5,8,10,15,20,30,40}:
    • Draw N random voices as "members" → build AggregateState
    • Score ALL voices; members are positive class, rest negative
    • Compute AUC, accuracy/precision/recall at Youden-optimal threshold
    • Repeat R=100 random trials for stable statistics

  Expected result: AUC decreases as N grows — each voice contributes only
  1/N to the aggregate mean, so individual membership signal dilutes.
  The Crystal's memory grows blurrier as the chorus swells.

Public API
----------
  verify(state, essence, mode)          → float   (membership score)
  score_batch(state, essences, mode)    → ndarray (N,)
  p_value(state, essence)               → float   (chi-sq calibration)
  membership_experiment(X, ...)         → ExperimentResults
  plot_accuracy_vs_chorus_size(results) → saves PNG
  dataset_qc(source, ...)               → VerificationReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import scipy.stats
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)

from aggregate import AggregateState, forge_variance, forge_std, build_aggregate, load_npz
from extract import ESSENCE_DIM, ESSENCE_LAYOUT

logger = logging.getLogger(__name__)

ScoreMode = Literal["mahalanobis", "cosine", "log_likelihood"]

# Minimum variance floor — prevents division-by-zero in low-N regimes
VAR_FLOOR: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Scoring functions
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_variance(state: AggregateState) -> np.ndarray:
    """
    Return per-dimension sample variance, clipped to VAR_FLOOR.
    Safe for small N (avoids NaN/Inf in Mahalanobis distance).
    """
    return np.maximum(forge_variance(state), VAR_FLOOR)


def _mahalanobis_dist(state: AggregateState, essence: np.ndarray) -> float:
    """
    Diagonal Mahalanobis distance between *essence* and the aggregate mean.

      d = sqrt( sum_i ((x_i - mu_i)^2 / sigma_i^2) )

    This is the L2 distance in the space standardised by per-dimension
    standard deviation — equivalent to full Mahalanobis when the covariance
    matrix is diagonal (i.e., features are assumed independent).
    """
    var   = _safe_variance(state)
    delta = essence - state.mean
    return float(np.sqrt(np.sum(delta ** 2 / var)))


def _cosine_score(state: AggregateState, essence: np.ndarray) -> float:
    """
    Cosine similarity between *essence* and the aggregate mean vector.
    Range: [-1, 1].  Does not use variance; pure direction similarity.
    """
    mu_norm  = np.linalg.norm(state.mean)
    x_norm   = np.linalg.norm(essence)
    if mu_norm < 1e-12 or x_norm < 1e-12:
        return 0.0
    return float(np.dot(essence, state.mean) / (x_norm * mu_norm))


def _log_likelihood(state: AggregateState, essence: np.ndarray) -> float:
    """
    Log-likelihood of *essence* under a diagonal Gaussian model
    N(mu, diag(sigma^2)) fitted from the Welford aggregate.

      log p(x) = -0.5 * sum_i [ (x_i - mu_i)^2 / sigma_i^2 + log(2*pi*sigma_i^2) ]

    Larger (less negative) values → essence is more consistent with the aggregate.
    """
    var = _safe_variance(state)
    delta = essence - state.mean
    return float(-0.5 * np.sum(delta ** 2 / var + np.log(2 * np.pi * var)))


def verify(
    aggregate_state: AggregateState,
    candidate_essence: np.ndarray,
    mode: ScoreMode = "mahalanobis",
) -> float:
    """
    Score how likely *candidate_essence* is a member of the aggregate chorus.

    Parameters
    ----------
    aggregate_state : AggregateState
        The chorus aggregate from Trial II (count, mean, M2).
    candidate_essence : np.ndarray, shape (D,)
        The candidate voice's essence vector.
    mode : {'mahalanobis', 'cosine', 'log_likelihood'}
        Scoring metric.  'mahalanobis' is recommended — it uses both the
        aggregate mean AND variance from the Welford state.

    Returns
    -------
    float
        Membership score.  **Higher = more likely member.**

        mahalanobis   : -d  where d = diagonal Mahalanobis distance
        cosine        : cosine similarity in [−1, 1]
        log_likelihood: log p(x | mu, sigma^2), unbounded above

    Notes
    -----
    The Welford state gives a *diagonal* covariance (per-dimension variance).
    Full off-diagonal covariance is richer but requires O(D^2) storage and
    is not tracked by the online algorithm.  Diagonal Mahalanobis is still
    substantially more powerful than plain Euclidean distance because it
    downweights high-variance (noisy) dimensions and amplifies low-variance
    (stable) ones.
    """
    if aggregate_state.count < 2:
        # Variance undefined for N < 2; fall back to negative Euclidean distance
        return float(-np.linalg.norm(candidate_essence - aggregate_state.mean))

    if mode == "mahalanobis":
        return -_mahalanobis_dist(aggregate_state, candidate_essence)
    elif mode == "cosine":
        return _cosine_score(aggregate_state, candidate_essence)
    elif mode == "log_likelihood":
        return _log_likelihood(aggregate_state, candidate_essence)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: mahalanobis, cosine, log_likelihood.")


def score_batch(
    aggregate_state: AggregateState,
    essences: np.ndarray,
    mode: ScoreMode = "mahalanobis",
) -> np.ndarray:
    """
    Score every row of *essences* against the aggregate.  Fully vectorized —
    one numpy broadcast operation instead of a Python loop.

    Parameters
    ----------
    essences : np.ndarray, shape (N, D)
    Returns  : np.ndarray, shape (N,)  — membership scores (higher = member)
    """
    if aggregate_state.count < 2:
        # Fall back to negative Euclidean when variance is undefined
        delta = essences - aggregate_state.mean      # (N, D)
        return -np.linalg.norm(delta, axis=1)

    var = _safe_variance(aggregate_state)            # (D,)
    mu  = aggregate_state.mean                       # (D,)

    if mode == "mahalanobis":
        delta = essences - mu                        # (N, D)  broadcast
        return -np.sqrt(np.sum(delta ** 2 / var, axis=1))

    elif mode == "cosine":
        mu_norm = np.linalg.norm(mu)
        if mu_norm < 1e-12:
            return np.zeros(len(essences))
        x_norms = np.maximum(np.linalg.norm(essences, axis=1), 1e-12)   # (N,)
        return (essences @ mu) / (x_norms * mu_norm)

    elif mode == "log_likelihood":
        delta     = essences - mu                    # (N, D)
        log_const = float(np.sum(np.log(2 * np.pi * var)))   # scalar per state
        return -0.5 * (np.sum(delta ** 2 / var, axis=1) + log_const)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")


def p_value(aggregate_state: AggregateState, candidate_essence: np.ndarray) -> float:
    """
    Chi-squared p-value for the Mahalanobis distance.

    Under the diagonal Gaussian model, d^2 ~ chi^2(D) for a voice drawn
    from the aggregate distribution.  The survival function p = P(chi^2 >= d^2)
    gives the probability that a random member would be THIS far or farther.

    High p-value (close to 1) → consistent with aggregate → likely member.
    Low p-value (close to 0)  → anomalous distance → likely non-member.
    """
    if aggregate_state.count < 2:
        return 0.5
    d   = _mahalanobis_dist(aggregate_state, candidate_essence)
    dim = len(aggregate_state.mean)
    return float(scipy.stats.chi2.sf(d ** 2, df=dim))


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalMetrics:
    """Classification metrics at a given decision threshold."""
    threshold:  float
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    auc_roc:    float
    auc_pr:     float    # area under precision-recall curve
    tp: int
    tn: int
    fp: int
    fn: int

    def __str__(self) -> str:
        return (
            f"AUC-ROC={self.auc_roc:.3f}  AUC-PR={self.auc_pr:.3f}  "
            f"Acc={self.accuracy:.3f}  P={self.precision:.3f}  "
            f"R={self.recall:.3f}  F1={self.f1:.3f}  "
            f"TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}"
        )


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
) -> EvalMetrics:
    """
    Compute full suite of classification metrics.

    One call to roc_curve() and one to precision_recall_curve() — each used
    for both threshold selection AND AUC, so no duplicate sklearn work.

    Parameters
    ----------
    scores    : membership scores (higher = more likely member)
    labels    : binary ground truth (1 = member, 0 = non-member)
    threshold : decision boundary; if None, Youden-optimal is used.
    """
    # ROC curve — computed once, used for AUC + Youden threshold
    try:
        fpr, tpr, roc_thresh = roc_curve(labels, scores)
        auc_roc = float(np.trapz(tpr, fpr))          # trapezoid = exact AUC
        if threshold is None:
            j_idx     = int(np.argmax(tpr - fpr))    # Youden's J
            threshold = float(roc_thresh[j_idx])
    except ValueError:
        auc_roc   = float("nan")
        threshold = threshold or float(np.median(scores))

    # PR curve — computed once, AUC via trapezoid
    try:
        prec_c, rec_c, _ = precision_recall_curve(labels, scores)
        # precision_recall_curve returns decreasing recall; flip for trapz
        auc_pr = float(np.trapz(prec_c[::-1], rec_c[::-1]))
    except ValueError:
        auc_pr = float("nan")

    preds = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return EvalMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(labels, preds)),
        precision=float(p), recall=float(r), f1=float(f1),
        auc_roc=auc_roc, auc_pr=auc_pr,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §3  The Experiment — accuracy vs. chorus size
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    """Single (chorus_size, trial_index, mode) result."""
    chorus_size: int
    trial:       int
    mode:        str
    metrics:     EvalMetrics


@dataclass
class ExperimentResults:
    """
    Aggregated results from the accuracy-vs-chorus-size experiment.

    Attributes
    ----------
    chorus_sizes : list of N values tested
    modes : scoring modes used
    trials : raw per-trial TrialResult objects
    summary : chorus_size → mode → {mean_auc, std_auc, mean_acc, std_acc, …}
    """
    chorus_sizes: list[int]
    modes:        list[str]
    trials:       list[TrialResult]
    summary:      dict

    def print_table(self) -> None:
        """Pretty-print the AUC and accuracy summary table."""
        print("\n" + "=" * 80)
        print(f"{'Chorus':>8}  {'Mode':<15}  {'AUC-ROC':>8}  {'AUC-PR':>8}  "
              f"{'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
        print("-" * 80)
        for n in self.chorus_sizes:
            for mode in self.modes:
                s = self.summary[n][mode]
                print(
                    f"{n:>8}  {mode:<15}  "
                    f"{s['auc_roc_mean']:.3f}±{s['auc_roc_std']:.3f}  "
                    f"{s['auc_pr_mean']:.3f}±{s['auc_pr_std']:.3f}  "
                    f"{s['acc_mean']:.3f}  "
                    f"{s['prec_mean']:.3f}  "
                    f"{s['rec_mean']:.3f}  "
                    f"{s['f1_mean']:.3f}"
                )
        print("=" * 80)


def membership_experiment(
    X:            np.ndarray,
    chorus_sizes: list[int]   = (3, 5, 8, 10, 15, 20, 30, 40),
    n_trials:     int          = 100,
    modes:        list[str]    = ("mahalanobis", "cosine", "log_likelihood"),
    seed:         int          = 42,
) -> ExperimentResults:
    """
    Measure verification accuracy at multiple chorus sizes.

    For each N in *chorus_sizes*, for each of *n_trials* random seeds:
      1. Draw N voices uniformly at random as "members".
      2. Build an AggregateState from those N voices.
      3. Score ALL voices in X with verify().
      4. Evaluate AUC, accuracy, precision, recall (members=1, rest=0).

    The key question: how does AUC change as N grows?

    Parameters
    ----------
    X : np.ndarray (N_total, D)  — all extracted essence vectors
    chorus_sizes : chorus sizes to sweep over
    n_trials : random trials per chorus size (for stable statistics)
    modes : scoring methods to compare
    seed : master random seed (each trial uses seed + trial_idx)
    """
    N_total = len(X)
    all_trials: list[TrialResult] = []

    for n in chorus_sizes:
        if n >= N_total:
            logger.warning("Skipping chorus_size=%d (>= N_total=%d)", n, N_total)
            continue

        for trial in range(n_trials):
            rng     = np.random.default_rng(seed + trial * 997)
            indices = rng.choice(N_total, size=n, replace=False)
            members = set(indices.tolist())

            chorus    = X[indices]
            state     = build_aggregate(chorus)
            labels    = np.array([1 if i in members else 0 for i in range(N_total)])

            # Guard: skip if only one class (shouldn't happen but defensive)
            if labels.sum() == 0 or labels.sum() == N_total:
                continue

            for mode in modes:
                scores  = score_batch(state, X, mode=mode)
                metrics = evaluate(scores, labels)
                all_trials.append(TrialResult(chorus_size=n, trial=trial,
                                              mode=mode, metrics=metrics))

    # Aggregate per (chorus_size, mode)
    summary: dict = {}
    for n in chorus_sizes:
        summary[n] = {}
        for mode in modes:
            t_metrics = [t.metrics for t in all_trials
                         if t.chorus_size == n and t.mode == mode]
            if not t_metrics:
                continue

            def _arr(attr):
                return np.array([getattr(m, attr) for m in t_metrics
                                 if not np.isnan(getattr(m, attr))])

            summary[n][mode] = {
                "auc_roc_mean": float(np.mean(_arr("auc_roc"))),
                "auc_roc_std":  float(np.std(_arr("auc_roc"))),
                "auc_pr_mean":  float(np.mean(_arr("auc_pr"))),
                "auc_pr_std":   float(np.std(_arr("auc_pr"))),
                "acc_mean":     float(np.mean(_arr("accuracy"))),
                "acc_std":      float(np.std(_arr("accuracy"))),
                "prec_mean":    float(np.mean(_arr("precision"))),
                "rec_mean":     float(np.mean(_arr("recall"))),
                "f1_mean":      float(np.mean(_arr("f1"))),
                "n_trials":     len(t_metrics),
            }

    return ExperimentResults(
        chorus_sizes=list(chorus_sizes),
        modes=list(modes),
        trials=all_trials,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

MODE_STYLES: dict[str, dict] = {
    "mahalanobis":    {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "cosine":         {"color": "#ff7f0e", "marker": "s", "ls": "--"},
    "log_likelihood": {"color": "#2ca02c", "marker": "^", "ls": "-."},
}


def plot_accuracy_vs_chorus_size(
    results:   ExperimentResults,
    save_path: str | Path = "verification_accuracy.png",
) -> Path:  # noqa: C901
    """
    Generate and save the accuracy-vs-chorus-size chart.

    Top panel    : AUC-ROC with ±1 std error band per scoring mode
    Bottom panel : Accuracy, Precision, Recall at Youden threshold
                   (Mahalanobis only, for clarity)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    valid_sizes = [n for n in results.chorus_sizes if n in results.summary]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        "Crystal Memory: Verification Accuracy vs. Chorus Size\n"
        "Does the Crystal's memory sharpen or blur as the chorus grows?",
        fontsize=13, fontweight="bold",
    )

    # ── Top: AUC-ROC per mode ────────────────────────────────────────────────
    for mode in results.modes:
        style  = MODE_STYLES.get(mode, {"color": "gray", "marker": "o", "ls": "-"})
        x_vals, y_means, y_stds = [], [], []
        for n in valid_sizes:
            s = results.summary[n].get(mode)
            if s:
                x_vals.append(n)
                y_means.append(s["auc_roc_mean"])
                y_stds.append(s["auc_roc_std"])

        x   = np.array(x_vals)
        y   = np.array(y_means)
        err = np.array(y_stds)

        ax1.plot(x, y, label=mode, color=style["color"],
                 marker=style["marker"], linestyle=style["ls"], lw=2, ms=7)
        ax1.fill_between(x, y - err, y + err,
                         color=style["color"], alpha=0.15)

    ax1.axhline(0.5, color="gray", linestyle=":", lw=1.5, label="Random baseline (0.5)")
    ax1.axhline(1.0, color="lightgray", linestyle=":", lw=1)
    ax1.set_ylabel("AUC-ROC", fontsize=11)
    ax1.set_ylim(0.3, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title("Top panel: AUC-ROC — threshold-independent discrimination ability",
                  fontsize=10)

    # ── Bottom: Acc / Prec / Recall for Mahalanobis only ────────────────────
    x_m, acc_m, prec_m, rec_m = [], [], [], []
    for n in valid_sizes:
        s = results.summary[n].get("mahalanobis")
        if s:
            x_m.append(n)
            acc_m.append(s["acc_mean"])
            prec_m.append(s["prec_mean"])
            rec_m.append(s["rec_mean"])

    ax2.plot(x_m, acc_m,  "o-",  color="#1f77b4", label="Accuracy",   lw=2, ms=7)
    ax2.plot(x_m, prec_m, "s--", color="#9467bd", label="Precision",  lw=2, ms=7)
    ax2.plot(x_m, rec_m,  "^-.", color="#d62728", label="Recall",     lw=2, ms=7)
    ax2.axhline(0.5, color="gray", linestyle=":", lw=1.5)

    ax2.set_xlabel("Chorus size  N  (voices in aggregate)", fontsize=11)
    ax2.set_ylabel("Score at Youden threshold", fontsize=11)
    ax2.set_ylim(0.3, 1.05)
    ax2.legend(fontsize=9)
    ax2.set_title(
        "Bottom panel: Accuracy / Precision / Recall (Mahalanobis, Youden-optimal threshold)",
        fontsize=10,
    )

    # ── Shared annotations ───────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.set_xticks(valid_sizes)
        ax.grid(True, alpha=0.3)
        # Shade the small-N region where variance estimation is unreliable
        ax.axvspan(0, 4, alpha=0.06, color="red",
                   label="N<4: variance unreliable" if ax is ax1 else None)

    # Annotate the trend direction on ax1
    mahal_aucs = [results.summary[n]["mahalanobis"]["auc_roc_mean"]
                  for n in valid_sizes if "mahalanobis" in results.summary[n]]
    if mahal_aucs:
        slope = np.polyfit(valid_sizes[:len(mahal_aucs)], mahal_aucs, 1)[0]
        direction = "decreasing" if slope < 0 else "increasing"
        ax1.annotate(
            f"Trend: AUC is {direction} with chorus size\n"
            f"({'+' if slope >= 0 else ''}{slope:.4f} AUC/voice)",
            xy=(valid_sizes[len(valid_sizes) // 2], 0.37),
            fontsize=9, style="italic", color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved → %s", save_path)
    return save_path


def plot_roc_curves(
    X:            np.ndarray,
    chorus_sizes: list[int] = (5, 15, 30),
    seed:         int        = 42,
    save_path:    str | Path = "roc_curves.png",
) -> Path:
    """
    Plot ROC curves for a few representative chorus sizes (single trial each).
    Shows concretely how discrimination degrades as N grows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    save_path = Path(save_path)
    fig, ax   = plt.subplots(figsize=(7, 6))
    cmap      = plt.cm.plasma(np.linspace(0.1, 0.85, len(chorus_sizes)))

    for idx, n in enumerate(chorus_sizes):
        rng     = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=min(n, len(X) - 2), replace=False)
        members = set(indices.tolist())
        state   = build_aggregate(X[indices])
        labels  = np.array([1 if i in members else 0 for i in range(len(X))])
        scores  = score_batch(state, X, mode="mahalanobis")

        try:
            auc  = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, lw=2, color=cmap[idx],
                    label=f"N={n:2d}  (AUC={auc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Mahalanobis Membership Score\n"
                 "Curves flatten as chorus grows (individual memory dilutes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Dataset QC  (original verify() logic, renamed)
# ═══════════════════════════════════════════════════════════════════════════════

# Updated BOUNDS for the 82-dim ESSENCE_LAYOUT
ESSENCE_BOUNDS: dict[int, tuple[float, float]] = {
    0:  (0.0,    1000.0),   # F0 mean (Hz)
    1:  (0.0,     500.0),   # F0 std
    2:  (0.0,    1000.0),   # F0 min
    3:  (0.0,    1500.0),   # F0 max
    4:  (0.0,       1.0),   # voiced fraction
    5:  (-500.0,  500.0),   # F0 slope (Hz/s)
    6:  (0.0,      20.0),   # vibrato rate (Hz)
    7:  (0.0,     500.0),   # vibrato depth (cents)
    66: (-20.0,    50.0),   # HNR mean (dB)
    68: (0.0,       1.0),   # RMS mean
    72: (0.0,    8001.0),   # spectral centroid mean (Hz)
    76: (0.0,       1.0),   # ZCR mean
}

Z_THRESHOLD: float = 4.0


@dataclass
class VerificationReport:
    n_samples:          int
    n_features:         int
    has_nan:            bool
    has_inf:            bool
    bounds_violations:  dict[str, list[int]]
    duplicate_indices:  list[tuple[int, int]]
    outlier_indices:    dict[int, list[int]]
    passed:             bool

    def summary(self) -> str:
        return "\n".join([
            f"Samples : {self.n_samples}",
            f"Features: {self.n_features}",
            f"NaN     : {self.has_nan}",
            f"Inf     : {self.has_inf}",
            f"Bounds  : {sum(len(v) for v in self.bounds_violations.values())} violation(s)",
            f"Dupes   : {len(self.duplicate_indices)} pair(s)",
            f"Outliers: {sum(len(v) for v in self.outlier_indices.values())} flag(s)",
            f"PASSED  : {self.passed}",
        ])


def dataset_qc(
    source:      str | Path | np.ndarray,
    files:       Optional[list[str]] = None,
    z_threshold: float = Z_THRESHOLD,
    strict:      bool  = False,
) -> VerificationReport:
    """
    Run sanity checks on a feature matrix (NaN, bounds, duplicates, outliers).
    Preserved from Trial I — this is the dataset-level QC, distinct from the
    per-voice membership verify() above.
    """
    if isinstance(source, (str, Path)):
        features, files = load_npz(source)
    else:
        features = np.asarray(source, dtype=np.float64)
        files = files or []

    logger.info("Dataset QC: shape=%s", features.shape)

    has_nan = bool(np.isnan(features).any())
    has_inf = bool(np.isinf(features).any())

    violations: dict[str, list[int]] = {}
    for col, (lo, hi) in ESSENCE_BOUNDS.items():
        if col >= features.shape[1]:
            continue
        bad = np.where((features[:, col] < lo) | (features[:, col] > hi))[0].tolist()
        if bad:
            violations[f"feat[{col}]"] = bad

    dupes: list[tuple[int, int]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if np.array_equal(features[i], features[j]):
                dupes.append((i, j))

    outliers: dict[int, list[int]] = {}
    col_means = features.mean(axis=0)
    col_stds  = features.std(axis=0)
    for col in range(features.shape[1]):
        if col_stds[col] == 0:
            continue
        z      = np.abs((features[:, col] - col_means[col]) / col_stds[col])
        flagged = np.where(z > z_threshold)[0].tolist()
        if flagged:
            outliers[col] = flagged

    passed = not has_nan and not has_inf and not violations and not dupes
    report = VerificationReport(
        n_samples=features.shape[0], n_features=features.shape[1],
        has_nan=has_nan, has_inf=has_inf,
        bounds_violations=violations, duplicate_indices=dupes,
        outlier_indices=outliers, passed=passed,
    )

    if strict and not passed:
        raise ValueError(f"QC failed:\n{report.summary()}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# §6  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Voice membership verification and accuracy experiment."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- score a single file against a saved aggregate
    p_score = sub.add_parser("score", help="Score one voice against an aggregate.")
    p_score.add_argument("aggregate_npz",  help="Path to aggregate_state.npz")
    p_score.add_argument("essence_npz",    help="Path to essences.npz")
    p_score.add_argument("--index",  type=int, default=0, help="Row index in essence matrix")
    p_score.add_argument("--mode",   default="mahalanobis")

    # -- run the accuracy-vs-chorus-size experiment
    p_exp = sub.add_parser("experiment", help="Run the accuracy-vs-chorus-size experiment.")
    p_exp.add_argument("essences_npz",  help="Path to essences.npz")
    p_exp.add_argument("--trials",  type=int, default=100)
    p_exp.add_argument("--seed",    type=int, default=42)
    p_exp.add_argument("--chart",   default="verification_accuracy.png")
    p_exp.add_argument("--roc",     default="roc_curves.png")

    # -- dataset QC
    p_qc = sub.add_parser("qc", help="Run dataset QC on a feature archive.")
    p_qc.add_argument("npz")
    p_qc.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    if args.cmd == "score":
        from aggregate import load_aggregate
        state  = load_aggregate(args.aggregate_npz)
        data   = np.load(args.essence_npz, allow_pickle=False)
        ess    = data["X"][args.index]
        s      = verify(state, ess, mode=args.mode)
        pv     = p_value(state, ess)
        print(f"Membership score ({args.mode}): {s:.6f}")
        print(f"Chi-sq p-value:                {pv:.6f}")
        print(f"Interpretation: {'LIKELY MEMBER' if pv > 0.05 else 'LIKELY NON-MEMBER'}")

    elif args.cmd == "experiment":
        data   = np.load(args.essences_npz, allow_pickle=False)
        X      = data["X"]
        print(f"Running experiment on {len(X)} voices…")
        results = membership_experiment(X, n_trials=args.trials, seed=args.seed)
        results.print_table()
        chart = plot_accuracy_vs_chorus_size(results, save_path=args.chart)
        roc   = plot_roc_curves(X, save_path=args.roc)
        print(f"\nCharts saved: {chart}  |  {roc}")

    elif args.cmd == "qc":
        report = dataset_qc(args.npz, strict=args.strict)
        print(report.summary())
        raise SystemExit(0 if report.passed else 1)
