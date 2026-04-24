#!/usr/bin/env python3
"""Histogram of `max_relative_jump` over the cross-validation test set.

Produces Figure B1 of the paper: the class-conditional distribution of
``max_relative_jump`` across all out-of-fold SAMs, split by the manual
swath-bias label, with the 0.6 decision threshold overlaid.

Usage:
    python -m src.analysis.max_relative_jump_histogram \
        [--processed-dir <path>] [--output-dir <path>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.utils.config_paths import PathConfig

plt.style.use("default")
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "DejaVu Sans"

DECISION_THRESHOLD: float = 0.6
FOLD_PREFIX: str = "final_best_config_fold_"

NO_BIAS_COLOR: str = "#1f77b4"
BIAS_COLOR: str = "#ff7f0e"


def load_oof_sam_features(processed_dir: Path) -> pd.DataFrame:
    """Load and concatenate per-fold SAM-feature tables into the full OOF test set.

    Each labeled SAM appears exactly once because the four CV folds form a partition
    of the labeled (non-uncertain) data set.
    """
    fold_dirs = sorted(p for p in processed_dir.iterdir() if p.name.startswith(FOLD_PREFIX))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories ({FOLD_PREFIX}*) under {processed_dir}")

    frames: list[pd.DataFrame] = []
    for fold_dir in fold_dirs:
        feat_path = fold_dir / "sam_features.parquet"
        if not feat_path.exists():
            print(f"  warning: {feat_path} missing, skipping")
            continue
        frames.append(pd.read_parquet(feat_path))

    if not frames:
        raise FileNotFoundError(f"No sam_features.parquet found under {processed_dir}")

    combined = pd.concat(frames, axis=0)
    if combined.index.has_duplicates:
        n_dup = combined.index.duplicated().sum()
        print(f"  warning: {n_dup} duplicate SAM ids across folds; keeping first")
        combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def plot_max_relative_jump_histogram(
    sam_features: pd.DataFrame,
    *,
    threshold: float = DECISION_THRESHOLD,
    bin_width: float = 0.05,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot histogram of max_relative_jump split by manual label.

    Parameters
    ----------
    sam_features : DataFrame indexed by SAM id with columns ``max_relative_jump`` and
        ``true_label`` (0 = no bias, 1 = bias).
    threshold : decision boundary highlighted in the plot (default 0.6).
    bin_width : histogram bin width in units of the feature.
    output_path : if given, the figure is saved as ``<stem>.png`` and ``<stem>.pdf``.
    """
    required = {"max_relative_jump", "true_label"}
    missing = required - set(sam_features.columns)
    if missing:
        raise KeyError(f"sam_features missing required columns: {sorted(missing)}")

    df = sam_features[["max_relative_jump", "true_label"]].dropna()
    df = df[df["true_label"].isin([0, 1])]

    no_bias = df.loc[df["true_label"] == 0, "max_relative_jump"].to_numpy()
    bias = df.loc[df["true_label"] == 1, "max_relative_jump"].to_numpy()

    upper = float(np.nanpercentile(df["max_relative_jump"], 99.5))
    upper = max(upper, threshold + 0.2)
    bins = np.arange(0.0, upper + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(no_bias, bins=bins, alpha=0.7, color=NO_BIAS_COLOR,
            label=f"No swath bias (n={len(no_bias)})")
    ax.hist(bias, bins=bins, alpha=0.7, color=BIAS_COLOR,
            label=f"Swath bias (n={len(bias)})")

    no_bias_median = float(np.median(no_bias))
    bias_median = float(np.median(bias))
    ax.axvline(no_bias_median, color=NO_BIAS_COLOR, linestyle="--", linewidth=2,
               label=f"No-bias median: {no_bias_median:.2f}")
    ax.axvline(bias_median, color=BIAS_COLOR, linestyle="--", linewidth=2,
               label=f"Bias median: {bias_median:.2f}")
    ax.axvline(threshold, color="black", linestyle=":", linewidth=2,
               label=f"Decision threshold: {threshold:g}")

    ax.set_xlabel("max_relative_jump", fontsize=11)
    ax.set_ylabel("Number of SAMs", fontsize=11)
    ax.set_xlim(0.0, upper)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved figure to {output_path.with_suffix('.png')}")
        print(f"Saved figure to {output_path.with_suffix('.pdf')}")

    return fig


def summarize(df: pd.DataFrame, *, threshold: float = DECISION_THRESHOLD) -> None:
    """Print descriptive statistics that are useful for the response letter."""
    df = df[df["true_label"].isin([0, 1])].dropna(subset=["max_relative_jump"])
    print(f"\nOOF test set: {len(df)} labeled SAMs")
    print(f"  No swath bias (label=0): {(df['true_label'] == 0).sum()}")
    print(f"  Swath bias    (label=1): {(df['true_label'] == 1).sum()}")

    for label, name in [(0, "no bias"), (1, "bias")]:
        sub = df.loc[df["true_label"] == label, "max_relative_jump"]
        print(
            f"  {name:>8}: median={sub.median():.3f}, mean={sub.mean():.3f}, "
            f"p90={sub.quantile(0.90):.3f}, p99={sub.quantile(0.99):.3f}, "
            f"frac > {threshold:g} = {(sub > threshold).mean():.1%}"
        )


def parse_args() -> argparse.Namespace:
    config = PathConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.PROCESSED_EXPERIMENT_DIR,
        help="Directory containing the per-fold final_best_config_fold_*/sam_features.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.FIGURES_DIR,
        help="Directory to write the histogram figure into.",
    )
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--bin-width", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading per-fold SAM features from: {args.processed_dir}")
    sam_features = load_oof_sam_features(args.processed_dir)
    print(f"Loaded {len(sam_features)} SAMs across all folds")

    summarize(sam_features, threshold=args.threshold)

    output_path = args.output_dir / "max_relative_jump_histogram"
    plot_max_relative_jump_histogram(
        sam_features,
        threshold=args.threshold,
        bin_width=args.bin_width,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
