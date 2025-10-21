import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Optional, Tuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def find_bias_flag_column(df: pd.DataFrame) -> str:
    """Find the column representing bias flag using flexible heuristics."""
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "bias_flag":
            return c
    for c in df.columns:
        cl = str(c).strip().lower()
        if "bias" in cl and "flag" in cl:
            return c
    raise ValueError("Couldn't find a `bias_flag` column.")


def normalize_bias_flag(series: pd.Series) -> pd.Series:
    """Normalize various bias flag encodings to integer {0,1}."""
    if series.dtype == bool:
        return series.astype(int)
    if not pd.api.types.is_numeric_dtype(series):
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "t": 1,
            "f": 0,
        }
        return (
            series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)
        )
    return (series != 0).astype(int)


def maybe_drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop a leading index-like column if present (e.g., 'Unnamed: 0', 'index', 0..n-1)."""
    if df.shape[1] == 0:
        return df
    first_col = df.columns[0]
    first_col_l = str(first_col).strip().lower()
    if first_col_l.startswith("unnamed:") or first_col_l == "index":
        return df.iloc[:, 1:]
    # Heuristic: if values equal the positional index (0..n-1)
    try:
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if s.notna().all() and np.array_equal(s.values, np.arange(len(s))):
            return df.iloc[:, 1:]
    except Exception:
        pass
    return df


def format_feature_label(name: str) -> str:
    name = str(name)
    # Skip transformation for known non-MERRA variables
    if "MERRA" not in name.upper():
        return name
    s = name.replace("_", " ")
    # Normalize MERRA-2 token
    s = re.sub(r"(?i)merra[-_ ]?2", "MERRA-2", s)
    return s


def add_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute viewing/illumination geometry: relative azimuth Δφ and scattering angle Θ.

    Adds columns:
      - relative_azimuth_deg: 0..180 degrees
      - scattering_angle_deg: 0..180 degrees
    """
    if "sensor_azimuth_angle" in df.columns and "solar_azimuth_angle" in df.columns:
        sa = pd.to_numeric(df["sensor_azimuth_angle"], errors="coerce")
        so = pd.to_numeric(df["solar_azimuth_angle"], errors="coerce")
        # Δφ = |((sensor - solar + 180) mod 360) - 180|
        dphi = ((sa - so + 180.0) % 360.0) - 180.0
        df["relative_azimuth_deg"] = dphi.abs().clip(0.0, 180.0)

    if (
        "solar_zenith_angle" in df.columns
        and "sensor_zenith_angle" in df.columns
        and "relative_azimuth_deg" in df.columns
    ):
        sza = np.deg2rad(pd.to_numeric(df["solar_zenith_angle"], errors="coerce"))
        vza = np.deg2rad(pd.to_numeric(df["sensor_zenith_angle"], errors="coerce"))
        dphi_r = np.deg2rad(pd.to_numeric(df["relative_azimuth_deg"], errors="coerce").clip(0.0, 180.0))
        cos_theta = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(dphi_r)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.rad2deg(np.arccos(cos_theta))
        df["scattering_angle_deg"] = theta
    return df


def _wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.959963984540054) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score interval for binomial proportion.

    Returns (lower, upper). Undefined bins (n==0) return (nan, nan).
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / np.maximum(n, 1.0)
        denom = 1.0 + (z**2) / np.maximum(n, 1.0)
        center = (p + (z**2) / (2.0 * np.maximum(n, 1.0))) / denom
        margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * np.maximum(n, 1.0))) / np.maximum(n, 1.0))) / denom
    lo = np.clip(center - margin, 0.0, 1.0)
    hi = np.clip(center + margin, 0.0, 1.0)
    lo[np.where(n <= 0)] = np.nan
    hi[np.where(n <= 0)] = np.nan
    return lo, hi


def binned_rate_with_ci(
    x: pd.Series,
    y: pd.Series,
    edges: Optional[np.ndarray] = None,
    n_bins: int = 12,
    drop_min_frac: float = 0.01,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Bin x, compute bias rate and Wilson 95% CI per bin.

    Returns centers, p_hat, counts, ci_lo, ci_hi, lefts, widths; may return None if insufficient data.
    Drops bins with counts < drop_min_frac * total_valid.
    """
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return None
    xv = pd.to_numeric(x[valid], errors="coerce")
    yv = pd.to_numeric(y[valid], errors="coerce")
    yv = (yv != 0).astype(int)
    total = len(xv)
    if edges is None:
        lo, hi = np.nanmin(xv), np.nanmax(xv)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        edges = np.linspace(lo, hi, n_bins + 1)
    bins = pd.cut(xv, bins=edges, include_lowest=True, duplicates="drop")
    grp = pd.DataFrame({"y": yv, "x": xv, "bin": bins}).groupby("bin", observed=False)
    k = grp["y"].sum()
    n = grp["y"].count()
    p_hat = (k / n).astype(float)
    ci_lo, ci_hi = _wilson_interval(k.values, n.values)
    categories = bins.cat.categories
    centers = np.array([(iv.left + iv.right) / 2.0 for iv in categories])
    lefts = np.array([iv.left for iv in categories])
    widths = np.array([iv.right - iv.left for iv in categories])
    # Drop sparse bins
    keep = n.values >= max(1, int(np.ceil(drop_min_frac * total)))
    return (
        centers[keep],
        p_hat.values[keep],
        n.values[keep].astype(int),
        ci_lo[keep],
        ci_hi[keep],
        lefts[keep],
        widths[keep],
    )


def plot_1d_bias_with_ci(
    df: pd.DataFrame,
    x_col: str,
    bias_col: str,
    out_png: str,
    edges: Optional[np.ndarray] = None,
    n_bins: int = 12,
    drop_min_frac: float = 0.01,
    x_label: Optional[str] = None,
) -> Optional[str]:
    if x_col not in df.columns:
        return None
    out = binned_rate_with_ci(df[x_col], df[bias_col], edges=edges, n_bins=n_bins, drop_min_frac=drop_min_frac)
    if out is None:
        return None
    centers, p_hat, counts, lo, hi, lefts, widths = out
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(lefts, counts, width=widths, align="edge", alpha=0.3, color="#ff7f0e")
    ax1.set_ylabel("SAM count per bin")
    ax2 = ax1.twinx()
    ax2.vlines(centers, lo, hi, colors="#1f77b4", linewidth=1.2, alpha=0.9)
    ax2.plot(centers, p_hat, color="#1f77b4", linewidth=1.0, alpha=0.9)
    ax2.plot(centers, p_hat, linestyle="None", marker="o", markersize=4, color="#1f77b4")
    ax2.set_ylabel("P(swath-bias)")
    ax2.set_ylim(0.0, 0.4)
    ax1.set_xlabel(x_label or format_feature_label(x_col))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png




def binned_bias_rate_fixed(x: pd.Series, y: pd.Series, n_bins: int = 10) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return None
    # Constrain range to central percentiles to reduce outlier impact
    q1, q99 = np.nanpercentile(x, [2, 98])
    if not np.isfinite(q1) or not np.isfinite(q99) or q99 <= q1:
        return None
    in_range = (x >= q1) & (x <= q99)
    x = x[in_range]
    y = y[in_range]
    if len(x) < 3:
        return None
    edges = np.linspace(q1, q99, n_bins + 1)
    # Create categorical bins with fixed categories
    bins = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
    categories = bins.cat.categories
    grp = pd.DataFrame({"y": y, "x": x, "bin": bins}).groupby("bin", observed=False)
    # Align aggregates to the full set of interval categories
    rate = grp["y"].mean().reindex(categories)
    count = grp["y"].count().reindex(categories).fillna(0).astype(int)

    centers = np.array([(iv.left + iv.right) / 2 for iv in categories])
    lefts = np.array([iv.left for iv in categories])
    widths = np.array([iv.right - iv.left for iv in categories])
    return centers, rate.values, count.values, lefts, widths


def default_feature_list() -> List[str]:
    return [
        "Black_Carbon_AOD_Merra2",
        "Dust_AOD_Merra2",
        "Organic_Carbon_AOD_Merra2",
        "Sea_Salt_AOD_Merra2",
        "Sulfate_AOD_Merra2",
        "Angstroem_Merra2",
        "Total_AOD_Merra2",
        "albedo_o2a",
        "aod_bc",
        "aod_dust",
        "aod_ice",
        "aod_oc",
        "aod_seasalt",
        "aod_strataer",
        "aod_sulfate",
        "aod_total",
        "aod_water",
        "solar_zenith_angle",
        "sensor_zenith_angle",
        "sensor_azimuth_angle",
        "solar_azimuth_angle",
    ]


def save_correlation_heatmap(df: pd.DataFrame, out_png: str, method: str = "pearson") -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numeric columns to compute a correlation matrix.")
    corr = numeric_df.corr(method=method, min_periods=3)

    plt.figure(figsize=(max(8, corr.shape[1] * 0.4), max(6, corr.shape[0] * 0.4)))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3)
    plt.title(f"Correlation matrix ({method})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return corr


def save_bias_correlations(df: pd.DataFrame, bias_col: str, out_csv: str, out_png: str, top_k: int = 30, method: str = "pearson") -> pd.Series:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if bias_col not in numeric_df.columns:
        numeric_df[bias_col] = normalize_bias_flag(df[bias_col])
    # Compute Pearson correlation between binary bias flag and numeric features (point-biserial)
    series = numeric_df.drop(columns=[bias_col]).apply(lambda s: s.corr(numeric_df[bias_col], method=method))
    series = series.dropna().sort_values(key=lambda v: v.abs(), ascending=False)
    series.to_csv(out_csv, header=["correlation_with_bias_flag"])

    top = series.head(top_k)
    plt.figure(figsize=(10, max(4, 0.35 * len(top))))
    sns.barplot(x=top.values, y=top.index, orient="h", palette="coolwarm", hue=None)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title(f"Top {min(top_k, len(series))} |corr| vs swath-bais ({method})")
    plt.xlabel("Correlation with swath-bias")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return series


def plot_binned_bias_rates(
    df: pd.DataFrame,
    bias_col: str,
    features: List[str],
    out_dir: str,
    n_bins: int = 20,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for feat in features:
        if feat not in df.columns:
            continue
        x = df[feat]
        y = df[bias_col]
        out = binned_bias_rate_fixed(x, y, n_bins=n_bins)
        if out is None:
            continue
        centers, rates, counts, lefts, widths = out

        fig, ax1 = plt.subplots(figsize=(6, 4))

        ax1.bar(lefts, counts, width=widths, align="edge", alpha=0.3, color="#ff7f0e")
        ax1.set_ylabel("SAM count per bin")

        ax1.legend(handles=[Line2D([0], [0], color="#1f77b4", marker="o", label="Bias rate"),
                            Rectangle((0, 0), 1, 1, facecolor="#ff7f0e", alpha=0.3, label="SAM count")], loc="lower right")
        ax2 = ax1.twinx()
        ax2.plot(centers, rates, marker="o", color="#1f77b4")
        ax1.set_xlabel(format_feature_label(feat))
        ax2.set_ylabel("P(swath-bias)")
        ax2.set_title(f"swath-bias rate vs. {feat}")

        fig.tight_layout()

        out_png = os.path.join(out_dir, f"bias_rate_vs__{feat}.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        saved.append(out_png)
    return saved


def plot_binned_bias_rates_panel(
    df: pd.DataFrame,
    bias_col: str,
    features: List[str],
    out_png: str,
    n_bins: int = 20,
    title: Optional[str] = None,
) -> Optional[str]:

    feats = [f for f in features if f in df.columns]


    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharey=False)
    axes = axes.flatten()

    letters = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]
    handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", label="swath-bias rate ± 1σ"),
        Rectangle((0, 0), 1, 1, facecolor="#ff7f0e", alpha=0.3, label="SAM count"),
    ]

    for i, feat in enumerate(feats):
        x = df[feat]
        y = df[bias_col]
        out = binned_bias_rate_fixed(x, y, n_bins=n_bins)
        if out is None:
            continue
        centers, rates, counts, lefts, widths = out

        ax1 = axes[i]
        ax1.bar(lefts, counts, width=widths, align="edge", alpha=0.3, color="#ff7f0e")
        ax1.set_ylabel("SAM count per bin")

        ax2 = ax1.twinx()
        # Vertical error lines (±1 SE) with circle at mean: "|-o-|" rotated
        p = np.asarray(rates, dtype=float)
        n = np.asarray(counts, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(np.clip(p, 0.0, 1.0) * (1.0 - np.clip(p, 0.0, 1.0)) / np.maximum(n, 1.0))
        ylo = np.clip(p - se, 0.0, 1.0)
        yhi = np.clip(p + se, 0.0, 1.0)
        mask = np.isfinite(p) & np.isfinite(se) & (n > 0)
        ax2.vlines(centers[mask], ylo[mask], yhi[mask], colors="#1f77b4", linewidth=1.2, alpha=0.9)
        # Connecting line through bin means
        ax2.plot(centers[mask], p[mask], color="#1f77b4", linewidth=1.0, alpha=0.9, zorder=2)
        # Circle markers at means
        ax2.plot(centers[mask], p[mask], linestyle="None", marker="o", markersize=4, color="#1f77b4", zorder=3)
        ax1.set_xlabel(format_feature_label(feat))
        ax2.set_ylabel("P(swath-bias)")
        ax1.set_ylim(0.0, 14000)
        ax2.set_ylim(0.0, 0.4)

        # Lettering in top-right
        ax1.text(0.98, 0.98, letters[i], transform=ax1.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")

    if title:
        fig.suptitle(title, y=0.98)
    # Shared legend
    # Use constrained_layout to better accommodate labels
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def parse_feature_arg(feature_arg: Optional[str]) -> List[str]:
    if feature_arg is None or feature_arg.strip().lower() in ("default", "auto"):
        return default_feature_list()
    # If a JSON list is provided
    s = feature_arg.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass
    # Fallback comma-separated
    return [p.strip() for p in s.split(",") if p.strip()]


def make_output_dir(base_dir: Optional[str], csv_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    if base_dir:
        out_dir = os.path.abspath(base_dir)
    else:
        out_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(csv_path), "..", "figures", "exploration", f"{base_name}"
            )
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explore correlations and swath-bias-rate relationships in a CSV."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to data/figures/exploration/<file>",
    )
    parser.add_argument(
        "--corr-method",
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method to use",
    )
    parser.add_argument(
        "--features",
        default="default",
        help="Feature list for binned bias plots: 'default', 'auto', comma-separated names, or JSON list",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of fixed-width bins for bias-rate plots",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="If set, also generate a 4x2 panel for selected MERRA2 variables",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="If set, compute geometry metrics and generate 1D/2D diagnostic plots",
    )

    args = parser.parse_args(argv)

    csv_path = os.path.abspath(args.input)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    out_dir = make_output_dir(args.outdir, csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # Load
    df = pd.read_csv(csv_path)
    df = maybe_drop_index_column(df)

    # Bias flag
    bias_col = find_bias_flag_column(df)
    df[bias_col] = normalize_bias_flag(df[bias_col])

    # Save correlations
    corr_png = os.path.join(out_dir, "correlation_matrix.png")
    corr_csv = os.path.join(out_dir, "correlation_matrix.csv")
    corr = save_correlation_heatmap(df, corr_png, method=args.corr_method)
    corr.to_csv(corr_csv)

    bias_corr_csv = os.path.join(out_dir, "correlation_vs_bias_flag.csv")
    bias_corr_png = os.path.join(out_dir, "correlation_vs_bias_flag_top.png")
    bias_corr = save_bias_correlations(
        df, bias_col=bias_col, out_csv=bias_corr_csv, out_png=bias_corr_png, method=args.corr_method
    )

    # Binned bias-rate plots
    features = parse_feature_arg(args.features)
    plots_dir = os.path.join(out_dir, "bias_rate_plots")
    saved_plots = plot_binned_bias_rates(
        df=df, bias_col=bias_col, features=features, out_dir=plots_dir, n_bins=args.bins
    )

    # Optional 6-panel figure for requested MERRA2 variables
    if args.panel:
        panel_feats = [
            "Black_Carbon_AOD_Merra2", "Organic_Carbon_AOD_Merra2",
            "Dust_AOD_Merra2",
            "Sea_Salt_AOD_Merra2",
            "Sulfate_AOD_Merra2",
            "Total_AOD_Merra2",
            "Angstroem_Merra2", "albedo_o2a"
        ]
        panel_png = os.path.join(out_dir, "bias_rate_panel_merra2.png")
        plot_binned_bias_rates_panel(
            df=df,
            bias_col=bias_col,
            features=panel_feats,
            out_png=panel_png,
            n_bins=args.bins
        )

    # Geometry diagnostics
    if args.geometry:
        df = add_geometry_columns(df)
        geom_dir = os.path.join(out_dir, "geometry")
        os.makedirs(geom_dir, exist_ok=True)

        # 1D: relative azimuth Δφ (0-180°), scattering angle Θ (0-180°), VZA, SZA
        one_d = [
            ("relative_azimuth_deg", np.linspace(0, 180, 12 + 1), "Relative azimuth Δφ (deg)"),
            ("scattering_angle_deg", np.linspace(0, 180, 12 + 1), "Scattering angle Θ (deg)"),
            ("sensor_zenith_angle", None, None),
            ("solar_zenith_angle", None, None),
        ]
        for col, edges, label in one_d:
            try:
                out_png = os.path.join(geom_dir, f"bias_vs__{col}.png")
                plot_1d_bias_with_ci(
                    df=df,
                    x_col=col,
                    bias_col=bias_col,
                    out_png=out_png,
                    edges=edges,
                    n_bins=12,
                    drop_min_frac=0.01,
                    x_label=label,
                )
            except Exception:
                pass


    # Write a small run manifest
    manifest = {
        "input_csv": csv_path,
        "output_dir": out_dir,
        "bias_col": bias_col,
        "num_rows": int(len(df)),
        "num_cols": int(df.shape[1]),
        "corr_method": args.corr_method,
        "features_requested": features,
        "num_bias_rate_plots": len(saved_plots),
        "saved_plots": [os.path.basename(p) for p in saved_plots],
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


