#!/usr/bin/env python3
"""
Compare RF classifier performance with Bell et al. 2023 paper's simple SB-flag method.

This script:
- Loads the labeled SAM list (excludes uncertain label=2)
- Computes current RF predictions from stored CV results
- Computes Bell et al. SB flag per SAM from Lite files (|dp_abp|<16, min N, swath clustering)
- Reports F1, precision, recall for both methods on the SAME SAM set

Usage:
    python -m src.analysis.compare_bell_vs_rf
"""

import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config_paths import PathConfig
from src.utils.main_util import read_oco_netcdf_to_df


def bell_sb_flag_for_sam(df_sam: pd.DataFrame,
                         k_candidates: Tuple[int, ...] = (4, 5, 6),
                         min_swath_n: int = 100) -> Dict[str, object]:
    """Compute Bell et al.'s SB flag for a single SAM DataFrame.

    Expected columns in df_sam: 'xco2', 'dp_abp', 'latitude', 'longitude'.
    Returns dict: {sb_flag: bool, sb_ratio: float, n_swaths: int, reason: Optional[str]}.
    """
    required_cols = ['xco2', 'dp_abp', 'latitude', 'longitude']
    for col in required_cols:
        if col not in df_sam.columns:
            return dict(sb_flag=False, reason=f"missing:{col}", sb_ratio=np.nan, n_swaths=0)

    # 1) Bell filter: keep "good" soundings (vEarly style). If using v10+, this is still fine for comparison.
    df = df_sam[np.abs(df_sam['dp_abp']) < 16].copy()
    if len(df) < 500:
        return dict(sb_flag=False, reason="N<500", sb_ratio=np.nan, n_swaths=0)

    # 3) Build across-swath coordinate using PCA; across-swath is minor axis (PC2)
    lat0, lon0 = df['latitude'].median(), df['longitude'].median()
    x = (df['longitude'] - lon0) * np.cos(np.deg2rad(lat0))
    y = (df['latitude'] - lat0)
    XY = np.c_[x.values, y.values]

    try:
        pcs = PCA(n_components=2).fit(XY).components_
    except Exception:
        return dict(sb_flag=False, reason="PCA_fail", sb_ratio=np.nan, n_swaths=0)
    proj = XY @ pcs.T
    across = proj[:, 1].reshape(-1, 1)

    best = None
    for k in k_candidates:
        try:
            labels = KMeans(n_clusters=k, n_init=50, random_state=0).fit_predict(across)
        except Exception:
            continue
        counts = pd.Series(labels).value_counts()
        keep_mask = np.isin(labels, counts[counts >= min_swath_n].index)
        if not keep_mask.any():
            continue

        df2 = df.loc[keep_mask].copy()
        df2['swath'] = labels[keep_mask]
        g = df2.groupby('swath')['xco2']
        sd_median = g.median().std(ddof=1)
        mean_sd = g.std(ddof=1).mean()
        if mean_sd is None or np.isnan(mean_sd) or mean_sd == 0:
            continue
        sb_ratio = float(sd_median / mean_sd)
        cand = dict(k=k, sb_ratio=sb_ratio, n_swaths=int(g.ngroups))
        if (best is None) or (cand['sb_ratio'] > best['sb_ratio']):
            best = cand

    if best is None or best['n_swaths'] < 4:
        return dict(sb_flag=False, reason="Too few valid swaths", sb_ratio=np.nan, n_swaths=0)

    return dict(sb_flag=best['sb_ratio'] > 0.75, sb_ratio=best['sb_ratio'], n_swaths=best['n_swaths'])


def load_labels(config: PathConfig) -> pd.DataFrame:
    labels_path = config.LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found at {labels_path}")
    labels_df = pd.read_csv(labels_path)
    # Normalize column names
    if 'identifier' not in labels_df.columns or 'label' not in labels_df.columns:
        raise ValueError("Labels CSV must have columns 'identifier' and 'label'")
    # Exclude uncertain class=2
    labels_df = labels_df[labels_df['label'] != 2].copy()
    labels_df['identifier'] = labels_df['identifier'].astype(str)
    labels_df['true_label'] = (labels_df['label'] == 1).astype(int)
    return labels_df[['identifier', 'true_label']]


def load_rf_predictions_same_sams(config: PathConfig, target_sams: Set[str]) -> pd.DataFrame:
    """Load RF predictions strictly from CV folds (out-of-sample) and subset to target_sams."""
    preds: List[pd.DataFrame] = []
    base = config.PROCESSED_EXPERIMENT_DIR
    for fold_dir in base.iterdir():
        if fold_dir.is_dir() and fold_dir.name.startswith('final_best_config_fold_'):
            fp = fold_dir / 'fold_predictions.csv'
            if fp.exists():
                dfp = pd.read_csv(fp)
                dfp['fold_dir'] = fold_dir.name
                preds.append(dfp)
    if not preds:
        raise FileNotFoundError("No CV fold predictions found under processed experiment directory.")
    all_preds = pd.concat(preds, ignore_index=True)

    all_preds['sam_id'] = all_preds['sam_id'].astype(str)
    all_preds = all_preds.drop_duplicates(subset=['sam_id'])
    subset = all_preds[all_preds['sam_id'].isin(target_sams)].copy()
    subset.rename(columns={'sam_id': 'identifier', 'predicted_label': 'rf_prediction', 'true_label': 'rf_true_label'}, inplace=True)
    # If true label missing here, we'll merge with labels later
    return subset[['identifier', 'rf_prediction']]


def iter_lite_files(config: PathConfig) -> Iterable[str]:
    """Yield candidate NetCDF file paths: prefer processed outputs, then raw Lite files."""
    # Prefer processed outputs (typically include all needed variables and are local products)
    output_dir = config.OUTPUT_FULL_DIR
    if isinstance(output_dir, (str, os.PathLike)) and os.path.isdir(output_dir):
        processed_paths = sorted(glob.glob(os.path.join(output_dir, '*.nc4')))
        for p in processed_paths:
            if os.path.isfile(p):
                yield p

    # Fallback to raw Lite files
    pattern = config.get_lite_files_pattern()
    for p in glob.iglob(pattern):
        if os.path.isfile(p):
            yield p


def collect_bell_flags_for_labeled_sams(config: PathConfig, target_sams: Set[str]) -> pd.DataFrame:
    """Scan Lite/raw files, compute Bell flags for labeled SAMs, and return DataFrame.

    This function first tries to use processed per-sounding evaluation parquet files
    (eval_data_corrected.parquet in each fold), which already contain required
    variables for Bell's method. If a SAM is not found there, it falls back to
    scanning NetCDF files.
    """
    needed = set(target_sams)
    results: List[Dict[str, object]] = []

    required_vars = [
        'sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id',
        'xco2', 'dp_abp'
    ]

    # --- Try processed per-sounding eval parquet files first ---
    eval_paths = []
    try:
        base = config.PROCESSED_EXPERIMENT_DIR
        for fold_dir in base.iterdir():
            if fold_dir.is_dir() and fold_dir.name.startswith('final_best_config_fold_'):
                p = fold_dir / 'eval_data_corrected.parquet'
                if p.exists():
                    eval_paths.append(p)
    except Exception:
        eval_paths = []

    if eval_paths:
        # Read only required columns, plus 'SAM' if present
        cols = list({*required_vars, 'SAM'})
        eval_frames = []
        for p in eval_paths:
            try:
                dfp = pd.read_parquet(p, columns=[c for c in cols if c in pd.read_parquet(p, columns=None).columns])
            except Exception:
                # Fallback: read all, then trim
                try:
                    dfp = pd.read_parquet(p)
                except Exception:
                    continue
            # Ensure SAM column exists
            if 'SAM' not in dfp.columns and {'orbit', 'target_id'}.issubset(dfp.columns):
                try:
                    dfp['orbit_str'] = dfp['orbit'].astype(int).astype(str)
                except Exception:
                    dfp['orbit_str'] = dfp['orbit'].astype(str)
                if dfp['target_id'].dtype == 'object' and len(dfp) > 0 and isinstance(dfp['target_id'].iloc[0], (bytes, bytearray)):
                    dfp['target_id'] = dfp['target_id'].str.decode('utf-8').str.strip()
                else:
                    dfp['target_id'] = dfp['target_id'].astype(str).str.strip()
                dfp['SAM'] = dfp['target_id'] + '_' + dfp['orbit_str']
            eval_frames.append(dfp)

        if eval_frames:
            eval_df = pd.concat(eval_frames, ignore_index=True)
            # Filter to SAMs of interest to reduce compute
            eval_df = eval_df[eval_df['SAM'].isin(needed)].copy()
            for sam_id, df_one in eval_df.groupby('SAM'):
                bell = bell_sb_flag_for_sam(df_one)
                results.append({
                    'identifier': sam_id,
                    'bell_flag': int(bool(bell.get('sb_flag', False))),
                    'bell_ratio': bell.get('sb_ratio', np.nan),
                    'bell_n_swaths': bell.get('n_swaths', 0),
                    'bell_reason': bell.get('reason', None),
                })
                if sam_id in needed:
                    needed.remove(sam_id)

    # --- Fallback: scan NetCDF files (processed outputs, then raw Lite) ---
    for nc_path in iter_lite_files(config):
        if not needed:
            break
        try:
            df = read_oco_netcdf_to_df(nc_path, variables_to_read=required_vars)
        except Exception as e:
            print(f"Warning: failed to read {os.path.basename(nc_path)}: {e}")
            continue

        if df.empty or 'operation_mode' not in df.columns:
            continue
        df_sam = df[df['operation_mode'] == 4].copy()
        if df_sam.empty or 'orbit' not in df_sam.columns or 'target_id' not in df_sam.columns:
            continue

        # Build SAM identifier
        try:
            df_sam['orbit_str'] = df_sam['orbit'].astype(int).astype(str)
        except Exception:
            df_sam['orbit_str'] = df_sam['orbit'].astype(str)
        # Handle bytes target_id if present
        if df_sam['target_id'].dtype == 'object' and len(df_sam) > 0 and isinstance(df_sam['target_id'].iloc[0], (bytes, bytearray)):
            df_sam['target_id'] = df_sam['target_id'].str.decode('utf-8').str.strip()
        else:
            df_sam['target_id'] = df_sam['target_id'].astype(str).str.strip()
        df_sam['SAM'] = df_sam['target_id'] + '_' + df_sam['orbit_str']

        present = set(df_sam['SAM'].unique()) & needed
        if not present:
            continue

        for sam_id in present:
            df_one = df_sam[df_sam['SAM'] == sam_id]
            bell = bell_sb_flag_for_sam(df_one)
            results.append({
                'identifier': sam_id,
                'bell_flag': int(bool(bell.get('sb_flag', False))),
                'bell_ratio': bell.get('sb_ratio', np.nan),
                'bell_n_swaths': bell.get('n_swaths', 0),
                'bell_reason': bell.get('reason', None),
            })
            needed.discard(sam_id)

    if needed:
        print(f"Warning: Bell flags missing for {len(needed)} SAM(s); they will be excluded from comparison.")

    if not results:
        return pd.DataFrame(columns=['identifier', 'bell_flag', 'bell_ratio', 'bell_n_swaths', 'bell_reason'])
    return pd.DataFrame(results)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(precision=precision, recall=recall, f1=f1)


def main() -> int:
    config = PathConfig()

    print("Loading labels...")
    labels_df = load_labels(config)
    labeled_sams: Set[str] = set(labels_df['identifier'].astype(str))
    print(f"Labeled SAMs (binary, excluding uncertain): {len(labeled_sams)}")

    print("Loading RF predictions...")
    rf_preds_df = load_rf_predictions_same_sams(config, labeled_sams)
    print(f"RF predictions loaded for {len(rf_preds_df)} SAMs")

    # Compute Bell flags only for SAMs we will evaluate on
    relevant_sams = set(rf_preds_df['identifier']) & labeled_sams
    print(f"Computing Bell flags for {len(relevant_sams)} SAMs...")
    bell_df = collect_bell_flags_for_labeled_sams(config, relevant_sams)
    print(f"Bell flags computed for {len(bell_df)} SAMs")

    # Align all to the common set where we have labels, RF predictions, and Bell flags
    common_ids = set(labels_df['identifier']) & set(rf_preds_df['identifier']) & set(bell_df['identifier'])
    if not common_ids:
        print("No overlapping SAMs across labels, RF predictions, and Bell flags.")
        return 1

    labels_sub = labels_df[labels_df['identifier'].isin(common_ids)].copy()
    rf_sub = rf_preds_df[rf_preds_df['identifier'].isin(common_ids)].copy()
    bell_sub = bell_df[bell_df['identifier'].isin(common_ids)].copy()

    merged = labels_sub.merge(rf_sub, on='identifier', how='inner').merge(bell_sub, on='identifier', how='inner')

    y_true = merged['true_label'].values
    y_pred_rf = merged['rf_prediction'].astype(int).values
    y_pred_bell = merged['bell_flag'].astype(int).values

    rf_metrics = compute_metrics(y_true, y_pred_rf)
    bell_metrics = compute_metrics(y_true, y_pred_bell)

    print("\n=== Comparison on common SAM set ===")
    print(f"N SAMs: {len(merged)}")
    print(f"RF     -> F1: {rf_metrics['f1']:.3f}, Precision: {rf_metrics['precision']:.3f}, Recall: {rf_metrics['recall']:.3f}")
    print(f"Bell   -> F1: {bell_metrics['f1']:.3f}, Precision: {bell_metrics['precision']:.3f}, Recall: {bell_metrics['recall']:.3f}")

    # Optional detailed reports
    print("\nRF classification report:")
    print(classification_report(y_true, y_pred_rf, target_names=['No Bias', 'Bias'], zero_division=0))
    print("Bell classification report:")
    print(classification_report(y_true, y_pred_bell, target_names=['No Bias', 'Bias'], zero_division=0))

    # Save summary CSV for reproducibility
    out_dir = Path(config.FIGURES_DIR) / 'bell_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    merged[['identifier', 'true_label', 'rf_prediction', 'bell_flag', 'bell_ratio', 'bell_n_swaths', 'bell_reason']].to_csv(out_dir / 'comparison_data.csv', index=False)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'rf': rf_metrics, 'bell': bell_metrics, 'n_sams': len(merged)}, f, indent=2)
    print(f"Saved comparison data and metrics to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


