#!/usr/bin/env python3
"""
Build a CSV of per-SAM query points for reanalysis lookups (e.g., MERRA-2).

Output columns: id, lat, lon, time_utc, bias_flag

Sources used (auto mode):
- Model-derived bias flag from processed Lite outputs (swath_bias_corrected) when available
- Otherwise, fall back to human labels (label 0/1; exclude 2)

Coordinates are taken from target center coordinates (CLASP report), which are
approximately at the SAM center and suitable for coarse reanalysis sampling.

Date is derived from the SAM orbit via the orbit-date mapping file.

Usage:
  python -m src.analysis.build_sam_bias_query_csv [--source auto|model|labels] [--max-files N]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Ensure project root on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_paths import PathConfig
from src.utils.orbit_date_utils import find_sam_date
from src.utils.main_util import read_oco_netcdf_to_df


def load_target_centers(target_csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Load target center coordinates from CLASP report CSV.

    Expected columns: 'Target ID', 'Site Center WKT' with WKT like 'POINT(lon lat)'.
    Returns mapping: target_id -> (lat, lon)
    """
    if not target_csv_path.exists():
        raise FileNotFoundError(f"Target centers file not found: {target_csv_path}")

    df = pd.read_csv(target_csv_path)
    if 'Target ID' not in df.columns or 'Site Center WKT' not in df.columns:
        raise ValueError("Target centers CSV must contain 'Target ID' and 'Site Center WKT' columns")

    centers: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        tid = str(row['Target ID'])
        wkt = str(row['Site Center WKT'])
        try:
            # Format: 'POINT(lon lat)'
            lon_str, lat_str = wkt.split('(')[1].split(')')[0].split()
            lon = float(lon_str)
            lat = float(lat_str)
            centers[tid] = (lat, lon)
        except Exception:
            continue
    return centers


def parse_sam_id_to_parts(sam_id: str) -> Tuple[Optional[str], Optional[int]]:
    """Split SAM id 'targetId_orbit' into (target_id, orbit int)."""
    try:
        parts = str(sam_id).rsplit('_', 1)
        if len(parts) != 2:
            return None, None
        target_id, orbit_str = parts
        return target_id, int(orbit_str)
    except Exception:
        return None, None


def collect_from_processed_outputs(config: PathConfig, target_centers: Dict[str, Tuple[float, float]], max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Build per-SAM rows from processed Lite outputs using 'swath_bias_corrected' as bias flag.
    Uses target center lat/lon and orbit-date mapping for time_utc.
    """
    output_dir = Path(config.OUTPUT_FULL_DIR)
    if not output_dir.exists():
        return pd.DataFrame()

    nc_files = sorted(output_dir.glob('*.nc4'))
    if max_files is not None:
        nc_files = nc_files[:max_files]
    if not nc_files:
        return pd.DataFrame()

    rows = []
    for nc_path in tqdm(nc_files, desc="Scanning processed outputs"):
        try:
            required = ['sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id', 'swath_bias_corrected']
            df = read_oco_netcdf_to_df(str(nc_path), variables_to_read=required)
            if df is None or df.empty:
                continue
            if 'operation_mode' not in df.columns:
                continue
            df = df[df['operation_mode'] == 4].copy()
            if df.empty:
                continue
            # Construct SAM id
            df['orbit_str'] = df['orbit'].astype(int).astype(str)
            df['SAM'] = df['target_id'].fillna('none') + '_' + df['orbit_str']
            df = df[~df['SAM'].str.contains('none', case=False, na=False)]
            if df.empty:
                continue

            # Parse per-sounding UTC times from sounding_id
            def parse_sid_to_ts(sid_val):
                try:
                    sid_str = str(sid_val)
                    # Expected formats: YYYYMMDDHHMMSS or YYYYMMDDHHMM
                    if len(sid_str) >= 14:
                        return pd.Timestamp(datetime.strptime(sid_str[:14], '%Y%m%d%H%M%S'))
                    elif len(sid_str) >= 12:
                        return pd.Timestamp(datetime.strptime(sid_str[:12], '%Y%m%d%H%M'))
                except Exception:
                    return pd.NaT
                return pd.NaT
            df['sounding_time'] = df['sounding_id'].apply(parse_sid_to_ts)

            # Aggregate to SAM level using model-derived correction flag
            by_sam = df.groupby('SAM').agg(
                is_corrected=('swath_bias_corrected', 'max'),
                sam_time=('sounding_time', 'median')
            )
            for sam_id, row in by_sam.iterrows():
                target_id, orbit_num = parse_sam_id_to_parts(sam_id)
                if not target_id or orbit_num is None:
                    continue
                lat_lon = target_centers.get(target_id)
                if not lat_lon:
                    continue
                lat, lon = lat_lon
                # Prefer precise time from data; fallback to date from orbit mapping
                if pd.notna(row.get('sam_time')):
                    ts = row['sam_time']
                    # Format as ISO8601 with Z suffix
                    time_utc = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                else:
                    date_str = find_sam_date(sam_id)
                    time_utc = f"{date_str}T00:00:00Z" if date_str else None
                rows.append({
                    'id': sam_id,
                    'lat': lat,
                    'lon': lon,
                    'time_utc': time_utc,
                    'bias_flag': int(row['is_corrected'] >= 1)
                })
        except BaseException:
            # Be robust to occasional file issues
            continue

    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows).drop_duplicates(subset=['id']).reset_index(drop=True)
    return out_df


def collect_from_labels(config: PathConfig, target_centers: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Build per-SAM rows from human labels only (binary labels 0/1; exclude 2).
    """
    labels_path = Path(config.LABELS_FILE)
    if not labels_path.exists():
        return pd.DataFrame()

    labels_df = pd.read_csv(labels_path)
    if 'identifier' not in labels_df.columns or 'label' not in labels_df.columns:
        return pd.DataFrame()

    # Keep latest label per SAM and drop uncertain (2)
    labels_df = labels_df.drop_duplicates(subset=['identifier'], keep='last')
    labels_df = labels_df[labels_df['label'].isin([0, 1])].copy()
    if labels_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in labels_df.iterrows():
        sam_id = str(r['identifier'])
        target_id, orbit_num = parse_sam_id_to_parts(sam_id)
        if not target_id or orbit_num is None:
            continue
        lat_lon = target_centers.get(target_id)
        if not lat_lon:
            continue
        lat, lon = lat_lon
        date_str = find_sam_date(sam_id)
        rows.append({
            'id': sam_id,
            'lat': lat,
            'lon': lon,
            'time_utc': date_str,
            'bias_flag': int(r['label'])
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=['id']).reset_index(drop=True)


def build_query_points(source: str = 'auto', max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Build the per-SAM query points dataframe.

    source: 'model' -> use processed outputs only
            'labels' -> use labels only
            'auto' -> use model if available, else labels
    """
    config = PathConfig()
    # Load target centers once
    target_centers = load_target_centers(Path(config.TARGET_DATA_FILE))

    model_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    if source in ('auto', 'model'):
        model_df = collect_from_processed_outputs(config, target_centers, max_files=max_files)

    if source in ('auto', 'labels') and (model_df.empty or source == 'labels'):
        labels_df = collect_from_labels(config, target_centers)

    if source == 'model':
        return model_df
    if source == 'labels':
        return labels_df

    # Auto: prefer model rows, fill gaps with labels
    if model_df.empty and labels_df.empty:
        return pd.DataFrame(columns=['id', 'lat', 'lon', 'time_utc', 'bias_flag'])

    if model_df.empty:
        return labels_df

    if labels_df.empty:
        return model_df

    # Merge, preferring model bias_flag when duplicate ids exist
    merged = pd.merge(labels_df, model_df, on=['id', 'lat', 'lon', 'time_utc'], how='outer', suffixes=('_label', '_model'))
    # If both exist for same id, take model bias_flag
    def resolve_bias(row):
        if pd.notna(row.get('bias_flag_model')):
            return int(row['bias_flag_model'])
        return int(row['bias_flag_label']) if pd.notna(row.get('bias_flag_label')) else None

    merged['bias_flag'] = merged.apply(resolve_bias, axis=1)
    out = merged[['id', 'lat', 'lon', 'time_utc', 'bias_flag']].dropna(subset=['id', 'lat', 'lon'])
    out = out.drop_duplicates(subset=['id']).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build per-SAM CSV for reanalysis queries")
    parser.add_argument('--source', choices=['auto', 'model', 'labels'], default='auto', help='Data source for bias_flag')
    parser.add_argument('--max-files', type=int, default=None, help='Limit number of processed files to scan (model source)')
    args = parser.parse_args()

    config = PathConfig()
    config.ensure_output_dirs()

    df = build_query_points(source=args.source, max_files=args.max_files)
    if df is None or df.empty:
        print("No data found to write.")
        return 1

    # Save to intermediate directory
    output_dir = Path(config.INTERMEDIATE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sam_bias_query_points.csv'
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


    return 0


if __name__ == '__main__':
    sys.exit(main())


