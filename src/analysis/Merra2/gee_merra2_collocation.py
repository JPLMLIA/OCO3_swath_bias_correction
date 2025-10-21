#!/usr/bin/env python3
"""
Google Earth Engine collocation of MERRA-2 variables at SAM query points.

Inputs:
  - CSV with columns: id,lat,lon,time_utc,bias_flag (from build_sam_bias_query_csv.py)

Outputs:
  - Export to Google Drive or Cloud Storage (asynchronous), or
  - Local small-batch retrieval for testing (synchronous; limited scale)

Auth options:
  - Interactive user auth (default): ee.Authenticate(); ee.Initialize()
  - Service account: --service-account and --private-key; optional --project

Usage examples:
  python -m src.analysis.gee_merra2_collocation --input-csv data/intermediate/sam_bias_query_points.csv --output-mode drive
  python -m src.analysis.gee_merra2_collocation --output-mode local --limit 50 --attach attach
  python -m src.analysis.gee_merra2_collocation --output-mode gcs --gcs-bucket my-bucket \
      --service-account svc@proj.iam.gserviceaccount.com --private-key /path/key.json --project my-gcp-proj
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure project root on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_paths import PathConfig


def init_ee(service_account: Optional[str] = None, private_key: Optional[str] = None, project: Optional[str] = None):
    import ee
    try:
        if service_account and private_key:
            creds = ee.ServiceAccountCredentials(service_account, private_key)
            if project:
                ee.Initialize(credentials=creds, project=project)
            else:
                ee.Initialize(credentials=creds)
            print(f"Initialized Earth Engine with service account {service_account}")
        else:
            try:
                ee.Initialize(project=project) if project else ee.Initialize()
                print("Initialized Earth Engine with existing user credentials")
            except Exception:
                print("No existing credentials found; starting interactive authentication...")
                ee.Authenticate()
                ee.Initialize(project=project) if project else ee.Initialize()
                print("Initialized Earth Engine after interactive authentication")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Earth Engine: {e}")


def build_feature_collection(df: pd.DataFrame, id_col: str = 'id'):
    import ee
    # Ensure types and presence
    required_cols = ['id', 'lat', 'lon', 'time_utc', 'bias_flag']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Input CSV missing column: {c}")

    # Convert rows to ee.Feature list
    features: List[ee.Feature] = []
    for _, r in df.iterrows():
        try:
            props = {
                'id': str(r['id']),
                'bias_flag': int(r['bias_flag']) if pd.notna(r['bias_flag']) else None,
                'time_utc': str(r['time_utc'])
            }
            geom = ee.Geometry.Point([float(r['lon']), float(r['lat'])])
            features.append(ee.Feature(geom, props))
        except Exception:
            continue
    return ee.FeatureCollection(features)


def attach_merra2(fc, bands: List[str], minutes_window: int = 90, scale_m: int = 60000):
    import ee
    merra = ee.ImageCollection("NASA/GSFC/MERRA/aer/2").select(bands)

    def _map(feat):
        t = ee.Date(feat.get('time_utc'))
        window = merra.filterDate(t.advance(-minutes_window, 'minute'), t.advance(minutes_window, 'minute'))
        img = window.sort('system:time_start').first()
        # If no image, return feature unchanged
        def when_img():
            vals = img.sample(feat.geometry(), scale=scale_m).first()
            return feat.set(vals.toDictionary(bands))
        return ee.Algorithms.If(img, when_img(), feat)

    return fc.map(_map)


def export_collection(out_fc, mode: str, description: str, output_dir: Path, gcs_bucket: Optional[str] = None):
    import ee
    if mode == 'drive':
        task = ee.batch.Export.table.toDrive(
            collection=out_fc,
            description=description,
            fileFormat='CSV'
        )
        task.start()
        print(f"Started Drive export task: {description}. Check your GEE Tasks tab.")
        return None
    elif mode == 'gcs':
        if not gcs_bucket:
            raise ValueError("--gcs-bucket is required for gcs export mode")
        task = ee.batch.Export.table.toCloudStorage(
            collection=out_fc,
            description=description,
            bucket=gcs_bucket,
            fileFormat='CSV'
        )
        task.start()
        print(f"Started GCS export task to bucket '{gcs_bucket}': {description}")
        return None
    elif mode == 'local':
        # Synchronous small download for testing; limited to avoid timeouts
        size = out_fc.size().getInfo()
        limit = min(500, size)  # cap for safety
        feats = out_fc.limit(limit).getInfo().get('features', [])
        rows = []
        for f in feats:
            props = f.get('properties', {})
            props = {k: props.get(k, None) for k in props.keys()}
            rows.append(props)
        df = pd.DataFrame(rows)
        out_path = output_dir / 'merra2_collocation_results_local.csv'
        df.to_csv(out_path, index=False)
        print(f"Wrote local results to {out_path} (rows={len(df)})")
        return out_path
    else:
        raise ValueError(f"Unknown output mode: {mode}")


def maybe_attach_to_input(input_csv: Path, results_csv: Path, output_dir: Path) -> Path:
    base = pd.read_csv(input_csv)
    res = pd.read_csv(results_csv)
    # Merge on 'id'
    merged = base.merge(res, on='id', how='left', suffixes=('', '_merra2'))
    out_path = output_dir / 'sam_bias_query_points_with_merra2.csv'
    merged.to_csv(out_path, index=False)
    print(f"Wrote merged CSV to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Collocate MERRA-2 via Earth Engine at SAM points")
    parser.add_argument('--input-csv', type=str, default=None, help='Path to input CSV; defaults to intermediate CSV from PathConfig')
    parser.add_argument('--output-mode', choices=['drive', 'gcs', 'local'], default='drive', help='Export destination')
    parser.add_argument('--gcs-bucket', type=str, default=None, help='GCS bucket for gcs mode')
    parser.add_argument('--attach', choices=['attach', 'separate'], default='separate', help='If local mode, whether to merge back into input CSV')
    parser.add_argument('--service-account', type=str, default=None, help='Service account email')
    parser.add_argument('--private-key', type=str, default=None, help='Path to service account key JSON')
    parser.add_argument('--project', type=str, default=None, help='GCP project ID for EE initialization')
    parser.add_argument('--minutes-window', type=int, default=90, help='Time window +/- minutes')
    parser.add_argument('--scale-m', type=int, default=60000, help='Sampling scale in meters')
    parser.add_argument('--limit', type=int, default=None, help='Limit input points (for testing)')
    parser.add_argument('--bands', nargs='*', default=['TOTEXTTAU','BCEXTTAU','SUEXTTAU','DUEXTTAU','SSEXTTAU','OCEXTTAU','TOTANGSTR'], help='MERRA-2 bands')
    args = parser.parse_args()

    # Initialize paths and EE
    config = PathConfig()
    output_dir = Path(config.INTERMEDIATE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_ee(service_account=args.service_account, private_key=args.private_key, project=args.project)

    # Load input CSV
    input_csv = Path(args.input_csv) if args.input_csv else (Path(config.INTERMEDIATE_DIR) / 'sam_bias_query_points.csv')
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if args.limit is not None:
        df = df.head(int(args.limit))

    # Build EE FeatureCollection and attach MERRA-2 values
    fc = build_feature_collection(df)
    out_fc = attach_merra2(fc, bands=args.bands, minutes_window=args.minutes_window, scale_m=args.scale_m)

    # Export or local download
    description = 'aod_merra2_collocation'
    results_path = export_collection(out_fc, mode=args.output_mode, description=description, output_dir=output_dir, gcs_bucket=args.gcs_bucket)

    # Optionally attach to input CSV for local mode
    if args.output_mode == 'local' and results_path is not None and args.attach == 'attach':
        maybe_attach_to_input(input_csv, results_path, output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())


