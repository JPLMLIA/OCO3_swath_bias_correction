import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from ..utils.config_paths import PathConfig
from ..utils.main_util import read_oco_netcdf_to_df, tg_overlap_before_after


def list_processed_files(config: PathConfig, max_files=None):
    pattern = os.path.join(config.OUTPUT_FULL_DIR, "*.nc4")
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    return files


def compute_tg_overlap_summary(max_files=None, radius_deg=0.1, samples_per_swath=100):
    config = PathConfig()
    files = list_processed_files(config, max_files=max_files)
    print(f"Found {len(files)} processed files to scan for TG scenes.")

    per_scene_metrics = []

    for fp in tqdm(files, desc="Scanning processed files"):
        # Read only necessary variables to speed up
        vars_needed = [
            'sounding_id', 'latitude', 'longitude', 'operation_mode', 'orbit', 'target_id',
            'xco2', 'xco2_swath_bc', 'pma_elevation_angle', 'swath_bias_corrected'
        ]
        df = read_oco_netcdf_to_df(fp, variables_to_read=vars_needed)
        if df.empty:
            continue

        df_tg = df[df['operation_mode'] == 2].copy()
        if df_tg.empty:
            continue

        df_tg['orbit_str'] = df_tg['orbit'].astype(int).astype(str)
        df_tg['SAM'] = df_tg['target_id'].fillna('none') + '_' + df_tg['orbit_str']

        # Keep only TG scenes that had any correction applied
        if 'swath_bias_corrected' in df_tg.columns:
            corrected_mask_per_scene = df_tg.groupby('SAM')['swath_bias_corrected'].max() == 1
            corrected_sams = corrected_mask_per_scene[corrected_mask_per_scene].index.tolist()
            if not corrected_sams:
                continue
            df_tg = df_tg[df_tg['SAM'].isin(corrected_sams)].copy()
        else:
            # If the flag is missing, skip since we cannot identify corrected scenes
            continue

        scene_stats = tg_overlap_before_after(
            df_tg,
            original_var='xco2',
            corrected_var_candidates=('xco2_swath_bc', 'xco2_swath-BC'),
            swath_grouping_threshold_angle=1.0,
            samples_per_swath=samples_per_swath,
            radius_deg=radius_deg,
            min_neighbors=3,
            min_soundings_for_swath=50
        )
        # Compute inter/intra-swath context metrics per scene
        inter_intra_stats = _inter_intra_swath_context(
            df_tg,
            original_var='xco2',
            corrected_var_candidates=('xco2_swath_bc', 'xco2_swath-BC'),
            swath_grouping_threshold_angle=1.0,
            min_soundings_for_swath=50
        )

        if not scene_stats.empty:
            if not inter_intra_stats.empty:
                scene_stats = scene_stats.merge(inter_intra_stats, on='SAM', how='left')
            scene_stats['file'] = os.path.basename(fp)
            per_scene_metrics.append(scene_stats)

    if not per_scene_metrics:
        print("No TG overlap metrics computed. Ensure processed files include Target-mode scenes and correction outputs.")
        return pd.DataFrame()

    all_stats = pd.concat(per_scene_metrics, ignore_index=True)

    # Derived perspective: normalized improvement vs inter-swath std (before)
    if 'inter_swath_std_before' in all_stats.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            all_stats['norm_improvement_vs_inter'] = all_stats['improvement'] / all_stats['inter_swath_std_before']

    # Simple summary
    print(f"Computed overlap metrics for {all_stats['SAM'].nunique()} TG scenes across {len(files)} files.")
    print(f"Mean before: {all_stats['overlap_std_before'].mean():.3f} ppm | After: {all_stats['overlap_std_after'].mean():.3f} ppm | Improvement: {all_stats['improvement'].mean():.3f} ppm")

    # Significance and effect sizes (paired across scenes)
    _print_and_attach_significance(all_stats)

    return all_stats


def _assign_swaths(df_scene: pd.DataFrame, swath_grouping_threshold_angle: float) -> pd.DataFrame:
    df_scene = df_scene.copy()
    # Ensure sorted along-track; read_oco_netcdf_to_df already sorts by sounding_id
    if 'pma_elevation_angle' not in df_scene.columns:
        return pd.DataFrame()
    df_scene['swath'] = (df_scene['pma_elevation_angle'].diff().abs() > swath_grouping_threshold_angle).cumsum()
    return df_scene


def _inter_intra_swath_context(
    df_tg: pd.DataFrame,
    original_var: str,
    corrected_var_candidates=(
        'xco2_swath_bc',
        'xco2_swath-BC'
    ),
    swath_grouping_threshold_angle: float = 1.0,
    min_soundings_for_swath: int = 50
) -> pd.DataFrame:
    # Determine corrected var
    corrected_var = None
    for cand in corrected_var_candidates:
        if cand in df_tg.columns:
            corrected_var = cand
            break
    if corrected_var is None:
        return pd.DataFrame()

    results = []
    for sam, df_scene in df_tg.groupby('SAM'):
        if len(df_scene) < min_soundings_for_swath:
            continue
        df_scene = _assign_swaths(df_scene, swath_grouping_threshold_angle)
        if 'swath' not in df_scene.columns:
            continue
        # Filter valid swaths
        swath_counts = df_scene.groupby('swath')[original_var].count()
        valid_swaths = swath_counts[swath_counts >= min_soundings_for_swath].index.tolist()
        if len(valid_swaths) < 2:
            continue
        df_scene = df_scene[df_scene['swath'].isin(valid_swaths)].copy()

        # Inter-swath std: std of swath means
        means_before = df_scene.groupby('swath')[original_var].mean()
        means_after = df_scene.groupby('swath')[corrected_var].mean()
        inter_before = float(np.nanstd(means_before.to_numpy()))
        inter_after = float(np.nanstd(means_after.to_numpy()))

        # Intra-swath residual std: mean of per-swath std of residuals to swath mean
        resid_before = []
        resid_after = []
        for swath_id, df_sw in df_scene.groupby('swath'):
            mu_b = float(means_before.loc[swath_id])
            mu_a = float(means_after.loc[swath_id])
            resid_before.append(np.nanstd((df_sw[original_var] - mu_b).to_numpy()))
            resid_after.append(np.nanstd((df_sw[corrected_var] - mu_a).to_numpy()))
        intra_before = float(np.nanmean(resid_before)) if len(resid_before) else np.nan
        intra_after = float(np.nanmean(resid_after)) if len(resid_after) else np.nan

        results.append({
            'SAM': sam,
            'inter_swath_std_before': inter_before,
            'inter_swath_std_after': inter_after,
            'intra_swath_std_before': intra_before,
            'intra_swath_std_after': intra_after,
            'inter_swath_improvement': inter_before - inter_after
        })

    return pd.DataFrame(results)


def _plot_composite_figure(stats_df: pd.DataFrame, output_dir: str, filename: str = 'tg_overlap_composite'):
    os.makedirs(output_dir, exist_ok=True)
    before = stats_df['overlap_std_before'].to_numpy()
    after = stats_df['overlap_std_after'].to_numpy()
    improvement = stats_df['improvement'].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_scatter, ax_hist = axes[0, 0], axes[0, 1]
    ax_cdf, ax_frac = axes[1, 0], axes[1, 1]

    # 1) Paired scatter (before vs after)
    lim_max = np.nanmax([before.max(), after.max()]) if len(before) else 1.0
    lim = (0, max(1e-6, lim_max))
    ax_scatter.scatter(before, after, s=12, alpha=0.5, edgecolor='none')
    ax_scatter.plot(lim, lim, 'k--', linewidth=1)
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel('Overlap std before (ppm)')
    ax_scatter.set_ylabel('Overlap std after (ppm)')
    ax_scatter.set_title('Per-scene overlap: before vs after')

    # 2) Histogram of improvements (before - after)
    if len(improvement) > 0:
        ax_hist.hist(improvement, bins=30, color='tab:blue', alpha=0.8)
        ax_hist.axvline(np.nanmean(improvement), color='r', linestyle='--', label=f"mean={np.nanmean(improvement):.3f}")
        ax_hist.legend()
    ax_hist.set_xlabel('Improvement (before − after) ppm')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Distribution of improvements')

    # 3) CDFs before vs after
    def ecdf(values):
        vals = np.sort(values[~np.isnan(values)])
        if len(vals) == 0:
            return np.array([0.0]), np.array([0.0])
        y = np.arange(1, len(vals) + 1) / len(vals)
        return vals, y

    x_b, y_b = ecdf(before)
    x_a, y_a = ecdf(after)
    ax_cdf.plot(x_b, y_b, label='Before')
    ax_cdf.plot(x_a, y_a, label='After')
    ax_cdf.set_xlabel('Overlap std (ppm)')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('CDFs of overlap std')
    ax_cdf.legend()

    # 4) Outcome summary and normalized context
    improved = np.sum(improvement > 0)
    unchanged = np.sum(improvement == 0)
    worsened = np.sum(improvement < 0)
    totals = np.array([improved, unchanged, worsened], dtype=float)
    if totals.sum() > 0:
        fracs = 100.0 * totals / totals.sum()
    else:
        fracs = np.array([0.0, 0.0, 0.0])
    ax_frac.bar(['Improved', 'Unchanged', 'Worsened'], fracs, color=['tab:green', 'gray', 'tab:red'])
    ax_frac.set_ylabel('Percent of scenes (%)')
    ax_frac.set_ylim(0, 100)
    title_suffix = ''
    if 'norm_improvement_vs_inter' in stats_df.columns:
        mean_norm = np.nanmean(stats_df['norm_improvement_vs_inter'].to_numpy())
        title_suffix = f"\nMean normalized (vs inter-swath): {mean_norm:.2f}"
    ax_frac.set_title('Outcome summary' + title_suffix)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f'{filename}.png')
    out_pdf = os.path.join(output_dir, f'{filename}.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved composite figure to: {out_png} and {out_pdf}")


def _plot_inter_swath_context(stats_df: pd.DataFrame, output_dir: str, filename: str = 'tg_inter_swath_context'):
    os.makedirs(output_dir, exist_ok=True)
    if 'inter_swath_std_before' not in stats_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes

    # Normalized improvement histogram
    if 'norm_improvement_vs_inter' in stats_df.columns:
        vals = stats_df['norm_improvement_vs_inter'].to_numpy()
        ax0.hist(vals[~np.isnan(vals)], bins=30, color='tab:purple', alpha=0.8)
        ax0.axvline(np.nanmean(vals), color='k', linestyle='--', label=f"mean={np.nanmean(vals):.2f}")
        ax0.legend()
        ax0.set_xlabel('Improvement / inter-swath std (unitless)')
        ax0.set_ylabel('Count')
        ax0.set_title('Normalized improvement')

    # Scatter: scale of inter-swath vs overlap std (before)
    ax1.scatter(stats_df['inter_swath_std_before'], stats_df['overlap_std_before'], s=12, alpha=0.6)
    lim_max = np.nanmax([
        stats_df['inter_swath_std_before'].max(),
        stats_df['overlap_std_before'].max()
    ]) if len(stats_df) else 1.0
    lim = (0, max(1e-6, lim_max))
    ax1.plot(lim, lim, 'k--', linewidth=1)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel('Inter-swath std before (ppm)')
    ax1.set_ylabel('Overlap std before (ppm)')
    ax1.set_title('Scale comparison')

    plt.tight_layout()
    out_png = os.path.join(output_dir, f'{filename}.png')
    out_pdf = os.path.join(output_dir, f'{filename}.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved inter-swath context figure to: {out_png} and {out_pdf}")


def _print_and_attach_significance(stats_df: pd.DataFrame):
    # Paired tests on overlap std
    diff = (stats_df['overlap_std_before'] - stats_df['overlap_std_after']).to_numpy()
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return
    # Paired t-test
    t_stat, t_p = stats.ttest_rel(stats_df['overlap_std_before'], stats_df['overlap_std_after'], nan_policy='omit')
    # Wilcoxon signed-rank (requires finite and non-zero diffs)
    finite_mask = np.isfinite(stats_df['overlap_std_before']) & np.isfinite(stats_df['overlap_std_after'])
    try:
        w_stat, w_p = stats.wilcoxon(
            stats_df.loc[finite_mask, 'overlap_std_before'],
            stats_df.loc[finite_mask, 'overlap_std_after'],
            zero_method='wilcox',
            alternative='greater'  # before > after expected
        )
    except Exception:
        w_stat, w_p = np.nan, np.nan

    # Effect size (paired Cohen's d, aka dz)
    dz = np.nan
    if diff.size > 1:
        dz = float(np.nanmean(diff) / np.nanstd(diff, ddof=1))

    # Bootstrap 95% CI for mean improvement
    rng = np.random.default_rng(42)
    if diff.size >= 2:
        n_boot = 2000
        means = []
        for _ in range(n_boot):
            sample = rng.choice(diff, size=diff.size, replace=True)
            means.append(np.nanmean(sample))
        ci_low, ci_high = np.nanpercentile(means, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    print(
        f"Paired t-test: t={t_stat:.2f}, p={t_p:.2e} | Wilcoxon: W={w_stat}, p={w_p:.2e} | Cohen's dz={dz:.2f} | 95% CI(mean Δ)=[{ci_low:.3f}, {ci_high:.3f}] ppm"
    )


def main():
    # Example default run; adjust max_files for quick tests
    df = compute_tg_overlap_summary(max_files=None, radius_deg=0.1, samples_per_swath=100)
    if not df.empty:
        config = PathConfig()
        out_csv = os.path.join(config.RESULTS_DIR, 'tg_overlap_validation.csv')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved per-scene TG overlap metrics to: {out_csv}")
        # Create composite figure
        fig_out_dir = os.path.join(config.FIGURES_DIR, 'tg_validation')
        _plot_composite_figure(df, fig_out_dir, filename='tg_overlap_composite')
        _plot_inter_swath_context(df, fig_out_dir, filename='tg_inter_swath_context')


if __name__ == '__main__':
    main()


