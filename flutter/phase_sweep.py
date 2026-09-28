#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feb 24 Phase Sweep Visualization Suite
Creates multiple plot styles for the same dataset

Notes:
    load_phase_sweep_data() and draw_violin_panel() are factored out so the
    violin plot (originally "PLOT 4" below) can be imported and drawn
    directly into a figurefirst-tagged axis from a notebook (e.g.
    layout.axes['violin']), using the exact same data-loading and plotting
    logic as the standalone PNG this script produces. The three other plot
    styles (polar mean+SEM, polar all-points, bar chart) are left as
    standalone-only for now — ask if you want any of those extracted too.
"""

import json
import numpy as np
import figurefirst as fifi
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from pathlib import Path

from .config import PHASE_SWEEP_DIR, OUTPUT_DIR


def combine_phase_data(file_list):
    """Combine multiple JSON files for same phase"""
    all_off, all_on = [], []
    for fpath in file_list:
        with open(fpath) as f:
            d = json.load(f)
        for pair in d['pairs']:
            all_off.append(pair['off']['avg_magnitude'])
            all_on.append(pair['on']['avg_magnitude'])

    off, on = np.array(all_off), np.array(all_on)
    reductions = (off - on) / off * 100
    t_stat, p_value = stats.ttest_rel(off, on)

    return {
        'n': len(off),
        'off_mean': np.mean(off),
        'on_mean': np.mean(on),
        'mean_red': np.mean(reductions),
        'std_red': np.std(reductions),
        'sem_red': np.std(reductions) / np.sqrt(len(reductions)),
        'p': p_value,
        't': t_stat,
        'sig': p_value < 0.05,
        'reductions': reductions,
    }


PHASE_FILES = {
    0: ['flutter_on_off_phase000_20260224_114426.json',
        'flutter_on_off_phase000_20260315_144223.json',
        'flutter_on_off_phase000_20260327_132434.json'],
    30: ['flutter_on_off_phase030_20260224_115330.json',
         'flutter_on_off_phase030_20260327_133338.json',
         'flutter_on_off_phase030_20260224_135152.json'],
    60: ['flutter_on_off_phase060_20260224_120233.json',
         'flutter_on_off_phase060_20260327_134241.json',
         'flutter_on_off_phase060_20260224_135755.json'],
    90: ['flutter_on_off_phase090_20260224_121137.json',
         'flutter_on_off_phase090_20260327_135145.json',
         'flutter_on_off_phase090_20260224_140358.json'],
    120: ['flutter_on_off_phase120_20260224_122041.json',
          'flutter_on_off_phase120_20260327_140049.json',
          'flutter_on_off_phase120_20260224_141001.json'],
    150: ['flutter_on_off_phase150_20260224_122944.json',
          'flutter_on_off_phase150_20260327_140952.json',
          'flutter_on_off_phase150_20260315_144826.json'],
    180: ['flutter_on_off_phase180_20260224_123848.json',
          'flutter_on_off_phase180_20260315_145430.json',
          'flutter_on_off_phase180_20260327_141856.json'],
    210: ['flutter_on_off_phase210_20260224_124752.json',
          'flutter_on_off_phase210_20260327_142800.json',
          'flutter_on_off_phase210_20260315_150033.json'],
    240: ['flutter_on_off_phase240_20260224_125655.json',
          'flutter_on_off_phase240_20260327_143704.json',
          'flutter_on_off_phase240_20260315_150636.json'],
    270: ['flutter_on_off_phase270_20260224_130559.json',
          'flutter_on_off_phase270_20260327_144607.json',
          'flutter_on_off_phase270_20260315_151239.json'],
    300: ['flutter_on_off_phase300_20260224_131503.json',
          'flutter_on_off_phase300_20260327_145511.json',
          'flutter_on_off_phase300_20260315_151843.json'],
    330: ['flutter_on_off_phase330_20260224_132406.json',
          'flutter_on_off_phase330_20260327_150415.json',
          'flutter_on_off_phase330_20260315_152446.json'],
}


def load_phase_sweep_data(base_path=None):
    """
    Loads and combines all phase-sweep JSON files, returning everything
    needed to call draw_violin_panel() (or any of the other plot styles
    below) without re-running the full script's plotting/save logic.

    base_path: directory containing the phase JSON files (defaults to
    config.PHASE_SWEEP_DIR).
    """
    if base_path is None:
        base_path = PHASE_SWEEP_DIR
    else:
        base_path = Path(base_path)

    phase_files = {p: [base_path / fname for fname in fnames]
                   for p, fnames in PHASE_FILES.items()}

    results = {p: combine_phase_data(files) for p, files in phase_files.items()}

    phases = np.array(sorted(results.keys()))
    reductions = np.array([results[p]['mean_red'] for p in phases])
    stds = np.array([results[p]['std_red'] for p in phases])
    sems = np.array([results[p]['sem_red'] for p in phases])
    ns = np.array([results[p]['n'] for p in phases])
    mean_red = np.mean(reductions)

    return {
        'results': results,
        'phases': phases,
        'reductions': reductions,
        'stds': stds,
        'sems': sems,
        'ns': ns,
        'mean_red': mean_red,
    }


# ---------------------------------------------------------------------------
# Reusable violin panel (originally "PLOT 4" in this script)
# ---------------------------------------------------------------------------
def draw_violin_panel(ax, phases, results, reductions, ns, mean_red, scale=1.0,
                      show_pct_labels=True, show_legend=True, show_n_labels=True):
    """
    Draws the phase-dependent violin plot (distribution of magnitude
    reduction at each tested phase offset, colored by phase, with n and
    mean-reduction labels) into whatever matplotlib Axes you give it.

    This is the reusable core of the standalone violin PNG this script
    produces — used both for that full-size figure and for drawing the
    identical panel directly into a figurefirst-tagged axis from a
    notebook (e.g. layout.axes['violin']).

    scale: multiplier applied to all fontsizes, marker sizes, and line
    widths. Defaults to 1.0 (unchanged, matches the original standalone
    sizing built for a 16"-wide figure). Use ~0.4-0.5 for a small
    figurefirst panel.

    show_pct_labels: if False, hides the per-phase mean-reduction "X.X%"
    text above each violin. Defaults to True.

    show_legend: if False, omits the "Grand mean: X.X%" legend. Defaults
    to True.

    show_n_labels: if False, hides the "n=X" sample-count label below
    each violin. Defaults to True.
    """
    data_groups = [results[p]['reductions'] for p in phases]
    positions = np.arange(len(phases))

    parts = ax.violinplot(data_groups, positions=positions, widths=0.7,
                           showmeans=True, showmedians=True, showextrema=True)

    # Color violins using the hsv colormap (matches the polar plots elsewhere
    # in this script, so a reader sees the same phase->color mapping across
    # every figure that uses this dataset)
    for i, pc in enumerate(parts['bodies']):
        color = plt.cm.hsv(phases[i] / 360)
        pc.set_facecolor(color)
        pc.set_alpha(0.8)

    parts['cmedians'].set_color('ghostwhite')
    parts['cmedians'].set_linewidth(2.5 * scale)
    parts['cmedians'].set_linestyle('dotted')
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(3 * scale)
    parts['cmeans'].set_linestyle('-')

    # Individual points with jitter
    for i, phase in enumerate(phases):
        individual_reds = results[phase]['reductions']
        np.random.seed(int(phase))
        jitter = np.random.normal(0, 0.08, len(individual_reds))
        ax.scatter(i + jitter, individual_reds, s=40 * scale, alpha=0.5,
                   color='black', edgecolors='black', linewidth=0.5 * scale, zorder=5)

    # n and mean-reduction labels
    for i, (p, n) in enumerate(zip(phases, ns)):
        if show_n_labels:
            ax.text(i, 4.5, f'n={n}', ha='center', fontsize=6,
                    style='italic', fontweight='bold')

        mean_reduction = reductions[i]
        if show_pct_labels:
            y_pos = max(results[p]['reductions']) + 0.5
            ax.text(i, y_pos, f'{mean_reduction:.1f}%',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    color='darkblue')

    ax.axhline(mean_red, color='darkred', linestyle='--', linewidth=2 * scale,
               alpha=0.6, label=(f'Grand mean: {mean_red:.1f}%' if show_legend else None))

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{p}\u00b0' for p in phases], fontsize=6, fontweight='bold')
    ax.set_xlabel('Phase Offset (degrees)', fontsize=8, labelpad=2)
    ax.set_ylabel('Magnitude Reduction (%)', fontsize=8, labelpad=2)
    ax.set_title('Distribution of Magnitude Reduction\n'
                 'Violins colored by phase | White dotted = median | Black solid = mean',
                 fontsize=8, fontweight='bold')
    if show_legend:
        ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([4, max([max(results[p]['reductions']) for p in phases]) + 3])
    ax.set_xlim([-0.5, len(phases) - 0.5])
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)


def main():
    data = load_phase_sweep_data()
    results     = data['results']
    phases      = data['phases']
    reductions  = data['reductions']
    stds        = data['stds']
    sems        = data['sems']
    ns          = data['ns']
    mean_red    = data['mean_red']

    print("Creating visualizations...")

    # ========================================================================
    # PLOT 1: POLAR PLOT - MEAN WITH SEM BAND
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    phases_rad = np.deg2rad(phases)
    phases_closed = np.append(phases_rad, phases_rad[0])
    red_closed = np.append(reductions, reductions[0])
    sem_closed = np.append(sems, sems[0])

    ax1.plot(phases_closed, red_closed, 'o-', linewidth=3, markersize=10,
             color='darkblue', label='Mean reduction', zorder=5)
    ax1.fill_between(phases_closed, red_closed - sem_closed, red_closed + sem_closed,
                     alpha=0.3, color='blue', label='\u00b1SEM')

    best_idx = np.argmax(reductions)
    ax1.plot(phases_rad[best_idx], reductions[best_idx], 'g*', markersize=30,
             label=f'Peak: {phases[best_idx]:.0f}\u00b0 ({reductions[best_idx]:.1f}%)', zorder=10)

    sig_mask = np.array([results[p]['sig'] for p in phases])
    sig_phases = phases_rad[sig_mask]
    sig_reds = reductions[sig_mask]
    ax1.scatter(sig_phases, sig_reds, s=200, facecolors='none', edgecolors='green',
                linewidths=3, label='p < 0.05', zorder=6)

    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('Phase-Locked Flutter Control: Mean Reduction by Phase\n'
                  'Shaded region = \u00b1SEM',
                  fontsize=8, fontweight='bold', pad=25)
    ax1.set_ylabel('Magnitude Reduction (%)', fontsize=8, fontweight='bold', labelpad=30)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=6)
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim([0, max(reductions) + 5])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feb24_polar_mean_with_sem.png', dpi=300, bbox_inches='tight')
    print("\u2713 Saved: feb24_polar_mean_with_sem.png")
    plt.close()

    # ========================================================================
    # PLOT 2: POLAR PLOT - ALL INDIVIDUAL POINTS WITH 95% CI
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for i, phase in enumerate(phases):
        phase_rad = phases_rad[i]
        individual_reds = results[phase]['reductions']

        np.random.seed(int(phase))
        jitter = np.random.normal(0, 0.03, len(individual_reds))
        phase_jittered = phase_rad + jitter

        ax2.scatter(phase_jittered, individual_reds, s=80, alpha=0.6,
                    color=plt.cm.hsv(phase / 360), edgecolors='black', linewidth=0.5, zorder=4)

    ax2.plot(phases_closed, red_closed, '-', linewidth=2.5,
             color='black', label='Mean', zorder=5, alpha=0.7)

    ci_95 = 1.96 * sems
    for i, phase in enumerate(phases):
        phase_rad = phases_rad[i]
        mean = reductions[i]
        ci = ci_95[i]

        ax2.plot([phase_rad, phase_rad], [mean - ci, mean + ci],
                 color='red', linewidth=3, alpha=0.7, zorder=6)
        ax2.plot(phase_rad, mean, 'ko', markersize=8, zorder=7)

    ax2.plot([], [], color='red', linewidth=3, label='95% CI')

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Phase-Locked Flutter Control: All Data Points\n'
                  'Colored by phase | Black bars = 95% CI',
                  fontsize=8, fontweight='bold', pad=25)
    ax2.set_ylabel('Magnitude Reduction (%)', fontsize=8, fontweight='bold', labelpad=30)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=6)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feb24_polar_all_points_with_ci.png', dpi=300, bbox_inches='tight')
    print("\u2713 Saved: feb24_polar_all_points_with_ci.png")
    plt.close()

    # ========================================================================
    # PLOT 3: BAR CHART WITH ERROR BARS
    # ========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 7))

    bar_colors = [plt.cm.hsv(p / 360) for p in phases]
    ax3.bar(range(len(phases)), reductions, yerr=sems, capsize=5,
            color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    for i, (p, n, sig) in enumerate(zip(phases, ns, sig_mask)):
        marker = '*' if sig else ''
        ax3.text(i, reductions[i] + sems[i] + 1, f'{reductions[i]:.1f}%{marker}',
                 ha='center', fontsize=6, fontweight='bold')
        ax3.text(i, -1.5, f'n={n}', ha='center', fontsize=6, style='italic')

    ax3.axhline(mean_red, color='red', linestyle='--', linewidth=2.5,
                alpha=0.7, label=f'Mean: {mean_red:.1f}%')
    ax3.set_xticks(range(len(phases)))
    ax3.set_xticklabels([f'{p}\u00b0' for p in phases], fontsize=6)
    ax3.set_xlabel('Phase Offset (degrees)', fontsize=8, fontweight='bold')
    ax3.set_ylabel('Magnitude Reduction (%)', fontsize=8, fontweight='bold')
    ax3.set_title('Mean Magnitude Reduction by Phase Offset\n'
                  '* = p < 0.05 vs zero',
                  fontsize=8, fontweight='bold')
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, max(reductions) + 8])
    ax3.set_xlim([-0.7, len(phases) - 0.3])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feb24_bar_chart_with_error_bars.png', dpi=300, bbox_inches='tight')
    print("\u2713 Saved: feb24_bar_chart_with_error_bars.png")
    plt.close()

    # ========================================================================
    # PLOT 4: VIOLIN PLOTS (reusable)
    # ========================================================================
    fig4, ax4 = plt.subplots(figsize=(16, 8))
    draw_violin_panel(ax4, phases, results, reductions, ns, mean_red)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feb24_violin_plots_with_cardioid.png', dpi=300, bbox_inches='tight')
    print("\u2713 Saved: feb24_violin_plots_with_cardioid.png")
    plt.close()

    # ========================================================================
    # ONE-WAY ANOVA
    # ========================================================================
    phase_groups = [results[p]['reductions'] for p in phases]
    f_stat, p_anova = f_oneway(*phase_groups)

    print("\n" + "=" * 70)
    print("ONE-WAY ANOVA: Effect of Phase Offset on Control Effectiveness")
    print("=" * 70)
    print("Null hypothesis: All phase means are equal")
    print("Alternative: At least one phase mean differs\n")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_anova:.6f}")

    if p_anova < 0.05:
        print("\u2192 SIGNIFICANT (p < 0.05)")
        print("   Phase offset significantly affects control effectiveness.")
        print(f"   Range = {max(reductions) - min(reductions):.1f}pp")
    else:
        print("\u2192 NOT SIGNIFICANT (p \u2265 0.05)")
        print("   No significant difference between phase means.")

    print("=" * 70)

    if p_anova < 0.05:
        print("\nPost-hoc note: For detailed pairwise comparisons, use Tukey HSD.")
        print("(Most differences likely < 5 pp based on visual inspection)")

        best_phase = phases[best_idx]
        best_stats = results[best_phase]

        print(f"\nDetailed statistics for best phase ({best_phase}\u00b0):")
        print(f"  Mean reduction: {best_stats['mean_red']:.1f}%")
        print(f"  Std Dev: {best_stats['std_red']:.1f}%")
        print(f"  SEM: {best_stats['sem_red']:.1f}%")
        print(f"  95% CI: [{best_stats['mean_red'] - 1.96*best_stats['sem_red']:.1f}%, "
              f"{best_stats['mean_red'] + 1.96*best_stats['sem_red']:.1f}%]")
        print(f"  n: {best_stats['n']}")

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print("\nFiles created:")
    print("  1. feb24_polar_mean_with_sem.png")
    print("  2. feb24_polar_all_points_with_ci.png")
    print("  3. feb24_bar_chart_with_error_bars.png")
    print("  4. feb24_violin_plots_with_cardioid.png")
    print("\nDataset summary:")
    print(f"  Phases tested: {len(phases)}")
    print(f"  Total measurements: {sum(ns)}")
    print(f"  Mean reduction: {mean_red:.1f} \u00b1 {np.std(reductions):.1f}%")
    print(f"  Best phase: {phases[best_idx]}\u00b0 ({reductions[best_idx]:.1f}%)")
    print(f"  Significant phases: {sum(sig_mask)}/{len(phases)}")
    print("=" * 60)


if __name__ == '__main__':
    main()