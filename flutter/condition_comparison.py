#!/usr/bin/env python3
"""
condition_comparison.py  (was plot_CL_RDM.py)
=============================================
Compares all conditions from per-sample JSON files in the OL_RDM folder.

Usage:
  python -m flutter.condition_comparison
"""

import json
import numpy as np
import figurefirst as fifi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from .config import OL_RDM_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Colors / labels
# ---------------------------------------------------------------------------
CONDITION_COLORS = {
    'CLOSED_LOOP':      '#2e8b57',
    'OPEN_LOOP_55':     '#2196F3',
    'POISSON':          '#FF9800',
    'UNIFORM_JITTER':   '#9C27B0',
    'PHASE_RANDOMIZED': '#F44336',
    'REPLAY':           '#795548',
}

CONDITION_LABELS = {
    'CLOSED_LOOP':      'Closed Loop\n(Freq Matched)',
    'OPEN_LOOP_55':     'Open Loop\n5.5 Hz',
    'POISSON':          'Poisson\nPulses',
    'UNIFORM_JITTER':   'Uniform\nJitter',
    'PHASE_RANDOMIZED': 'Polarity\nRandomized',
    'REPLAY':           'Closed Loop\nReplay',
}

NEW_ORDER = ['CLOSED_LOOP', 'OPEN_LOOP_55', 'POISSON', 'UNIFORM_JITTER', 'PHASE_RANDOMIZED', 'REPLAY']

PHASE_COHERENCE = {
    'PHASE_RANDOMIZED': 0.00,
    'POISSON':          0.33,
    'UNIFORM_JITTER':   0.50,
    'OPEN_LOOP_55':     0.75,
    'CLOSED_LOOP':      1.00,
}

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_all_conditions(ol_rdm_dir):
    folder = Path(ol_rdm_dir)
    condition_data = {}
    print(f"\nLoading all conditions from {folder}")

    for jf in sorted(folder.glob('*.json')):
        try:
            with open(jf) as f:
                data = json.load(f)
            meta      = data.get('meta', {})
            reduction = data.get('reduction', {})
            condition = meta.get('condition', None)
            pct       = reduction.get('magnitude_reduction_pct', None)
            if condition is None or pct is None:
                continue
            if condition not in condition_data:
                condition_data[condition] = []
            condition_data[condition].append(pct)
            print(f"  {jf.name}: {condition} → {pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] {jf.name}: {e}")

    for cond, vals in condition_data.items():
        print(f"  Total {cond}: {len(vals)} samples")

    return condition_data

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def draw_ci_box(ax, x_center, values, color, box_width=0.42):
    mean  = np.mean(values)
    n     = len(values)
    se    = stats.sem(values)
    ci    = se * stats.t.ppf(0.975, df=max(n - 1, 1))
    half  = box_width / 2.0
    rect  = mpatches.FancyBboxPatch(
        (x_center - half, mean - ci), box_width, 2 * ci,
        boxstyle="square,pad=0",
        linewidth=1.8, linestyle="--",
        edgecolor=color, facecolor="none", zorder=4
    )
    ax.add_patch(rect)
    ax.plot([x_center - half, x_center + half], [mean, mean],
            color=color, linewidth=2.5, zorder=5)
    return mean, mean - ci, mean + ci


def _wrap_two_lines(label):
    """
    Wraps a label onto at most two lines, breaking at the space closest
    to the middle. Also collapses any pre-existing '\n' in the label
    first, so labels that already have a manual line break (like
    CONDITION_LABELS here) get re-wrapped consistently.
    """
    label = label.replace('\n', ' ')
    if ' ' not in label:
        return label
    words = label.split(' ')
    best_split, best_diff = 1, float('inf')
    for i in range(1, len(words)):
        left = ' '.join(words[:i])
        right = ' '.join(words[i:])
        diff = abs(len(left) - len(right))
        if diff < best_diff:
            best_diff = diff
            best_split = i
    return ' '.join(words[:best_split]) + '\n' + ' '.join(words[best_split:])


def draw_flutter_pct_reduction_panel(ax, all_conditions, cond_keys=None, scale=1.0):
    """
    Draws the "All Conditions vs Closed Loop" panel (jittered individual
    samples + dashed 95% CI box + mean line + significance vs Closed
    Loop) into whatever matplotlib Axes you give it.
    """
    if cond_keys is None:
        cond_keys = [c for c in NEW_ORDER if c in all_conditions]

    n_cond = len(cond_keys)
    all_vals = np.concatenate([all_conditions[c] for c in cond_keys])
    y_min = min(all_vals) - 10
    y_max = max(all_vals) + 8
    y_range = y_max - y_min

    np.random.seed(42)

    for i, cond in enumerate(cond_keys):
        values = all_conditions[cond]
        color = CONDITION_COLORS.get(cond, '#666666')

        jitter = np.random.normal(0, 0.07, size=len(values))
        ax.scatter(i + jitter, values,
           color=color, alpha=0.85, s=25 * scale, zorder=3,
           edgecolors=color, linewidth=0.5 * scale)

        mean, ci_lo, ci_hi = draw_ci_box(ax, i, values, color, box_width=0.42)

        sig_label = ''
        if 'CLOSED_LOOP' in all_conditions and cond != 'CLOSED_LOOP':
            ref = all_conditions['CLOSED_LOOP']
            _, p = stats.ttest_ind(ref, values)
            sig_label = ('***' if p < 0.001 else
                         '**'  if p < 0.01  else
                         '*'   if p < 0.05  else 'ns')

        ax.annotate(
            f'{mean:.1f}%',
            xy=(i + 0.22, mean),
            xytext=(i + 0.32, mean),
            fontsize=6, fontweight='bold', color=color,
            va='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0 * scale)
        )

        if sig_label:
            # Offset scaled to the panel's OVERALL y-range rather than a
            # fixed "+0.5" constant, so spacing stays proportional
            # whether the data spans a small or large percentage range.
            sig_offset = y_range * 0.02
            ax.text(i + 0.38, mean + (ci_hi - ci_lo) * 0.15 + sig_offset,
                    sig_label, ha='center', va='bottom',
                    fontsize=6, color='#444444', fontweight='bold',
                    clip_on=True)

        # n= label: anchored to a fixed fraction of the AXES height (not
        # a data-derived y position), so it always sits just above the
        # bottom of the plot area and can never overlap the x-axis line
        # or tick labels regardless of the data's actual y-range.
        ax.text(i, 0.03, f'n={len(values)}',
                ha='center', va='bottom', fontsize=6, color='#555555',
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8 * scale, alpha=0.3)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels([_wrap_two_lines(CONDITION_LABELS.get(c, c)) for c in cond_keys],
                    fontsize=6, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=6 * scale)
    ax.set_ylabel('Flutter Magnitude\nReduction (%)', fontsize=8, labelpad=2)
    ax.set_title('All Conditions vs Closed Loop', fontsize=8, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    ax.set_xlim(-0.6, n_cond - 0.4)
    ax.set_ylim(y_min, y_max)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plt.ion()

    raw = load_all_conditions(OL_RDM_DIR)

    # Build ordered dict — only include conditions present in data
    all_conditions = {}
    for cond in NEW_ORDER:
        if cond in raw and len(raw[cond]) > 0:
            all_conditions[cond] = np.array(raw[cond])

    if not all_conditions:
        print("\n[ERROR] No data loaded.")
        return

    cond_keys = list(all_conditions.keys())
    n_cond    = len(cond_keys)
    all_vals  = np.concatenate(list(all_conditions.values()))
    y_min     = min(all_vals) - 10
    y_max     = max(all_vals) + 8

    from datetime import datetime
    dated_dir = OUTPUT_DIR / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_CL_RDM')
    dated_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # =========================================================================
    # FIGURE 1: Phase coherence line plot (excludes REPLAY)
    # =========================================================================
    coherence_order = sorted(
        [c for c in cond_keys if c in PHASE_COHERENCE],
        key=lambda c: PHASE_COHERENCE[c]
    )
    coh_x     = [PHASE_COHERENCE[c] for c in coherence_order]
    coh_means = np.array([np.mean(all_conditions[c]) for c in coherence_order])
    coh_sems  = np.array([stats.sem(all_conditions[c]) for c in coherence_order])

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(coh_x, coh_means, '--', color='gray', alpha=0.4, linewidth=1.5, zorder=1)

    for i, cond in enumerate(coherence_order):
        color  = CONDITION_COLORS.get(cond, '#666666')
        marker = 's' if cond == 'CLOSED_LOOP' else 'o'
        ax1.errorbar(
            coh_x[i], coh_means[i], yerr=coh_sems[i],
            fmt=marker, color=color, markersize=14,
            capsize=6, capthick=2.5, elinewidth=2.5,
            linewidth=2.5, zorder=5,
            label=CONDITION_LABELS.get(cond, cond).replace('\n', ' ')
        )

    # Add REPLAY as a separate annotated point if available
    if 'REPLAY' in all_conditions:
        r_mean = np.mean(all_conditions['REPLAY'])
        r_sem  = stats.sem(all_conditions['REPLAY'])
        ax1.errorbar(1.0, r_mean, yerr=r_sem,
                     fmt='D', color=CONDITION_COLORS['REPLAY'], markersize=10,
                     capsize=5, capthick=2.0, elinewidth=2.0, zorder=6,
                     label='Closed Loop Replay')
        ax1.annotate('Replay', xy=(1.0, r_mean), xytext=(1.02, r_mean),
                     fontsize=6, color=CONDITION_COLORS['REPLAY'], va='center')

    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax1.set_xlabel('Phase Coherence\n(0 = random phase,  1 = fully locked)',
                   fontsize=8, labelpad=2)
    ax1.set_ylabel('Flutter Magnitude Reduction (%)', fontsize=8, labelpad=2)
    ax1.set_title('Control Effectiveness vs Phase Coherence\n(Error bars = SEM)',
                  fontsize=8, fontweight='bold')
    ax1.set_xlim(-0.1, 1.15)
    ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=6, framealpha=0.9)
    plt.tight_layout()
    fig1.savefig(dated_dir / 'OL_RDM_phase_coherence.png', dpi=300, bbox_inches='tight')
    fig1.savefig(dated_dir / 'OL_RDM_phase_coherence.pdf', bbox_inches='tight')
    print(f"\n[SAVED] OL_RDM_phase_coherence.png / .pdf")

    # =========================================================================
    # FIGURE 2: Individual points + CI box
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(13, 7))
    np.random.seed(42)

    for i, cond in enumerate(cond_keys):
        values = all_conditions[cond]
        color  = CONDITION_COLORS.get(cond, '#666666')

        jitter = np.random.normal(0, 0.07, size=len(values))
        ax2.scatter(i + jitter, values,
                    color=color, alpha=0.85, s=55, zorder=3,
                    edgecolors=color, linewidth=0.5)

        mean, ci_lo, ci_hi = draw_ci_box(ax2, i, values, color)

        sig_label = ''
        if 'CLOSED_LOOP' in all_conditions and cond != 'CLOSED_LOOP':
            ref = all_conditions['CLOSED_LOOP']
            _, p = stats.ttest_ind(ref, values)
            sig_label = ('***' if p < 0.001 else
                         '**'  if p < 0.01  else
                         '*'   if p < 0.05  else 'ns')

        ax2.annotate(
            f'{mean:.1f}%',
            xy=(i + 0.22, mean),
            xytext=(i + 0.32, mean),
            fontsize=6, fontweight='bold', color=color,
            va='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2)
        )

        if sig_label:
            ax2.text(i + 0.38, mean + (ci_hi - ci_lo) * 0.15 + 0.5,
                     sig_label, ha='center', va='bottom',
                     fontsize=6, color='#444444', fontweight='bold')

        ax2.text(i, y_min + 1, f'n={len(values)}',
                 ha='center', va='top', fontsize=6, color='#555555')

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax2.set_xticks(range(n_cond))
    ax2.set_xticklabels([CONDITION_LABELS.get(c, c) for c in cond_keys], fontsize=6)
    ax2.set_ylabel('Flutter Magnitude Reduction (%)', fontsize=8, labelpad=2)
    ax2.set_xlabel('Control Condition', fontsize=8, labelpad=2)
    ax2.set_title('All Conditions vs Closed Loop\nIndividual samples with mean and 95% CI',
                  fontsize=8, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.25, linestyle=':')
    ax2.set_xlim(-0.6, n_cond - 0.4)
    ax2.set_ylim(y_min, y_max)
    plt.tight_layout()
    fig2.savefig(dated_dir / 'OL_RDM_vs_closed_loop.png', dpi=300, bbox_inches='tight')
    fig2.savefig(dated_dir / 'OL_RDM_vs_closed_loop.pdf', bbox_inches='tight')
    print(f"[SAVED] OL_RDM_vs_closed_loop.png / .pdf")

    # --- Print stats ---
    print("\n" + "="*65)
    print("SUMMARY STATISTICS")
    print("="*65)
    for cond in cond_keys:
        vals = all_conditions[cond]
        n    = len(vals)
        mean = np.mean(vals)
        ci   = stats.sem(vals) * stats.t.ppf(0.975, df=max(n - 1, 1))
        print(f"  {CONDITION_LABELS.get(cond,cond).replace(chr(10),' '):30} "
              f"mean={mean:6.1f}%  95CI±{ci:.2f}%  n={n}")

    if 'CLOSED_LOOP' in all_conditions:
        ref = all_conditions['CLOSED_LOOP']
        print("\n" + "="*65)
        print("STATISTICAL TESTS vs CLOSED_LOOP")
        print("="*65)
        for cond in cond_keys:
            if cond == 'CLOSED_LOOP':
                continue
            vals = all_conditions[cond]
            t_stat, p = stats.ttest_ind(ref, vals)
            n1, n2 = len(ref), len(vals)
            pooled = np.sqrt(((n1-1)*np.var(ref,ddof=1)+(n2-1)*np.var(vals,ddof=1))/(n1+n2-2))
            d = (np.mean(ref) - np.mean(vals)) / pooled
            sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
            lbl = CONDITION_LABELS.get(cond, cond).replace('\n', ' ')
            print(f"  CLOSED_LOOP vs {lbl:25}  t={t_stat:.3f}  p={p:.4f} {sig}  d={d:.2f}")
    print("="*65)

    plt.close('all')


if __name__ == '__main__':
    main()