#!/usr/bin/env python3
"""
trial_order.py  (was plot_trial_order.py)
=========================================
Shows the ACTUAL chronological order conditions were run in, extracted
directly from bag filename timestamps (CONDITION_sampleNN_YYYYMMDD_HHMMSS.bag).

This answers the methodological question "were conditions interleaved/
randomized across a session, or run in blocks?" — which matters because
blocked ordering confounds condition with time-of-day effects (wing
warm-up, actuator drift, fatigue, etc.), while interleaved/randomized
ordering does not.

Produces a single CONTINUOUS linear timeline (not wrapped into fixed-width
rows) — every trial across every recording day, in true chronological
order, left to right in one strip. Vertical dashed lines mark day
boundaries so multi-day sessions are still readable.

Also produces a printed chronological trial list per day (for a methods-
section appendix / supplementary table).

Notes:
    draw_trial_order_panel() is factored out so it can be imported and
    drawn directly into a figurefirst-tagged axis from a notebook, using
    the same logic as the standalone PNG/PDF this script produces.

Usage:
    python -m flutter.trial_order
    python -m flutter.trial_order --bag-dir /path/to/bags
"""
import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import figurefirst as fifi
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .config import OL_RDM_DIR, OUTPUT_DIR

CONDITIONS = ['CLOSED_LOOP', 'OPEN_LOOP_54', 'OPEN_LOOP_55', 'POISSON', 'UNIFORM_JITTER', 'PHASE_RANDOMIZED', 'REPLAY']

CONDITION_LABELS = {
    'CLOSED_LOOP':      'Closed Loop',
    'OPEN_LOOP_54':     'Open Loop 5.4 Hz',
    'OPEN_LOOP_55':     'Open Loop 5.5 Hz',
    'POISSON':          'Poisson',
    'UNIFORM_JITTER':   'Uniform Jitter',
    'PHASE_RANDOMIZED': 'Polarity Randomized',
    'REPLAY':           'Replay',
}

CONDITION_COLORS = {
    'CLOSED_LOOP':      '#4CAF50',
    'OPEN_LOOP_54':     '#2196F3',
    'OPEN_LOOP_55':     '#2196F3',
    'POISSON':          '#FF9800',
    'UNIFORM_JITTER':   '#9C27B0',
    'PHASE_RANDOMIZED': '#F44336',
    'REPLAY':           '#795548',
}

TIMESTAMP_PATTERN = re.compile(r'_(\d{8})_(\d{6})')


def collect_trial_order(bag_dir):
    """
    Scan for every condition's bag files, parse the real recording
    timestamp out of each filename, and return one flat chronologically
    sorted list of (datetime, condition, filename).
    """
    trials = []
    for cond in CONDITIONS:
        pattern = os.path.join(bag_dir, f'{cond}_sample*.bag')
        for bag_path in glob.glob(pattern):
            stem = Path(bag_path).stem
            m = TIMESTAMP_PATTERN.search(stem)
            if not m:
                print(f'  [WARN] no timestamp found in {Path(bag_path).name}, skipping')
                continue
            dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            trials.append((dt, cond, Path(bag_path).name))

    trials.sort(key=lambda x: x[0])
    return trials


def draw_trial_order_panel(ax, trials, scale=1.0, show_legend=True):
    """
    Draws the full trial sequence as ONE continuous horizontal strip —
    every trial, in true chronological order across all days, left to
    right — into whatever matplotlib Axes you give it. Each trial is an
    adjacent colored block (no gaps), so a "blocked" run of the same
    condition (a design problem) reads as a solid stretch of one color,
    while a properly interleaved/randomized sequence reads as a scattered
    mix with no long same-color runs.

    Vertical dashed lines mark where one recording day ends and the next
    begins, with the date labeled above each day's segment.

    This is the reusable core of the standalone PNG/PDF this script
    produces — used both for that figure and for drawing the identical
    panel directly into a figurefirst-tagged axis from a notebook.

    scale: multiplier applied to all fontsizes and line widths. Defaults
    to 1.0. Use ~0.5-0.7 for a small figurefirst panel.

    show_legend: if False, omits the condition-color legend below the
    strip. Defaults to True.
    """
    if not trials:
        ax.text(0.5, 0.5, 'No trial data', ha='center', va='center', transform=ax.transAxes)
        return

    n = len(trials)
    xs = np.arange(n)
    colors = [CONDITION_COLORS[cond] for _, cond, _ in trials]

    ax.bar(xs, height=1.0, width=1.0, bottom=0, color=colors, align='edge',
           edgecolor='white', linewidth=0.8 * scale, zorder=3)

    # Day boundaries
    day_start_idx = {}
    for i, (dt, cond, name) in enumerate(trials):
        day = dt.date()
        if day not in day_start_idx:
            day_start_idx[day] = i

    sorted_days = sorted(day_start_idx.keys())
    for day in sorted_days[1:]:  # skip first, it's just the left edge
        idx = day_start_idx[day]
        ax.axvline(idx, color='black', lw=1.0 * scale, ls='--', alpha=0.6, zorder=4)

    for day in sorted_days:
        idx = day_start_idx[day]
        ax.text(idx + 0.3, 1.08, day.strftime('%-m/%-d/%y'),
                ha='left', va='bottom', fontsize=6, fontweight='bold',
                rotation=0, transform=ax.get_xaxis_transform())

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=7 * scale)
    ax.set_title('Randomized Trial Order',
                 fontsize=8, fontweight='bold')
    fifi.mpl_functions.adjust_spines(ax, ['bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'bottom': 5},
                                     linewidth=0.5 * scale)

    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor=CONDITION_COLORS[c], edgecolor='white', label=CONDITION_LABELS[c])
            for c in CONDITIONS if any(cond == c for _, cond, _ in trials)
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc='upper center',
                 bbox_to_anchor=(0.5, -0.22), ncol=min(len(legend_elements), 7),
                 frameon=True)


def plot_trial_order(trials, output_dir='.'):
    """Standalone single-panel PNG/PDF version of draw_trial_order_panel()."""
    if not trials:
        print('[ERROR] No trials found — check --bag-dir path and filename format.')
        return

    fig, ax = plt.subplots(figsize=(14, 3.5))
    draw_trial_order_panel(ax, trials)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = f'{output_dir}/trial_order_timeline.{ext}'
        plt.savefig(p, dpi=200, bbox_inches='tight')
    print('[SAVED] trial_order_timeline.png / .pdf')
    plt.close(fig)


def print_trial_order(trials):
    by_day = defaultdict(list)
    for dt, cond, name in trials:
        by_day[dt.date()].append((dt, cond, name))

    print('\n' + '=' * 70)
    print('ACTUAL CHRONOLOGICAL TRIAL ORDER (from filename timestamps)')
    print('=' * 70)
    for day in sorted(by_day.keys()):
        print(f'\n{day.strftime("%A %m/%d/%Y")}:')
        for i, (dt, cond, name) in enumerate(by_day[day], 1):
            print(f'  {i:>2}. {dt.strftime("%H:%M:%S")}  {CONDITION_LABELS[cond]:<20} ({name})')
    print('\n' + '=' * 70)


def main():
    parser = argparse.ArgumentParser(description='Visualize actual trial run order from bag timestamps')
    parser.add_argument('--bag-dir', type=str, default=str(OL_RDM_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    print(f'Scanning: {args.bag_dir}\n')
    trials = collect_trial_order(args.bag_dir)
    print(f'Found {len(trials)} trials across '
          f'{len(set(dt.date() for dt, _, _ in trials))} recording day(s)')

    print_trial_order(trials)
    plot_trial_order(trials, args.output_dir)


if __name__ == '__main__':
    main()