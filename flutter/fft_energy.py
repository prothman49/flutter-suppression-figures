#!/usr/bin/env python3
"""
fft_energy.py  (was plot_fft_energy.py)
=======================================
Two figures:

Figure 1 — Total spectral energy per condition
  For each bag, compute the FFT over the full 60s uncontrolled and 60s
  controlled segments of /optic_flow_mean_vx. Integrate across all
  frequencies to get total energy, then compare conditions.
  Shows whether control reduces total energy or redistributes it.

  Points are colored by condition (fill) and by recording date (edge ring),
  so you can see whether the effect is consistent across sessions or
  driven by one day's data. Per-date reduction percentages are listed
  alongside the pooled reduction for each condition.

Figure 2 — Mean FFT spectrum by condition
  Overlay mean FFT spectrum (averaged across all bags) for uncontrolled
  vs controlled segments. Shows exactly which frequencies are suppressed
  and whether energy appears elsewhere.

Usage:
    python -m flutter.fft_energy
    python -m flutter.fft_energy --bag-dir /path/to/bags --output-dir /path/to/output

Notes:
    draw_energy_panel() is factored out so it can be imported and drawn
    directly into a figurefirst-tagged axis (e.g. layout.axes['Total_power'])
    from a notebook, using the exact same plotting logic as the standalone
    PNG/PDF output produced by this script.
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import figurefirst as fifi
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import welch
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

from .config import OL_RDM_DIR, OUTPUT_DIR

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_UNCONTROLLED = 60.0
T_CONTROLLED   = 60.0
FLUTTER_FREQ   = 5.4
FLUTTER_BAND   = (4.0, 6.0)

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

CLOSED_LOOP_PATTERN = 'CLOSED_LOOP_sample*.bag'


def _wrap_two_lines(label):
    """
    Wraps a label onto at most two lines, breaking at the space closest
    to the middle of the string — used for x-axis tick labels so
    multi-word condition names (e.g. "Open Loop 5.5 Hz") wrap into a
    readable two-line label instead of either running too wide on one
    line or breaking at every space into three+ short lines.
    """
    if ' ' not in label:
        return label
    words = label.split(' ')
    if len(words) <= 1:
        return label
    best_split = 1
    best_diff = float('inf')
    for i in range(1, len(words)):
        left = ' '.join(words[:i])
        right = ' '.join(words[i:])
        diff = abs(len(left) - len(right))
        if diff < best_diff:
            best_diff = diff
            best_split = i
    return ' '.join(words[:best_split]) + '\n' + ' '.join(words[best_split:])

DATE_EDGE_PALETTE = ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#BCBD22', '#E377C2']

DATE_PATTERN = re.compile(r'_(\d{8})_\d{6}')


def parse_sample_date(path):
    m = DATE_PATTERN.search(Path(path).stem)
    return m.group(1) if m else None


def format_date_label(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        return f'{dt.month}/{dt.day}'
    except ValueError:
        return date_str


def read_optic_flow_segment(bag_path, t_start, t_end):
    times, vals = [], []
    with Reader(bag_path) as reader:
        t0_ns = reader.start_time
        conns = [c for c in reader.connections if c.topic == '/optic_flow_mean_vx']
        if not conns:
            return None, None
        for conn, ts, raw in reader.messages(connections=conns):
            t_sec = (ts - t0_ns) / 1e9
            if t_sec < t_start or t_sec >= t_end:
                continue
            try:
                msg = TYPESTORE.deserialize_ros1(raw, conn.msgtype)
                times.append(t_sec - t_start)
                vals.append(float(msg.data))
            except Exception:
                continue
    if not times:
        return None, None
    return np.array(times), np.array(vals)


def compute_fft(times, signal):
    if len(times) < 10:
        return None, None, None
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    t_uniform = np.arange(times[0], times[-1], dt)
    sig_uniform = np.interp(t_uniform, times, signal)
    sig_uniform -= np.mean(sig_uniform)
    N = len(sig_uniform)
    fft_vals = np.fft.rfft(sig_uniform)
    freqs    = np.fft.rfftfreq(N, d=dt)
    mags     = np.abs(fft_vals) / N
    return freqs, mags, fs


def total_energy(freqs, mags, f_low=None, f_high=None):
    if f_low is not None or f_high is not None:
        mask = np.ones(len(freqs), dtype=bool)
        if f_low is not None:
            mask &= freqs >= f_low
        if f_high is not None:
            mask &= freqs <= f_high
        f = freqs[mask]
        m = mags[mask]
    else:
        f = freqs
        m = mags
    return np.trapezoid(m**2, f)


def compute_psd_welch(times, signal, nperseg_seconds=10.0):
    """
    Computes a proper Power Spectral Density estimate via Welch's method
    (averaged periodograms over overlapping windowed segments), used ONLY
    for the energy/power SUMMARY panels (Total Power, Flutter-Band Power
    scatter plots) — "energy" from a single raw FFT magnitude is a vague
    quantity; Welch's method gives a statistically robust PSD estimate in
    proper power units, which is what's actually being compared across
    conditions in those panels.

    This is intentionally kept SEPARATE from compute_fft() above, which
    remains unchanged and is still used for the visual spectrum overlay
    curves (the actual FFT line plots) — Welch's method involves windowing
    and averaging that smooths the spectrum differently than a raw single
    FFT, which would alter the visual shape/sharpness of the spectrum
    curves if used there too. Only the summary power NUMBERS switch to
    Welch; the spectrum shape you see in every FFT plot is untouched.

    nperseg_seconds: window length for Welch's method, in seconds. Longer
    windows give better frequency resolution; shorter windows give more
    averaging (a less noisy PSD estimate) at coarser resolution. 10s is a
    reasonable default for the ~60s segments used throughout this project
    (giving roughly 6 windows at 50% overlap).

    Returns (freqs_psd, psd, fs), or (None, None, None) if there's not
    enough data.
    """
    if len(times) < 10:
        return None, None, None
    dt = np.median(np.diff(times))
    fs = 1.0 / dt
    t_uniform = np.arange(times[0], times[-1], dt)
    sig_uniform = np.interp(t_uniform, times, signal)
    sig_uniform = sig_uniform - np.mean(sig_uniform)

    nperseg = min(len(sig_uniform), int(nperseg_seconds * fs))
    if nperseg < 8:
        return None, None, None
    freqs_psd, psd = welch(sig_uniform, fs=fs, nperseg=nperseg)
    return freqs_psd, psd, fs


def total_power_psd(freqs_psd, psd, f_low=None, f_high=None):
    """
    Integrates a Welch PSD estimate over a frequency band to get band
    power, in proper power units — unlike total_energy() above, which
    integrates raw FFT magnitude-squared and doesn't carry the same
    normalization/statistical guarantees a true PSD estimate does.
    """
    if f_low is not None or f_high is not None:
        mask = np.ones(len(freqs_psd), dtype=bool)
        if f_low is not None:
            mask &= freqs_psd >= f_low
        if f_high is not None:
            mask &= freqs_psd <= f_high
        f = freqs_psd[mask]
        p = psd[mask]
    else:
        f = freqs_psd
        p = psd
    return np.trapezoid(p, f)


def process_condition_fft(bag_paths, is_closed_loop=False):
    results = {
        'uncontrolled': {'freqs': [], 'mags': [], 'energy_total': [], 'energy_flutter': [], 'dates': []},
        'controlled':   {'freqs': [], 'mags': [], 'energy_total': [], 'energy_flutter': [], 'dates': []},
    }
    MIN_DURATION = 100.0
    for bag_path in bag_paths:
        name = Path(bag_path).name
        date_str = parse_sample_date(bag_path)
        if date_str is None:
            print(f"    [WARN] {name}: no date found in filename")
        with Reader(bag_path) as r:
            duration = (r.end_time - r.start_time) / 1e9
        if duration < MIN_DURATION:
            print(f"    [SKIP] {name}: duration {duration:.1f}s < {MIN_DURATION}s")
            continue
        print(f"    {name} ({duration:.1f}s)")
        if is_closed_loop:
            split = detect_split_point_fft(bag_path)
            if split is None:
                seg_bounds = [
                    ('uncontrolled', 0.0,           T_UNCONTROLLED),
                    ('controlled',   T_UNCONTROLLED, T_UNCONTROLLED + T_CONTROLLED),
                ]
            else:
                print(f"      Split at {split:.1f}s")
                seg_bounds = [
                    ('uncontrolled', 0.0,   split),
                    ('controlled',   split, split + T_CONTROLLED),
                ]
        else:
            seg_bounds = [
                ('uncontrolled', 0.0,           T_UNCONTROLLED),
                ('controlled',   T_UNCONTROLLED, T_UNCONTROLLED + T_CONTROLLED),
            ]
        for seg_name, t0, t1 in seg_bounds:
            times, signal = read_optic_flow_segment(bag_path, t0, t1)
            if times is None or len(times) < 100:
                print(f"      [SKIP] {seg_name}: insufficient data")
                continue
            freqs, mags, fs = compute_fft(times, signal)
            if freqs is None:
                continue

            # Summary power numbers now use a proper Welch PSD estimate
            # rather than integrating the raw single-FFT magnitude squared
            # — "energy" from a single FFT is a vague, less statistically
            # grounded quantity than a PSD-based power estimate. The
            # visual spectrum curves above (freqs, mags) are untouched.
            freqs_psd, psd, _ = compute_psd_welch(times, signal)
            if freqs_psd is not None:
                e_total   = total_power_psd(freqs_psd, psd)
                e_flutter = total_power_psd(freqs_psd, psd, *FLUTTER_BAND)
            else:
                e_total = e_flutter = np.nan

            results[seg_name]['freqs'].append(freqs)
            results[seg_name]['mags'].append(mags)
            results[seg_name]['energy_total'].append(e_total)
            results[seg_name]['energy_flutter'].append(e_flutter)
            results[seg_name]['dates'].append(date_str)
            print(f"      {seg_name}: Power_total={e_total:.4g}  Power_flutter={e_flutter:.4g}  fs={fs:.1f}Hz")
    return results


def detect_split_point_fft(bag_path, bin_size=5.0, min_bins=3):
    times, v0 = [], []
    with Reader(bag_path) as reader:
        t0_ns = reader.start_time
        conns = [c for c in reader.connections if c.topic == '/piezo_voltages']
        if not conns:
            return None
        for conn, ts, raw in reader.messages(connections=conns):
            try:
                msg = TYPESTORE.deserialize_ros1(raw, conn.msgtype)
                times.append((ts - t0_ns) / 1e9)
                v0.append(float(msg.data[0]))
            except Exception:
                continue
    if not times:
        return None
    times = np.array(times)
    v0    = np.array(v0)
    t_max = times[-1]
    bins  = np.arange(0, t_max, bin_size)
    active = np.array([
        np.any(v0[(times >= b) & (times < b + bin_size)] > 0.1)
        if np.any((times >= b) & (times < b + bin_size)) else False
        for b in bins
    ])
    for i in range(len(active) - min_bins + 1):
        if all(active[i:i + min_bins]):
            return float(bins[i])
    return None


def add_panel_label(target, letter, x=0.03, y=0.99, fontsize=11):
    """
    Draws a bold panel-reference letter ('A', 'B', 'a', 'b', ...) near the
    top-left of a panel, for referencing individual panels/groups of
    panels in text (standard multi-panel figure convention).

    target: either a single matplotlib Axes (label sits at that axis's
    own top-left corner, in axes-fraction coordinates) or a Figure (label
    sits at an absolute figure-fraction position — use this when ONE
    label needs to cover a GROUP of several panels, e.g. a whole row of
    icons + FFT panels treated as one lettered unit; you'll need to pick
    x/y by eye for the group's actual top-left corner in figure-fraction
    coordinates, typically found via trial and error or by checking
    where that group's leftmost/topmost axis sits via
    ax.get_position()).

    x, y: position of the label, in axes-fraction (Axes target) or
    figure-fraction (Figure target) coordinates. IMPORTANT: keep these
    within [0, 1] — figurefirst embeds each matplotlib axis as a snippet
    clipped to its own box boundaries at the SVG level, independent of
    matplotlib's own clip_on setting, so anything positioned outside the
    axis's normal [0,1] range (e.g. y > 1 to sit "above" the box) gets
    silently cut off during the embed, no matter what's set here. The
    defaults (0.03, 0.99) sit just inside the top-left corner instead of
    hovering above/left of it — slightly different convention (inside
    rather than floating outside the panel) but the one that's actually
    guaranteed to render inside a figurefirst-embedded panel.

    fontsize: defaults to 11 (a bit larger than the standard 8pt titles/
    6pt labels used elsewhere, since these need to stand out as
    reference markers, not blend in with regular panel text).
    """
    if hasattr(target, 'transAxes'):
        target.text(x, y, letter, transform=target.transAxes,
                    fontsize=fontsize, fontweight='bold', va='top', ha='left',
                    clip_on=False, zorder=100)
    else:
        target.text(x, y, letter, transform=target.transFigure,
                    fontsize=fontsize, fontweight='bold', va='top', ha='left',
                    clip_on=False, zorder=100)


def build_date_color_map(all_results):
    all_dates = set()
    for res in all_results.values():
        for seg in ('uncontrolled', 'controlled'):
            all_dates.update(d for d in res[seg]['dates'] if d is not None)
    sorted_dates = sorted(all_dates)
    if len(sorted_dates) > len(DATE_EDGE_PALETTE):
        print(f"[WARN] {len(sorted_dates)} distinct dates found but only "
              f"{len(DATE_EDGE_PALETTE)} palette colors defined — colors will repeat.")
    return {
        d: DATE_EDGE_PALETTE[i % len(DATE_EDGE_PALETTE)]
        for i, d in enumerate(sorted_dates)
    }


# ---------------------------------------------------------------------------
# Figure 1 — Total spectral energy comparison
# ---------------------------------------------------------------------------
def draw_energy_panel(ax, all_results, energy_key, title, ylabel, scale=1.0):
    """
    Draws one energy-comparison panel (strip plot of total spectral energy
    per condition, uncontrolled vs controlled) into whatever matplotlib Axes
    you give it.

    This is the reusable core of plot_energy_comparison() below — it's used
    both for the standalone two-panel PNG/PDF this script produces, and for
    drawing the identical panel directly into a figurefirst-tagged axis from
    a notebook (e.g. layout.axes['Total_power']).

    scale: multiplier applied to all fontsizes, marker sizes, and line
    widths in this panel. Defaults to 1.0 (unchanged, matches original
    standalone sizing). Use something like 0.5-0.6 when drawing into a
    small figurefirst panel (e.g. a 90mm-wide dissertation figure) so text
    and markers don't overflow the box.

    No arrows connecting uncontrolled -> controlled (removed per feedback —
    they added visual noise without conveying more than the two dots
    already do). Alternating light/white vertical stripes mark condition
    boundaries so groups are easy to tell apart at a glance. The percent
    change and significance stars are anchored to a single consistent
    height near the top of each panel (not floating at each condition's
    own mid-point), so significance is easy to scan across conditions in
    one horizontal sweep rather than hunting at varying heights.
    """
    cond_keys = [c for c in CONDITIONS if c in all_results]
    n_cond    = len(cond_keys)
    np.random.seed(42)

    # --- Alternating condition-boundary stripes (behind everything) ---
    for i in range(n_cond):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='#f0f0f0', zorder=0, lw=0)

    # --- First pass: plot points/means, collect stats for top labels ---
    per_cond_stats = []  # (i, color, pct, sig, panel_max_for_this_cond)
    for i, cond in enumerate(cond_keys):
        color = CONDITION_COLORS[cond]
        res   = all_results[cond]

        unc = np.array(res['uncontrolled'][energy_key])
        ctl = np.array(res['controlled'][energy_key])

        x_unc = i - 0.18
        x_ctl = i + 0.18

        j_unc = np.random.normal(0, 0.04, len(unc))
        ax.scatter(x_unc + j_unc, unc, color='#888', alpha=0.7, s=25 * scale, zorder=3)
        ax.plot([x_unc - 0.1, x_unc + 0.1], [np.mean(unc), np.mean(unc)],
                color='#444', lw=2.0 * scale, zorder=4)

        j_ctl = np.random.normal(0, 0.04, len(ctl))
        ax.scatter(x_ctl + j_ctl, ctl, color=color, alpha=0.7, s=25 * scale, zorder=3)
        ax.plot([x_ctl - 0.1, x_ctl + 0.1], [np.mean(ctl), np.mean(ctl)],
                color=color, lw=2.0 * scale, zorder=4)

        mean_unc = np.mean(unc) if len(unc) else np.nan
        mean_ctl = np.mean(ctl) if len(ctl) else np.nan
        pct = (mean_ctl - mean_unc) / mean_unc * 100 if mean_unc > 0 else np.nan

        sig = None
        if len(unc) > 1 and len(ctl) > 1:
            _, p = stats.ttest_rel(unc, ctl) if len(unc) == len(ctl) else stats.ttest_ind(unc, ctl)
            sig = ('***' if p < 0.001 else '**' if p < 0.01 else
                   '*'   if p < 0.05  else 'ns')

        local_max = max(
            np.max(unc) if len(unc) else -np.inf,
            np.max(ctl) if len(ctl) else -np.inf,
        )
        per_cond_stats.append((i, color, pct, sig, local_max))

    # --- Second pass: place pct% + significance at one shared height ---
    panel_max = max(s[4] for s in per_cond_stats if np.isfinite(s[4]))
    label_y   = panel_max * 1.10

    for i, color, pct, sig, _ in per_cond_stats:
        parts = []
        if np.isfinite(pct):
            parts.append(f'{pct:+.1f}%')
        if sig is not None:
            parts.append(sig)
        if parts:
            ax.text(i, label_y, '  '.join(parts),
                    ha='center', va='bottom', fontsize=6,
                    color=color, fontweight='bold')

    ax.set_xticks(range(n_cond))
    ax.set_xticklabels([_wrap_two_lines(CONDITION_LABELS[c]) for c in cond_keys],
                       fontsize=6, rotation=0, ha='center')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=6 * scale)
    ax.set_ylim(top=label_y * 1.12)
    ax.set_xlim(-0.6, n_cond - 0.4)
    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)
    # Shrink y-tick label fontsize so the full "0.12" (not just "12")
    # fits within the available left margin — the previous overlap
    # wasn't a scientific-notation issue, it was the "0." prefix getting
    # physically clipped by the panel's own left edge because the full
    # text was too wide for the space at the default tick fontsize.
    ax.tick_params(axis='y', labelsize=6 * scale)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)


def plot_energy_comparison(all_results, output_dir):
    """
    Standalone two-panel figure (total energy + flutter-band energy),
    saved as PNG/PDF. Both panels are drawn via draw_energy_panel() so the
    logic stays identical to whatever gets drawn into figurefirst.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={'wspace': 0.35})
    fig.suptitle('Power Spectral Density: Uncontrolled vs Controlled\n'
                 '(integral of FFT magnitude\u00b2 over frequency)',
                 fontsize=8, fontweight='bold')

    draw_energy_panel(axes[0], all_results, 'energy_total',
                       'Total Power (PSD)', 'Power (a.u.)')
    draw_energy_panel(axes[1], all_results, 'energy_flutter',
                       f'Flutter-Band Power ({FLUTTER_BAND[0]}\u2013{FLUTTER_BAND[1]} Hz, PSD)', 'Power (a.u.)')

    legend_elements = [
        mpatches.Patch(color='#888', label='Uncontrolled'),
        mpatches.Patch(color='C0',   label='Controlled (condition color)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6,
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_energy_comparison.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_energy_comparison.pdf / .png')
    plt.close(fig)


def plot_energy_reduction_dumbbell(all_results, output_dir, date_color_map):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={'wspace': 0.35})
    fig.suptitle('Power Reduction by Recording Date\n'
                 'Grey = uncontrolled mean, colored = controlled mean, line = date',
                 fontsize=8, fontweight='bold')

    for ax_idx, (energy_key, title, ylabel) in enumerate([
        ('energy_total',   'Total Power (PSD)',   'Power (a.u.)'),
        ('energy_flutter', f'Flutter-Band Power ({FLUTTER_BAND[0]}\u2013{FLUTTER_BAND[1]} Hz, PSD)', 'Power (a.u.)'),
    ]):
        ax = axes[ax_idx]
        cond_keys = [c for c in CONDITIONS if c in all_results]
        n_cond    = len(cond_keys)

        for i, cond in enumerate(cond_keys):
            color = CONDITION_COLORS[cond]
            res   = all_results[cond]

            unc       = np.array(res['uncontrolled'][energy_key])
            ctl       = np.array(res['controlled'][energy_key])
            unc_dates = res['uncontrolled']['dates']
            ctl_dates = res['controlled']['dates']

            unique_dates = sorted(set(d for d in (unc_dates + ctl_dates) if d is not None))

            n_slots    = len(unique_dates) + 1
            band_width = 0.7
            slot_w     = band_width / n_slots
            slot_x     = lambda k: i - band_width / 2 + slot_w * (k + 0.5)

            for k, d in enumerate(unique_dates):
                u_d = unc[[dd == d for dd in unc_dates]]
                c_d = ctl[[dd == d for dd in ctl_dates]]
                if len(u_d) == 0 or len(c_d) == 0:
                    continue
                mean_u_d = np.mean(u_d)
                mean_c_d = np.mean(c_d)
                x        = slot_x(k)
                dcolor   = date_color_map.get(d, '#333333')

                ax.plot([x, x], [mean_u_d, mean_c_d], color=dcolor, lw=2.2, zorder=3, alpha=0.9)
                ax.scatter([x], [mean_u_d], color='#999', s=55, zorder=4,
                          edgecolors='white', linewidths=0.8)
                ax.scatter([x], [mean_c_d], color=dcolor, s=55, zorder=4,
                          edgecolors='white', linewidths=0.8)

                if mean_u_d > 0:
                    pct_d  = (mean_c_d - mean_u_d) / mean_u_d * 100
                    y_top  = max(mean_u_d, mean_c_d)
                    ax.text(x, y_top, f'{pct_d:+.0f}%',
                            ha='center', va='bottom', fontsize=6,
                            fontweight='bold', color=dcolor,
                            rotation=90 if n_slots > 4 else 0)

                ax.text(x, min(mean_u_d, mean_c_d), format_date_label(d),
                        ha='center', va='top', fontsize=6, color=dcolor,
                        rotation=0)

            if len(unc) and len(ctl):
                mean_unc = np.mean(unc)
                mean_ctl = np.mean(ctl)
                x        = slot_x(len(unique_dates))

                ax.plot([x, x], [mean_unc, mean_ctl], color='black', lw=3.2, zorder=5)
                ax.scatter([x], [mean_unc], color='#999', s=75, zorder=6,
                          edgecolors='black', linewidths=1.2)
                ax.scatter([x], [mean_ctl], color=color, s=75, zorder=6,
                          edgecolors='black', linewidths=1.2)

                if mean_unc > 0:
                    pct_avg = (mean_ctl - mean_unc) / mean_unc * 100
                    y_top   = max(mean_unc, mean_ctl)
                    ax.text(x, y_top, f'{pct_avg:+.0f}%',
                            ha='center', va='bottom', fontsize=6,
                            fontweight='bold', color='black')
                ax.text(x, min(mean_unc, mean_ctl), 'AVG',
                        ha='center', va='top', fontsize=6,
                        fontweight='bold', color='black')

            if i > 0:
                ax.axvline(i - 0.5, color='#ddd', lw=0.6, zorder=0)

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels([CONDITION_LABELS[c].replace(' ', '\n') for c in cond_keys],
                           fontsize=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')
        fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                         tick_length=2.5,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         linewidth=0.5)
        ax.set_xlim(-0.6, n_cond - 0.4)

    legend_elements = [
        mpatches.Patch(color='#999', label='Uncontrolled mean'),
        mpatches.Patch(color='C0',   label='Controlled mean (condition color)'),
        mlines.Line2D([], [], color='black', lw=3.2, label='Pooled average (bold)'),
    ]
    for d, c in sorted(date_color_map.items()):
        legend_elements.append(
            mlines.Line2D([], [], color=c, lw=2.2,
                          label=f'{format_date_label(d)} (thin)'))

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(legend_elements), 6), fontsize=6,
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_energy_reduction_dumbbell.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_energy_reduction_dumbbell.pdf / .png')
    plt.close(fig)


def _half_violin(ax, x_center, data, color, side, width=0.16, alpha=0.55, min_n=3):
    data = np.asarray(data)
    if len(data) < min_n or np.ptp(data) == 0:
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, width * 0.15, size=len(data))
        sign = -1 if side == 'left' else 1
        ax.scatter(x_center + sign * (0.02 + np.abs(jitter)), data,
                  color=color, s=12, alpha=0.8, zorder=4, linewidths=0)
        return
    kde     = gaussian_kde(data)
    y_grid  = np.linspace(data.min(), data.max(), 200)
    density = kde(y_grid)
    density = density / density.max() * width
    sign    = -1 if side == 'left' else 1
    x_curve = x_center + sign * density
    ax.fill_betweenx(y_grid, x_center, x_curve, color=color, alpha=alpha,
                     zorder=3, linewidth=0)
    ax.plot(x_curve, y_grid, color=color, lw=0.8, alpha=0.9, zorder=4)


def plot_energy_raincloud_by_date(all_results, output_dir, date_color_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={'wspace': 0.35})
    fig.suptitle('Spectral Energy Distributions by Recording Date\n'
                 'Left half = uncontrolled, right half = controlled, outline = date',
                 fontsize=8, fontweight='bold')

    for ax_idx, (energy_key, title, ylabel) in enumerate([
        ('energy_total',   'Total Power (PSD)',   'Power (a.u.)'),
        ('energy_flutter', f'Flutter-Band Power ({FLUTTER_BAND[0]}\u2013{FLUTTER_BAND[1]} Hz, PSD)', 'Power (a.u.)'),
    ]):
        ax = axes[ax_idx]
        cond_keys = [c for c in CONDITIONS if c in all_results]
        n_cond    = len(cond_keys)
        np.random.seed(42)

        for i, cond in enumerate(cond_keys):
            color = CONDITION_COLORS[cond]
            res   = all_results[cond]

            unc       = np.array(res['uncontrolled'][energy_key])
            ctl       = np.array(res['controlled'][energy_key])
            unc_dates = res['uncontrolled']['dates']
            ctl_dates = res['controlled']['dates']

            unique_dates = sorted(set(d for d in (unc_dates + ctl_dates) if d is not None))
            n_dates = max(len(unique_dates), 1)

            band_width = 0.75
            slot_w     = band_width / n_dates
            slot_x     = lambda k: i - band_width / 2 + slot_w * (k + 0.5)

            for k, d in enumerate(unique_dates):
                u_d = unc[[dd == d for dd in unc_dates]]
                c_d = ctl[[dd == d for dd in ctl_dates]]
                if len(u_d) == 0 or len(c_d) == 0:
                    continue

                x       = slot_x(k)
                dcolor  = date_color_map.get(d, '#333333')
                half_w  = slot_w * 0.42

                _half_violin(ax, x, u_d, '#999999', side='left',  width=half_w)
                _half_violin(ax, x, c_d, color,      side='right', width=half_w)

                ax.axvline(x, color=dcolor, lw=1.0, alpha=0.5, zorder=2, ymin=0.02, ymax=0.98)

                ax.plot([x - half_w * 0.5, x], [np.mean(u_d)] * 2, color='#555', lw=1.6, zorder=5)
                ax.plot([x, x + half_w * 0.5], [np.mean(c_d)] * 2, color=color, lw=1.6, zorder=5)

                if np.mean(u_d) > 0:
                    pct_d = (np.mean(c_d) - np.mean(u_d)) / np.mean(u_d) * 100
                    y_top = max(np.max(u_d), np.max(c_d))
                    ax.text(x, y_top, f'{pct_d:+.0f}%', ha='center', va='bottom',
                            fontsize=6, fontweight='bold', color=dcolor)

                y_bot = min(np.min(u_d), np.min(c_d))
                ax.text(x, y_bot, format_date_label(d), ha='center', va='top',
                        fontsize=6, color=dcolor)

            if i > 0:
                ax.axvline(i - 0.5, color='#ddd', lw=0.6, zorder=0)

        ax.set_xticks(range(n_cond))
        ax.set_xticklabels([CONDITION_LABELS[c].replace(' ', '\n') for c in cond_keys],
                           fontsize=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
        ax.grid(True, axis='y', alpha=0.2, linestyle=':')
        fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                         tick_length=2.5,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         linewidth=0.5)
        ax.set_xlim(-0.6, n_cond - 0.4)

    legend_elements = [
        mpatches.Patch(color='#999999', label='Uncontrolled (left half)'),
        mpatches.Patch(color='C0',      label='Controlled (right half, condition color)'),
    ]
    for d, c in sorted(date_color_map.items()):
        legend_elements.append(
            mlines.Line2D([], [], color=c, lw=1.0, label=f'{format_date_label(d)} (outline)'))

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(legend_elements), 6), fontsize=6,
               frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_energy_raincloud_by_date.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_energy_raincloud_by_date.pdf / .png')
    plt.close(fig)


def plot_fft_spectra(all_results, output_dir):
    cond_keys = [c for c in CONDITIONS if c in all_results]
    n_cond    = len(cond_keys)
    f_max     = 25.0

    fig, axes = plt.subplots(
        2, n_cond,
        figsize=(3.0 * n_cond, 7.0),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.12},
        sharex=True,
    )
    if n_cond == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle('Mean FFT Spectrum by Condition\nRow 1 = Uncontrolled  |  Row 2 = Controlled',
                 fontsize=8, fontweight='bold')

    all_means = []
    common_f  = None
    precomp   = {}

    for cond in cond_keys:
        res = all_results[cond]
        precomp[cond] = {}
        for seg in ['uncontrolled', 'controlled']:
            fl = res[seg]['freqs']
            ml = res[seg]['mags']
            if not fl:
                precomp[cond][seg] = None
                continue
            f_ref = fl[0][fl[0] <= f_max]
            if common_f is None:
                common_f = f_ref
            mags_i = [np.interp(common_f, f, m) for f, m in zip(fl, ml)]
            mean_m = np.mean(mags_i, axis=0)
            std_m  = np.std(mags_i, axis=0)
            precomp[cond][seg] = {'mags_i': mags_i, 'mean': mean_m, 'std': std_m}
            all_means.append(mean_m)

    y_max = max(np.max(m) for m in all_means) * 1.15 if all_means else 1.0

    for col, cond in enumerate(cond_keys):
        color = CONDITION_COLORS[cond]

        for row, seg in enumerate(['uncontrolled', 'controlled']):
            ax        = axes[row, col]
            seg_color = '#555555' if seg == 'uncontrolled' else color
            data      = precomp[cond][seg]

            ax.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.4, zorder=0)
            ax.axvline(FLUTTER_FREQ, color='#e65100', lw=0.8, ls='--', alpha=0.6, zorder=1)

            if data is not None:
                for m_i in data['mags_i']:
                    ax.plot(common_f, m_i, color=seg_color, alpha=0.12, lw=0.5, zorder=2)
                ax.plot(common_f, data['mean'], color=seg_color, lw=1.8, zorder=4)
                ax.fill_between(common_f,
                                data['mean'] - data['std'],
                                data['mean'] + data['std'],
                                color=seg_color, alpha=0.20, zorder=3)

                from scipy.signal import find_peaks as _find_peaks
                mean_m = data['mean']
                peak_idx, _ = _find_peaks(mean_m,
                                          height=np.max(mean_m) * 0.05,
                                          distance=int(0.5 / (common_f[1] - common_f[0])))
                if len(peak_idx) > 0:
                    peak_idx = peak_idx[common_f[peak_idx] > 0.5]
                    if len(peak_idx) == 0:
                        continue
                    top3 = peak_idx[np.argsort(mean_m[peak_idx])[-3:]]
                    top3 = top3[np.argsort(common_f[top3])]
                    used_x = []
                    for pi in top3:
                        fx = common_f[pi]
                        fy = mean_m[pi]
                        lx = fx + 0.4
                        for ux in used_x:
                            if abs(lx - ux) < 1.2:
                                lx = ux + 1.3
                        used_x.append(lx)
                        ax.annotate(f'{fx:.1f} Hz',
                                    xy=(fx, fy),
                                    xytext=(lx, fy + y_max * 0.04),
                                    fontsize=6, color=seg_color,
                                    arrowprops=dict(arrowstyle='->', color=seg_color,
                                                    lw=0.8, shrinkA=0, shrinkB=2),
                                    zorder=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='#aaa', fontsize=6)

            ax.set_xlim(0, f_max)
            ax.set_ylim(0, y_max)
            fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                             tick_length=2.5,
                                             spine_locations={'left': 5, 'bottom': 5},
                                             linewidth=0.5)
            ax.grid(True, alpha=0.18)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.xaxis.set_tick_params(labelsize=7)

            if row == 0:
                ax.set_title(CONDITION_LABELS[cond], fontsize=8,
                             color=color, fontweight='bold', pad=5)

            if col == 0:
                row_label = 'Uncontrolled' if seg == 'uncontrolled' else 'Controlled'
                ax.set_ylabel(f'{row_label}\nFFT Magnitude', fontsize=8)
            else:
                ax.tick_params(labelleft=False)

            if row == 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=8)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_spectra_by_condition.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_spectra_by_condition.pdf / .png')
    plt.close(fig)


def draw_paired_reduction_panel(ax, all_results, condition='CLOSED_LOOP', scale=1.0,
                                controlled_color=None, uncontrolled_color=None,
                                show_legend=True):
    """
    Draws a jittered-scatter of per-sample PERCENT CHANGE (paired
    controlled vs uncontrolled) for a single condition, across two
    groups: total spectral power and flutter-band (4-6 Hz) power. Matches
    the "% Change" convention used in fig3/fig4's triptych panels exactly
    — one cluster per group showing the actual percent-change values
    (not raw absolute power in two side-by-side clusters), y-axis
    centered on 0%, condition means + significance stacked in one shared
    line near the top.

    scale: multiplier applied to fontsizes, marker sizes, and line widths.
    Defaults to 1.0. Use ~0.5 for a small figurefirst panel.

    controlled_color: color for the scatter/mean-line in each group.
    Defaults to the condition's own color if left as None.
    uncontrolled_color: unused now that raw uncontrolled/controlled
    values are no longer separately plotted — kept as a parameter only
    so existing calls passing it don't break; has no visual effect.

    show_legend: no longer draws a legend (there's nothing left to
    label — only one series per group now) — kept as a parameter only
    so existing calls passing it don't break.
    """
    default_color = CONDITION_COLORS[condition]
    color = controlled_color if controlled_color is not None else default_color
    res   = all_results[condition]
    np.random.seed(42)

    groups = [
        ('Total\nPower', 'energy_total'),
        (f'{FLUTTER_BAND[0]:.0f}\u2013{FLUTTER_BAND[1]:.0f} Hz\nBand', 'energy_flutter'),
    ]

    per_group_pcts = []
    sigs = []
    for gi, (label, key) in enumerate(groups):
        unc = np.array(res['uncontrolled'][key])
        ctl = np.array(res['controlled'][key])
        if len(unc) == len(ctl) and len(unc) > 0:
            pcts_i = (ctl - unc) / unc * 100
        else:
            mean_unc = np.mean(unc) if len(unc) else np.nan
            mean_ctl = np.mean(ctl) if len(ctl) else np.nan
            pcts_i = np.array([(mean_ctl - mean_unc) / mean_unc * 100]) if mean_unc > 0 else np.array([np.nan])
        per_group_pcts.append(pcts_i)

        if len(unc) > 1 and len(ctl) > 1:
            _, p = stats.ttest_rel(unc, ctl) if len(unc) == len(ctl) else stats.ttest_ind(unc, ctl)
            sigs.append('***' if p < 0.001 else '**' if p < 0.01 else
                        '*'   if p < 0.05  else 'ns')
        else:
            sigs.append('')

    for gi, pcts_i in enumerate(per_group_pcts):
        finite = pcts_i[np.isfinite(pcts_i)]
        if len(finite) == 0:
            continue
        jitter = np.random.normal(0, 0.06, size=len(finite))
        ax.scatter(gi + jitter, finite, color=color, alpha=0.7, s=35 * scale, zorder=3)
        ax.plot([gi - 0.15, gi + 0.15], [np.mean(finite), np.mean(finite)],
                color=color, lw=2.0 * scale, zorder=4)

    # Means + significance placed together in one shared line near the
    # TOP of the panel (axes-fraction y), matching fig3/fig4's
    # convention, instead of floating above each group's own points.
    for gi, (pcts_i, sig) in enumerate(zip(per_group_pcts, sigs)):
        finite = pcts_i[np.isfinite(pcts_i)]
        if len(finite) == 0:
            continue
        mean_pct = np.mean(finite)
        ax.text(gi, 0.96, f'{mean_pct:+.1f}%\n{sig}', ha='center', va='top',
                fontsize=6, fontweight='bold', color=color, clip_on=True,
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color='black', lw=0.8 * scale)

    all_finite = np.concatenate([p[np.isfinite(p)] for p in per_group_pcts if np.any(np.isfinite(p))])
    if len(all_finite):
        y_min, y_max = np.min(all_finite), np.max(all_finite)
        span = y_max - y_min if y_max > y_min else 1.0
        ax.set_ylim(y_min - span * 0.15, y_max + span * 0.35)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6)
    ax.set_ylabel('Power Change (%)', fontsize=8, labelpad=2)
    ax.set_title(f'{CONDITION_LABELS.get(condition, condition)}: Power Change',
                 fontsize=8, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def draw_single_condition_fft_overlay(ax, all_results, condition, f_max=25.0, scale=1.0,
                                      controlled_color=None, uncontrolled_color=None,
                                      show_legend=True, show_inset=False,
                                      inset_freq_range=None, inset_bbox=(0.55, 0.55, 0.42, 0.4),
                                      title_color=None, show_title=True):
    """
    Draws the uncontrolled vs controlled mean FFT spectrum for a single
    condition into whatever matplotlib Axes you give it. Uncontrolled is
    drawn first (behind), controlled is drawn on top... wait, see note
    below — actual on-top trace is uncontrolled, drawn last.

    scale: multiplier applied to all fontsizes and line widths. Defaults
    to 1.0 (unchanged). Use ~0.5-0.6 for a small figurefirst panel.

    controlled_color / uncontrolled_color: override the default colors
    (condition color for controlled, grey for uncontrolled). Pass e.g.
    controlled_color='green', uncontrolled_color='red' to match a
    time-trace panel's color scheme elsewhere in the same figure. If
    left as None, falls back to the original per-condition/grey scheme
    (used by fig3/fig4's multi-condition FFT rows, which rely on each
    condition having its own distinct color).

    show_legend: if False, omits the Controlled/Uncontrolled legend.
    Defaults to True (unchanged behavior).

    show_inset: if True, adds a small zoomed-in inset axis (in the
    panel's own upper-right corner, over otherwise-empty space) showing
    a magnified view of the flutter-band peak region. Defaults to False.

    inset_freq_range: (f_low, f_high) for the inset's x-axis. Defaults
    to a window around FLUTTER_BAND with some padding if not given.

    inset_bbox: (x0, y0, width, height) in axes-fraction coordinates for
    where the inset sits. Default places it in the upper-right corner.

    title_color: override the title's color (defaults to the condition's
    own color). Pass 'black' for a neutral title when using custom
    controlled_color/uncontrolled_color that no longer matches the
    condition's own color scheme.

    show_title: if False, omits the condition-name title above the panel
    entirely — useful when the condition is already labeled elsewhere
    (e.g. a row of icons above the FFT panels). Defaults to True.

    Reusable in a figurefirst-tagged axis (e.g. layout.axes['FFT']) the
    same way as draw_energy_panel() above.
    """
    if condition not in all_results:
        ax.text(0.5, 0.5, f'No data for {condition}', ha='center', va='center',
                transform=ax.transAxes, color='#aaa', fontsize=6)
        return

    default_color = CONDITION_COLORS[condition]
    ctl_color = controlled_color if controlled_color is not None else default_color
    unc_color = uncontrolled_color if uncontrolled_color is not None else '#555'
    unc_ls    = '--' if uncontrolled_color is None else '-'  # dashed only in the default grey scheme

    res   = all_results[condition]

    # Track (color, freqs, mean_spectrum) for whichever traces actually
    # have data, so we can find the single tallest peak across both below,
    # and so the inset (if requested) can reuse the same computed means.
    traces = []

    # Controlled drawn FIRST/behind, so uncontrolled ends up visually in
    # front of it (matches the established convention elsewhere).
    ctl_freqs = res['controlled']['freqs']
    ctl_mags  = res['controlled']['mags']
    mean_ctl = std_ctl = f_ref_ctl = None
    if ctl_freqs:
        f_ref_ctl = ctl_freqs[0][ctl_freqs[0] <= f_max]
        mags_i    = [np.interp(f_ref_ctl, f, m) for f, m in zip(ctl_freqs, ctl_mags)]
        mean_ctl  = np.mean(mags_i, axis=0)
        std_ctl   = np.std(mags_i, axis=0)
        ax.plot(f_ref_ctl, mean_ctl, color=ctl_color, lw=2.0 * scale,
                label='Controlled', zorder=3)
        ax.fill_between(f_ref_ctl, mean_ctl - std_ctl, mean_ctl + std_ctl,
                        color=ctl_color, alpha=0.15, zorder=4)
        traces.append((ctl_color, f_ref_ctl, mean_ctl))

    # Uncontrolled reference — drawn LAST/on top, per established
    # convention, so it visually sits in front of the controlled trace.
    unc_freqs = res['uncontrolled']['freqs']
    unc_mags  = res['uncontrolled']['mags']
    mean_unc = f_ref_unc = None
    if unc_freqs:
        f_ref_unc = unc_freqs[0][unc_freqs[0] <= f_max]
        mags_i    = [np.interp(f_ref_unc, f, m) for f, m in zip(unc_freqs, unc_mags)]
        mean_unc  = np.mean(mags_i, axis=0)
        ax.plot(f_ref_unc, mean_unc, color=unc_color, lw=2.0 * scale, ls=unc_ls,
                label='Uncontrolled', zorder=5, alpha=0.8)
        traces.append((unc_color, f_ref_unc, mean_unc))

    ax.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
    ax.axvline(FLUTTER_FREQ, color='#e65100', lw=0.8 * scale, ls=':', alpha=0.7, zorder=1)

    # Label the top peak in each trace separately. Text is anchored to a
    # FIXED x=3 Hz position (not wherever the peak happens to sit) so
    # labels land in the same consistent spot across every condition's
    # panel, rather than drifting around based on where each condition's
    # peak actually is. The arrow still points from that fixed text
    # position to the real peak location. The two traces' labels are
    # stacked at different axes-fraction y-heights (not data y-values),
    # so they never overlap each other regardless of how tall either
    # peak actually is. Whichever trace's peak is actually LARGER gets
    # the higher slot (0.92), not whichever trace it is (uncontrolled vs
    # controlled) — e.g. if Open Loop's controlled peak (1.24) exceeds
    # its uncontrolled peak (0.30), the 1.24 label sits on top.
    label_x = 0.8
    peak_info = []
    for trace_color, f_arr, m_arr in traces:
        i = np.argmax(m_arr)
        peak_f, peak_mag = f_arr[i], m_arr[i]
        peak_info.append((trace_color, peak_f, peak_mag))

    peak_info.sort(key=lambda p: p[2], reverse=True)  # largest peak first
    y_slots = [0.70, 0.60]
    for (trace_color, peak_f, peak_mag), y_frac in zip(peak_info, y_slots):
        ax.annotate(f'{peak_mag:.2f}',
                    xy=(peak_f, peak_mag), xycoords='data',
                    xytext=(label_x, y_frac), textcoords=('data', 'axes fraction'),
                    fontsize=6, color=trace_color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=trace_color, lw=0.8 * scale))

    ax.set_xlabel('Frequency (Hz)', fontsize=8, labelpad=2)
    ax.set_ylabel('FFT Magnitude (normalized)', fontsize=8, labelpad=2)
    if show_title:
        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=8,
                     color=(title_color if title_color is not None else default_color),
                     fontweight='bold')
    ax.set_xlim(0, f_max)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)
    ax.grid(True, alpha=0.2)

    if show_inset and mean_ctl is not None and mean_unc is not None:
        if inset_freq_range is None:
            pad = (FLUTTER_BAND[1] - FLUTTER_BAND[0])
            inset_freq_range = (max(0, FLUTTER_BAND[0] - pad), FLUTTER_BAND[1] + pad)

        axins = ax.inset_axes(inset_bbox)
        mask_ctl = (f_ref_ctl >= inset_freq_range[0]) & (f_ref_ctl <= inset_freq_range[1])
        mask_unc = (f_ref_unc >= inset_freq_range[0]) & (f_ref_unc <= inset_freq_range[1])
        axins.plot(f_ref_ctl[mask_ctl], mean_ctl[mask_ctl], color=ctl_color, lw=1.5 * scale)
        axins.plot(f_ref_unc[mask_unc], mean_unc[mask_unc], color=unc_color, lw=1.5 * scale, ls=unc_ls)
        axins.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
        axins.set_xlim(*inset_freq_range)
        axins.tick_params(labelsize=6 * scale)
        fifi.mpl_functions.adjust_spines(axins, ['left', 'bottom'],
                                         tick_length=2.0 * scale,
                                         spine_locations={'left': 3, 'bottom': 3},
                                         linewidth=0.5 * scale)
        axins.set_facecolor('white')
        axins.patch.set_alpha(0.95)


def plot_single_condition_overlay(all_results, output_dir, condition):
    """Standalone single-panel PNG/PDF version of draw_single_condition_fft_overlay()."""
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_single_condition_fft_overlay(ax, all_results, condition)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_overlay_{condition.lower()}.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_overlay_{condition.lower()}.pdf / .png')
    plt.close(fig)


def plot_fft_overlay(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Mean FFT Spectrum — All Conditions\nControlled segments vs uncontrolled reference',
                 fontsize=8, fontweight='bold')

    f_max = 25.0
    uncontrolled_plotted = False

    for cond in CONDITIONS:
        if cond not in all_results:
            continue
        color = CONDITION_COLORS[cond]
        res   = all_results[cond]

        if not uncontrolled_plotted:
            freqs_list = res['uncontrolled']['freqs']
            mags_list  = res['uncontrolled']['mags']
            if freqs_list:
                f_ref = freqs_list[0][freqs_list[0] <= f_max]
                mags_i = [np.interp(f_ref, f, m) for f, m in zip(freqs_list, mags_list)]
                mean_m = np.mean(mags_i, axis=0)
                ax.plot(f_ref, mean_m, color='#555', lw=2.2, ls='--',
                        label='Uncontrolled (ref)', zorder=5, alpha=0.7)
                uncontrolled_plotted = True

        freqs_list = res['controlled']['freqs']
        mags_list  = res['controlled']['mags']
        if not freqs_list:
            continue
        f_ref  = freqs_list[0][freqs_list[0] <= f_max]
        mags_i = [np.interp(f_ref, f, m) for f, m in zip(freqs_list, mags_list)]
        mean_m = np.mean(mags_i, axis=0)
        std_m  = np.std(mags_i, axis=0)
        ax.plot(f_ref, mean_m, color=color, lw=1.8,
                label=CONDITION_LABELS[cond], zorder=4)
        ax.fill_between(f_ref, mean_m - std_m, mean_m + std_m,
                        color=color, alpha=0.10, zorder=3)

    ax.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
    ax.axvline(FLUTTER_FREQ, color='#e65100', lw=0.8, ls=':', alpha=0.7)
    ax.text(FLUTTER_FREQ + 0.2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.01,
            f'{FLUTTER_FREQ} Hz', fontsize=6, color='#e65100', va='top')

    ax.set_xlabel('Frequency (Hz)', fontsize=8, labelpad=2)
    ax.set_ylabel('FFT Magnitude (normalized)', fontsize=8, labelpad=2)
    ax.set_xlim(0, f_max)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        p = os.path.join(output_dir, f'fft_overlay_all_conditions.{ext}')
        plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f'[SAVED] fft_overlay_all_conditions.pdf / .png')
    plt.close(fig)


def print_energy_stats(all_results):
    print('\n' + '='*72)
    print(f'{"SPECTRAL ENERGY SUMMARY":^72}')
    print('='*72)
    print(f'{"Condition":<22} {"Unc E_total":>12} {"Ctl E_total":>12} '
          f'{"Δ Total":>10} {"Unc E_flutter":>14} {"Ctl E_flutter":>14} {"Δ Flutter":>10}')
    print('-'*72)

    for cond in CONDITIONS:
        if cond not in all_results:
            continue
        res = all_results[cond]
        unc_t = np.array(res['uncontrolled']['energy_total'])
        ctl_t = np.array(res['controlled']['energy_total'])
        unc_f = np.array(res['uncontrolled']['energy_flutter'])
        ctl_f = np.array(res['controlled']['energy_flutter'])

        if not len(unc_t) or not len(ctl_t):
            continue

        d_total   = (np.mean(ctl_t) - np.mean(unc_t)) / np.mean(unc_t) * 100
        d_flutter = (np.mean(ctl_f) - np.mean(unc_f)) / np.mean(unc_f) * 100

        print(f'{CONDITION_LABELS[cond]:<22} '
              f'{np.mean(unc_t):>12.4f} {np.mean(ctl_t):>12.4f} {d_total:>+10.1f}% '
              f'{np.mean(unc_f):>14.4f} {np.mean(ctl_f):>14.4f} {d_flutter:>+10.1f}%')

    print('='*72)


def main():
    # These rcParams only apply when this script is run standalone
    # (python -m flutter.fft_energy), producing its own PNG/PDF outputs.
    # Setting them at module level used to silently override whatever
    # font/style a notebook importing this module had already set up
    # (e.g. Times New Roman), since the import itself ran this code.
    matplotlib.rcParams.update({
        'font.family':    'sans-serif',
        'font.size':      9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.4,
        'pdf.fonttype':   42,
        'ps.fonttype':    42,
    })

    parser = argparse.ArgumentParser(description='FFT spectral energy analysis')
    parser.add_argument('--bag-dir',    type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument('--f-max',      type=float, default=25.0,
                        help='Max frequency to plot (default 25 Hz)')
    args = parser.parse_args()

    if args.bag_dir:
        bag_dir = args.bag_dir
    else:
        bag_dir = str(OL_RDM_DIR)

    output_dir = args.output_dir or str(OUTPUT_DIR)
    dated_dir  = os.path.join(output_dir, datetime.now().strftime('%Y%m%d_%H%M%S') + '_fft_energy')
    os.makedirs(dated_dir, exist_ok=True)
    output_dir = dated_dir
    print(f'Output directory: {output_dir}')
    print()

    all_results = {}

    for condition in args.conditions:
        if condition == 'CLOSED_LOOP':
            pattern = os.path.join(bag_dir, CLOSED_LOOP_PATTERN)
        elif condition == 'REPLAY':
            pattern = os.path.join(bag_dir, 'REPLAY_sample*.bag')
        else:
            pattern = os.path.join(bag_dir, f'{condition}_sample*.bag')

        bag_paths = sorted(glob.glob(pattern))
        if not bag_paths:
            print(f'[{condition}] No bags found')
            continue

        print(f'[{condition}] {len(bag_paths)} bags')
        results = process_condition_fft(
            bag_paths,
            is_closed_loop=(condition == 'CLOSED_LOOP')
        )
        all_results[condition] = results

    if not all_results:
        print('[ERROR] No data loaded.')
        return

    date_color_map = build_date_color_map(all_results)
    if date_color_map:
        print('\nDate -> edge color mapping:')
        for d, c in sorted(date_color_map.items()):
            print(f'  {format_date_label(d)} ({d}): {c}')

    print('\nGenerating figures...')
    plot_energy_comparison(all_results, output_dir)
    plot_energy_reduction_dumbbell(all_results, output_dir, date_color_map)
    plot_energy_raincloud_by_date(all_results, output_dir, date_color_map)
    plot_fft_spectra(all_results, output_dir)
    plot_fft_overlay(all_results, output_dir)
    if 'CLOSED_LOOP' in all_results:
        plot_single_condition_overlay(all_results, output_dir, 'CLOSED_LOOP')
    print_energy_stats(all_results)
    print('\nDone.')


if __name__ == '__main__':
    main()