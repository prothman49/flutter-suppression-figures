#!/usr/bin/env python3
"""
fig5_jitter_freq_sweep.py
===========================
Two portions for fig5:

  Portion 1 (draw_overlaid_panel): Fixed Open Loop vs Uniform Jitter at
  off-flutter frequencies, individual samples + 95% CI, with Open Loop
  5.5 Hz (no sync) and Closed Loop plotted as satellite dot clusters next
  to the matched-frequency jitter cloud — extracted from the authoritative
  version of analyze_jitter_freq_sweep.py's plot_overlaid() so it can be
  drawn directly into a figurefirst-tagged axis.

  Portion 2 (draw_jitter_fft_row): uncontrolled-vs-controlled FFT overlay
  for Uniform Jitter at each of the 4 tested frequencies, computed
  directly from the actual bag files
  (UNIFORM_JITTER_{freq}p{decimal}hz_sample*.bag in the jitter_freq_sweep
  folder) — reuses the FFT machinery from plot_fft_energy.py so the
  computation stays consistent with your other figures.

Usage (standalone):
    python3 fig5_jitter_freq_sweep.py
"""

import sys
import os
import glob
import json
import numpy as np
import matplotlib
import figurefirst as fifi
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

from .fft_energy import (process_condition_fft, read_optic_flow_segment, compute_fft,
                              total_energy, FLUTTER_BAND, FLUTTER_FREQ)

from .frequency_profile import process_multi_trial_bag
from .config import OFF_FREQ_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_PATH    = OFF_FREQ_DIR
JITTER_DIR   = BASE_PATH / 'jitter_freq_sweep'
OL_RDM_DIR   = BASE_PATH / 'OL_RDM'

FIXED_OL_DIRS = {
    3.0:  BASE_PATH / '3hz',
    8.0:  BASE_PATH / '8hz',
    10.0: BASE_PATH / '10hz',
}

MATCHED_FREQ_TARGET = 5.5

COLOR_JITTER   = '#9C27B0'
COLOR_FIXED_OL = '#2196F3'
COLOR_CL       = '#4CAF50'
COLOR_OL55     = '#2196F3'

JITTER_FREQUENCIES = [3.0, 5.5, 8.0, 10.0]


def freq_to_filename_fragment(freq):
    """3.0 -> '3p0', 5.5 -> '5p5', 10.0 -> '10p0'."""
    s = f'{freq:.1f}'
    return s.replace('.', 'p')


def load_jitter_sweep(jitter_dir):
    """Load jitter frequency sweep JSONs — keyed by control_freq_hz."""
    data = {}
    for jf in sorted(Path(jitter_dir).glob('*.json')):
        try:
            with open(jf) as f:
                d = json.load(f)
            freq = d['meta']['control_freq_hz']
            pct  = d['reduction']['magnitude_reduction_pct']
            if freq not in data:
                data[freq] = []
            data[freq].append(pct)
            print(f"  {jf.name}: {freq} Hz -> {pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] {jf.name}: {e}")
    return {k: np.array(v) for k, v in sorted(data.items())}


def load_fixed_ol(dirs):
    """Load existing fixed open-loop off-frequency data."""
    data = {}
    for freq, folder in dirs.items():
        if not folder.exists():
            print(f"  [SKIP] {folder} not found")
            continue
        reductions = []
        for jf in sorted(folder.glob('*.json')):
            try:
                with open(jf) as f:
                    d = json.load(f)
                pct = None
                if 'pairs' in d:
                    for pair in d['pairs']:
                        if 'reduction_percent' in pair:
                            reductions.append(pair['reduction_percent'])
                    continue
                if 'control_off' in d and 'control_on' in d:
                    off_c = d['control_off'].get('cycles', [])
                    on_c  = d['control_on'].get('cycles', [])
                    for oc, nc in zip(off_c, on_c):
                        om = oc.get('avg_magnitude', 0)
                        nm = nc.get('avg_magnitude', 0)
                        if om > 0:
                            reductions.append((om - nm) / om * 100)
                    continue
                if 'effectiveness' in d:
                    pct = (d['effectiveness'].get('magnitude_reduction_percent') or
                           d['effectiveness'].get('reduction_percent'))
                elif 'reduction' in d:
                    pct = d['reduction'].get('magnitude_reduction_pct')
                if pct is not None:
                    reductions.append(pct)
            except Exception:
                continue
        if reductions:
            data[freq] = np.array(reductions)
            print(f"  Fixed OL {freq} Hz: n={len(reductions)}, mean={np.mean(reductions):.1f}%")
    return data


def load_new_session_jsons(json_dir, condition):
    """
    Load magnitude_reduction_pct from new-style per-sample JSONs, e.g.
    CLOSED_LOOP_sample*.json or OPEN_LOOP_55_sample*.json.
    """
    json_dir = Path(json_dir)
    reductions = []
    for jf in sorted(json_dir.glob(f'{condition}_sample*.json')):
        try:
            with open(jf) as f:
                d = json.load(f)
            pct = d.get('reduction', {}).get('magnitude_reduction_pct', None)
            if pct is not None:
                reductions.append(pct)
                print(f"  {jf.name}: {pct:.1f}%")
        except Exception as e:
            print(f"  [ERROR] {jf.name}: {e}")
    if reductions:
        print(f"  {condition}: n={len(reductions)}, mean={np.mean(reductions):.1f}%")
    return np.array(reductions) if reductions else None


def load_closed_loop_ref(ol_rdm_dir):
    """Load closed loop reference from OL_RDM JSONs (CLOSED_LOOP_sample*.json)."""
    return load_new_session_jsons(ol_rdm_dir, 'CLOSED_LOOP')


def load_open_loop_55_ref(ol_rdm_dir):
    """Load Open Loop 5.5 Hz (no sync) reference from OL_RDM JSONs."""
    return load_new_session_jsons(ol_rdm_dir, 'OPEN_LOOP_55')


def draw_ci_box(ax, x, values, color, box_width=0.35):
    mean  = np.mean(values)
    n     = len(values)
    ci    = stats.sem(values) * stats.t.ppf(0.975, df=max(n - 1, 1))
    half  = box_width / 2.0
    rect  = mpatches.FancyBboxPatch(
        (x - half, mean - ci), box_width, 2 * ci,
        boxstyle="square,pad=0",
        linewidth=1.8, linestyle="--",
        edgecolor=color, facecolor="none", zorder=4
    )
    ax.add_patch(rect)
    ax.plot([x - half, x + half], [mean, mean],
            color=color, linewidth=2.5, zorder=5)
    return mean, mean - ci, mean + ci


def draw_overlaid_panel(ax, jitter_data, fixed_ol_data, cl_ref, ol55_ref, scale=1.0):
    """
    Draws Fixed Open Loop vs Uniform Jitter at off-flutter frequencies,
    with Open Loop 5.5 Hz (no sync) and Closed Loop plotted as satellite
    dot clusters right next to the matched-frequency jitter cloud, into
    whatever matplotlib Axes you give it. Matches the plain scatter +
    mean-line style (no dashed CI boxes) and shared top-line label
    convention used in fig3/fig4's % Change panels — each x-position's
    labels (however many series sit there) are stacked vertically near
    the top of the panel instead of floating next to each cluster.

    scale: multiplier for fontsizes, marker sizes, and line widths.
    """
    all_freqs = sorted(set(list(jitter_data.keys()) + list(fixed_ol_data.keys())))

    matched_freq = None
    if jitter_data:
        matched_freq = min(jitter_data.keys(), key=lambda f: abs(f - MATCHED_FREQ_TARGET))

    all_vals = []
    for d in [jitter_data, fixed_ol_data]:
        for v in d.values():
            all_vals.extend(v)
    if cl_ref is not None:
        all_vals.extend(cl_ref)
    if ol55_ref is not None:
        all_vals.extend(ol55_ref)
    y_min = min(all_vals) - 8
    y_max = max(all_vals) + 10

    np.random.seed(42)
    x_pos = {f: i for i, f in enumerate(all_freqs)}

    # Collect (color, mean, sig_or_None) per x-tick position, in draw
    # order, so multiple series sharing the same tick (e.g. fixed open
    # loop + jitter at every frequency, plus OL55/Closed Loop satellites
    # at the matched frequency) can be stacked into one shared label
    # column instead of each floating next to its own cluster.
    top_labels = {}  # tick_index -> list of (x_actual, color, mean, sig_or_None)

    def add_label(tick_idx, x_actual, color, values, sig=None):
        top_labels.setdefault(tick_idx, []).append((x_actual, color, np.mean(values), sig))

    for freq in all_freqs:
        if freq not in fixed_ol_data:
            continue
        vals = fixed_ol_data[freq]
        x    = x_pos[freq] - 0.18
        jit  = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(x + jit, vals, color=COLOR_FIXED_OL, alpha=0.75,
                   s=30 * scale, zorder=3, edgecolors=COLOR_FIXED_OL, linewidth=0.5 * scale)
        ax.plot([x - 0.1, x + 0.1], [np.mean(vals), np.mean(vals)],
                color=COLOR_FIXED_OL, lw=2.0 * scale, zorder=4)
        add_label(x_pos[freq], x, COLOR_FIXED_OL, vals)

    for freq in all_freqs:
        if freq not in jitter_data:
            continue
        vals = jitter_data[freq]
        x    = x_pos[freq] if freq == matched_freq else x_pos[freq] + 0.18
        jit  = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(x + jit, vals, color=COLOR_JITTER, alpha=0.85,
                   s=30 * scale, zorder=3, edgecolors=COLOR_JITTER, linewidth=0.5 * scale)
        ax.plot([x - 0.1, x + 0.1], [np.mean(vals), np.mean(vals)],
                color=COLOR_JITTER, lw=2.0 * scale, zorder=4)

        sig = None
        if freq in fixed_ol_data:
            _, p = stats.ttest_ind(fixed_ol_data[freq], vals)
            sig = ('***' if p < 0.001 else '**' if p < 0.01 else
                   '*'   if p < 0.05  else 'ns')
        add_label(x_pos[freq], x, COLOR_JITTER, vals, sig)

        if freq == matched_freq:
            tick_x = x_pos[freq]
            ol55_x = tick_x - 0.15
            cl_x   = tick_x - 0.20

            if ol55_ref is not None:
                jit_ol55 = np.random.normal(0, 0.025, size=len(ol55_ref))
                ax.scatter(ol55_x + jit_ol55, ol55_ref, color=COLOR_OL55,
                           alpha=0.85, s=25 * scale, zorder=3, marker='o',
                           edgecolors=COLOR_OL55, linewidth=0.5 * scale,
                           label='Open Loop 5.5 Hz (no sync)')
                ax.plot([ol55_x - 0.05, ol55_x + 0.05], [np.mean(ol55_ref), np.mean(ol55_ref)],
                        color=COLOR_OL55, lw=2.0 * scale, zorder=4)
                add_label(tick_x, ol55_x, COLOR_OL55, ol55_ref)

            if cl_ref is not None:
                jit_cl = np.random.normal(0, 0.025, size=len(cl_ref))
                ax.scatter(cl_x + jit_cl, cl_ref, color=COLOR_CL,
                           alpha=0.85, s=25 * scale, zorder=3, marker='o',
                           edgecolors=COLOR_CL, linewidth=0.5 * scale,
                           label='Closed Loop')
                ax.plot([cl_x - 0.05, cl_x + 0.05], [np.mean(cl_ref), np.mean(cl_ref)],
                        color=COLOR_CL, lw=2.0 * scale, zorder=4)
                add_label(tick_x, cl_x, COLOR_CL, cl_ref)

    # All labels sit on ONE shared row (same y) at each tick group, but
    # spread out horizontally in their original left-to-right order (by
    # actual cluster x-position) so text never overlaps even when several
    # clusters sit close together (e.g. the matched frequency's four
    # series) — approximately over their own data, not stacked in rows.
    label_y = 0.96
    for tick_idx, entries in top_labels.items():
        entries_sorted = sorted(entries, key=lambda e: e[0])
        n = len(entries_sorted)
        min_spacing = 0.26  # enough horizontal room for a 2-line "%\nsig" label at fontsize=6
        total_width = min_spacing * (n - 1)
        for i, (x_actual, color, mean_val, sig) in enumerate(entries_sorted):
            x_label = tick_idx - total_width / 2 + i * min_spacing
            txt = f'{mean_val:+.1f}%\n{sig}' if sig else f'{mean_val:+.1f}%'
            ax.text(x_label, label_y, txt,
                    ha='center', va='top', fontsize=6, fontweight='bold',
                    color=color, clip_on=True, transform=ax.get_xaxis_transform())

    ax.axhline(0, color='black', lw=0.8 * scale, alpha=0.3)

    # "Open Loop" is one conceptual condition tested at multiple
    # frequencies (fixed off-flutter frequencies in orange, the matched
    # 5.5 Hz frequency in blue) — rather than two separate legend rows,
    # HandlerTuple lets both color patches share ONE row as a single
    # split-color swatch, visually tying them together as variants of
    # the same underlying condition.
    from matplotlib.legend_handler import HandlerTuple
    ol_orange_patch = mpatches.Patch(color=COLOR_FIXED_OL)
    ol_blue_patch   = mpatches.Patch(color=COLOR_OL55)
    jitter_patch    = mpatches.Patch(color=COLOR_JITTER)

    handles = [(ol_orange_patch, ol_blue_patch), jitter_patch]
    labels  = ['Open Loop', 'Uniform Jitter']
    if cl_ref is not None:
        handles.append(mpatches.Patch(color=COLOR_CL))
        labels.append('Closed Loop')

    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
             fontsize=6, loc='upper right', framealpha=0.9)

    ax.set_xticks(range(len(all_freqs)))
    ax.set_xticklabels([f'{f:g} Hz' for f in all_freqs], fontsize=6)
    ax.tick_params(axis='y', labelsize=6 * scale)
    ax.set_ylabel('Power Change', fontsize=8, labelpad=2)
    ax.set_xlabel('Control Frequency (Hz)', fontsize=8, labelpad=2)
    ax.set_title('Fixed Open Loop vs Uniform Jitter\nat Off-Flutter Frequencies',
                 fontsize=8, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    ax.set_xlim(-0.6, len(all_freqs) - 0.15)
    ax.set_ylim(y_min, y_max)
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)


def plot_overlaid(jitter_data, fixed_ol_data, cl_ref, ol55_ref, output_dir):
    """Standalone single-panel PNG/PDF version of draw_overlaid_panel()."""
    fig, ax = plt.subplots(figsize=(11, 6))
    draw_overlaid_panel(ax, jitter_data, fixed_ol_data, cl_ref, ol55_ref)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'jitter_freq_sweep_overlaid.{ext}'),
                    dpi=200, bbox_inches='tight')
    print("[SAVED] jitter_freq_sweep_overlaid.png / .pdf")
    plt.close(fig)


def load_jitter_fft_data(jitter_dir=None):
    """
    Loads and processes bags for Uniform Jitter at each of the 4 tested
    frequencies, returning a dict keyed by frequency -> process_condition_fft()
    result dict.
    """
    if jitter_dir is None:
        jitter_dir = JITTER_DIR

    all_results = {}
    for freq in JITTER_FREQUENCIES:
        frag = freq_to_filename_fragment(freq)
        pattern = f'UNIFORM_JITTER_{frag}hz_sample*.bag'
        bag_paths = sorted(glob.glob(os.path.join(str(jitter_dir), pattern)))
        if not bag_paths:
            print(f'[{freq} Hz] No bags found matching {pattern}')
            continue
        print(f'[{freq} Hz] {len(bag_paths)} bags')
        all_results[freq] = process_condition_fft(bag_paths, is_closed_loop=False)

    return all_results


def _pct_reduction_from_results(results):
    """
    Computes paired per-sample % reduction in flutter-band (4-6 Hz) PSD
    power from a process_condition_fft()/process_multi_trial_bag()-style
    results dict — i.e. the SAME Welch PSD-based energy_flutter values
    used everywhere else (fig1/fig3/fig4), rather than a separately
    pre-computed JSON percentage. Returns a 1D array of per-sample %
    values (empty array if the arrays are empty or mismatched/unpaired).

    Sign convention matches fig3's draw_frequency_triptych_panel
    pct_reduction metric exactly: (ctl - unc) / unc * 100. When control
    REDUCES flutter (ctl < unc, the good outcome), this is NEGATIVE —
    same convention used in every other figure (fig1, fig3, fig4). Do
    NOT flip this without also flipping the other figures' formulas, or
    fig5 will read backwards relative to everything else again.
    """
    unc = np.array(results['uncontrolled']['energy_flutter'])
    ctl = np.array(results['controlled']['energy_flutter'])
    if len(unc) == 0 or len(ctl) == 0:
        return np.array([])
    if len(unc) == len(ctl):
        return (ctl - unc) / unc * 100
    # Unpaired/mismatched lengths (shouldn't normally happen here, but
    # fall back to comparing means as a single-sample array rather than
    # crashing) — matches the same fallback fig3's pct_reduction metric
    # uses for the analogous case.
    mean_unc, mean_ctl = np.mean(unc), np.mean(ctl)
    return np.array([(mean_ctl - mean_unc) / mean_unc * 100]) if mean_unc > 0 else np.array([])


def load_psd_based_reduction_data(jitter_fft_results=None):
    """
    Builds the same four data structures draw_overlaid_panel() expects
    (jitter_data, fixed_ol_data, cl_ref, ol55_ref) — but computed from the
    actual bag files via the same Welch PSD-based energy_flutter
    calculation used in fig1/fig3/fig4, instead of reading separately
    pre-computed percentages from JSON files. This keeps the "%
    reduction" numbers in fig5's overlaid panel methodologically
    consistent with every other figure.

    jitter_fft_results: optionally pass in the dict already returned by
    load_jitter_fft_data() (which fig5's FFT row loads anyway) to avoid
    reprocessing those bags a second time. If not given, loads them fresh.
    """
    # --- Jitter sweep frequencies: reuse (or load) the same
    # process_condition_fft results already used for the FFT row ---
    if jitter_fft_results is None:
        jitter_fft_results = load_jitter_fft_data()

    jitter_data = {}
    for freq, results in jitter_fft_results.items():
        pcts = _pct_reduction_from_results(results)
        if len(pcts):
            jitter_data[freq] = pcts
    if 6.0 in jitter_data and 5.5 not in jitter_data:
        jitter_data[5.5] = jitter_data.pop(6.0)
        jitter_data = dict(sorted(jitter_data.items()))

    # --- Fixed open-loop conditions (3/8/10 Hz): these bags follow the
    # multi-trial 120s-off/120s-on x3 protocol, same as fig3's 3HZ/8HZ/10HZ
    # conditions, so reuse fig3's process_multi_trial_bag directly ---
    FIXED_OL_BAG_PATTERNS = {
        3.0:  (BASE_PATH / '3hz' / 'bags',  'off_freq_3hz_run*.bag'),
        8.0:  (BASE_PATH / '8hz' / 'bags',  'off_freq_8hz_run*.bag'),
        10.0: (BASE_PATH / '10hz' / 'bags', 'off_freq_10hz_run*.bag'),
    }
    fixed_ol_data = {}
    for freq, (bag_dir, pattern) in FIXED_OL_BAG_PATTERNS.items():
        bag_paths = sorted(glob.glob(os.path.join(str(bag_dir), pattern)))
        if not bag_paths:
            print(f'[Fixed OL {freq} Hz] No bags found matching {pattern} in {bag_dir}')
            continue
        print(f'[Fixed OL {freq} Hz] {len(bag_paths)} bags')
        results = process_multi_trial_bag(bag_paths, expected_trials=3)
        pcts = _pct_reduction_from_results(results)
        if len(pcts):
            fixed_ol_data[freq] = pcts

    # --- Closed Loop / Open Loop 5.5 Hz reference (from OL_RDM bags,
    # standard 60s/60s split via process_condition_fft) ---
    cl_bags = sorted(glob.glob(os.path.join(str(OL_RDM_DIR), 'CLOSED_LOOP_sample*.bag')))
    ol55_bags = sorted(glob.glob(os.path.join(str(OL_RDM_DIR), 'OPEN_LOOP_55_sample*.bag')))

    cl_ref = None
    if cl_bags:
        print(f'[Closed Loop ref] {len(cl_bags)} bags')
        cl_results = process_condition_fft(cl_bags, is_closed_loop=True)
        pcts = _pct_reduction_from_results(cl_results)
        cl_ref = pcts if len(pcts) else None

    ol55_ref = None
    if ol55_bags:
        print(f'[Open Loop 5.5 Hz ref] {len(ol55_bags)} bags')
        ol55_results = process_condition_fft(ol55_bags, is_closed_loop=False)
        pcts = _pct_reduction_from_results(ol55_results)
        ol55_ref = pcts if len(pcts) else None

    return jitter_data, fixed_ol_data, cl_ref, ol55_ref


def draw_jitter_fft_row(axes_list, jitter_fft_results, frequencies=JITTER_FREQUENCIES,
                         f_max=25.0, scale=1.0, show_legend=True, show_inset=False,
                         inset_freq_range=(4.0, 6.0), inset_bbox=(0.5, 0.5, 0.44, 0.44),
                         inset_show_ticks=True):
    """
    Draws one uncontrolled-vs-controlled FFT overlay per jitter frequency
    into axes_list. Grey uncontrolled is drawn last/on top.

    show_legend: if False, omits the Controlled/Uncontrolled legend on
    the first panel. Defaults to True.

    show_inset: if True, adds a small zoomed-in inset to each panel (in
    its own upper-right corner) showing a magnified view of the flutter-
    band peak region, matching the inset style used in fig1/fig3.
    Defaults to False.

    inset_freq_range: (f_low, f_high) for the inset's x-axis. Defaults to
    exactly the flutter band (4-6 Hz).

    inset_bbox: (x0, y0, width, height) in axes-fraction coordinates for
    where each inset sits.

    inset_show_ticks: if False, hides the tick number labels on the
    inset's axes (keeping the small tick marks themselves).
    """
    for ax, freq in zip(axes_list, frequencies):
        if freq not in jitter_fft_results:
            ax.text(0.5, 0.5, f'No data\n({freq} Hz)', ha='center', va='center',
                    transform=ax.transAxes, color='#aaa', fontsize=6)
            continue

        res = jitter_fft_results[freq]

        ctl_freqs = res['controlled']['freqs']
        ctl_mags  = res['controlled']['mags']
        mean_ctl = f_ref_ctl = None
        if ctl_freqs:
            f_ref_ctl = ctl_freqs[0][ctl_freqs[0] <= f_max]
            mags_i = [np.interp(f_ref_ctl, f, m) for f, m in zip(ctl_freqs, ctl_mags)]
            mean_ctl = np.mean(mags_i, axis=0)
            std_ctl  = np.std(mags_i, axis=0)
            ax.plot(f_ref_ctl, mean_ctl, color=COLOR_JITTER, lw=1.5 * scale,
                    label='Controlled', zorder=3)
            ax.fill_between(f_ref_ctl, mean_ctl - std_ctl, mean_ctl + std_ctl,
                            color=COLOR_JITTER, alpha=0.15, zorder=4)

        unc_freqs = res['uncontrolled']['freqs']
        unc_mags  = res['uncontrolled']['mags']
        mean_unc = f_ref_unc = None
        if unc_freqs:
            f_ref_unc = unc_freqs[0][unc_freqs[0] <= f_max]
            mags_i = [np.interp(f_ref_unc, f, m) for f, m in zip(unc_freqs, unc_mags)]
            mean_unc = np.mean(mags_i, axis=0)
            ax.plot(f_ref_unc, mean_unc, color='#555', lw=1.5 * scale, ls='--',
                    label='Uncontrolled', zorder=5, alpha=0.8)

        ax.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
        ax.axvline(FLUTTER_FREQ, color='#e65100', lw=0.6 * scale, ls=':', alpha=0.7, zorder=1)

        # Peak-magnitude labels — same fixed x-position and dynamic
        # top/bottom ordering (larger peak gets the higher slot,
        # regardless of which trace it belongs to) as
        # draw_single_condition_fft_overlay in plot_fft_energy.py and
        # fig3's draw_fft_row, so all figures read consistently.
        traces = []
        if mean_ctl is not None:
            traces.append((COLOR_JITTER, f_ref_ctl, mean_ctl))
        if mean_unc is not None:
            traces.append(('#555', f_ref_unc, mean_unc))

        label_x = 0.8
        y_slots = [0.70, 0.60]
        peak_info = []
        for trace_color, f_arr, m_arr in traces:
            pi = np.argmax(m_arr)
            peak_info.append((trace_color, f_arr[pi], m_arr[pi]))
        peak_info.sort(key=lambda p: p[2], reverse=True)
        for (trace_color, peak_f, peak_mag), y_frac in zip(peak_info, y_slots):
            ax.annotate(f'{peak_mag:.2f}',
                        xy=(peak_f, peak_mag), xycoords='data',
                        xytext=(label_x, y_frac), textcoords=('data', 'axes fraction'),
                        fontsize=6, color=trace_color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=trace_color, lw=0.8 * scale))

        ax.set_xlim(0, f_max)
        ax.set_ylim(bottom=0)
        ax.set_title(f'{freq:.0f} Hz' if freq == int(freq) else f'{freq:.1f} Hz',
                     fontsize=8, color=COLOR_JITTER, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=8, labelpad=2)
        ax.tick_params(axis='both', labelsize=6 * scale, pad=2)
        fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                         tick_length=2.5 * scale,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         linewidth=0.5 * scale)
        ax.grid(True, alpha=0.2)

        if show_inset and mean_ctl is not None and mean_unc is not None:
            axins = ax.inset_axes(inset_bbox)
            mask_ctl = (f_ref_ctl >= inset_freq_range[0]) & (f_ref_ctl <= inset_freq_range[1])
            mask_unc = (f_ref_unc >= inset_freq_range[0]) & (f_ref_unc <= inset_freq_range[1])
            axins.plot(f_ref_ctl[mask_ctl], mean_ctl[mask_ctl], color=COLOR_JITTER, lw=1.2 * scale)
            axins.plot(f_ref_unc[mask_unc], mean_unc[mask_unc], color='#555', lw=1.2 * scale, ls='--')
            axins.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
            axins.set_xlim(*inset_freq_range)
            if inset_show_ticks:
                axins.tick_params(labelsize=6)
            else:
                axins.tick_params(labelsize=6, labelleft=False, labelbottom=False)
            fifi.mpl_functions.adjust_spines(axins, ['left', 'bottom'],
                                             tick_length=2.0 * scale,
                                             spine_locations={'left': 3, 'bottom': 3},
                                             linewidth=0.5 * scale)
            axins.set_facecolor('white')
            axins.patch.set_alpha(0.95)

    if len(axes_list) > 0:
        axes_list[0].set_ylabel('FFT Magnitude', fontsize=8, labelpad=2)
        if show_legend:
            axes_list[0].legend(fontsize=6, loc='upper right', framealpha=0.9)


def plot_jitter_fft_row(jitter_fft_results, output_dir):
    """Standalone verification figure: all 4 jitter FFT panels side by side."""
    fig, axes = plt.subplots(1, len(JITTER_FREQUENCIES), figsize=(4 * len(JITTER_FREQUENCIES), 3.5),
                             sharey=True)
    draw_jitter_fft_row(axes, jitter_fft_results)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'jitter_fft_row.{ext}'),
                    dpi=200, bbox_inches='tight')
    print("[SAVED] jitter_fft_row.png / .pdf")
    plt.close(fig)


def main():
    output_dir = str(OUTPUT_DIR)

    print("Loading Uniform Jitter FFT data from bags (also reused for Portion 1's PSD-based % reduction):")
    jitter_fft_results = load_jitter_fft_data()

    print("\nComputing PSD-based % reduction for all four data sources (Portion 1):")
    jitter_data, fixed_ol_data, cl_ref, ol55_ref = load_psd_based_reduction_data(jitter_fft_results)

    if jitter_data:
        plot_overlaid(jitter_data, fixed_ol_data, cl_ref, ol55_ref, output_dir)
    else:
        print("[WARN] No jitter percent-reduction data found — skipping Portion 1 standalone check.")

    if jitter_fft_results:
        plot_jitter_fft_row(jitter_fft_results, output_dir)
    else:
        print("[WARN] No jitter FFT bag data found — skipping Portion 2 standalone check.")

    print("\nDone.")


if __name__ == '__main__':
    main()