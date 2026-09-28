#!/usr/bin/env python3
"""
fig3_frequency_profile.py
==========================
Frequency-profile comparison across 3 Hz, 5.5 Hz (open loop), 8 Hz, 10 Hz,
and closed loop ("freq tracked").

Two reusable pieces, matching the pattern used in plot_fft_energy.py and
plot_time_trace_updated.py:

  draw_fft_row(fig_or_axes, all_results, conditions, f_max=25.0, scale=1.0)
      Draws one uncontrolled-vs-controlled FFT overlay panel per condition,
      side by side, into a list of axes you provide (e.g. five
      figurefirst-tagged axes for a "side by side" layout).

  draw_frequency_triptych_panel(ax, all_results, conditions, metric, scale=1.0)
      Draws ONE panel of the triptych (metric is 'energy_total',
      'energy_flutter', or 'pct_reduction') across all five conditions.
      Call it three times, once per metric, into three separate axes.

Data loading reuses process_condition_fft(), compute_fft(), and
total_energy() from plot_fft_energy.py in OL_RDM — this script imports them
rather than redefining, so both scripts stay in sync on how FFTs are
computed.

Usage (standalone):
    python3 fig3_frequency_profile.py
"""

import sys
import os
import glob
import numpy as np
import matplotlib
import figurefirst as fifi
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Reuse the core FFT machinery from the six-condition script rather than
# duplicating it. Adjust this path if OL_RDM moves.
from .fft_energy import (process_condition_fft, read_optic_flow_segment, compute_fft,
                              total_energy, compute_psd_welch, total_power_psd,
                              parse_sample_date, TYPESTORE, FLUTTER_BAND,
                              FLUTTER_FREQ, CLOSED_LOOP_PATTERN)
from rosbags.rosbag1 import Reader

from .config import DATA_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Condition definitions for this figure
# ---------------------------------------------------------------------------
# Each entry: (key, label, color, bag_dir, glob_pattern, is_closed_loop, n_trials)
# n_trials=None means use the standard single 60s-unc/60s-ctl extraction
# (process_condition_fft). A number (e.g. 3) means the bag contains that
# many repeated off/on cycles and every one should be extracted — but since
# the real on/off timing turned out not to follow a clean fixed 120s/120s
# pattern (confirmed by inspecting /piezo_voltages directly), segments are
# now detected from actual piezo activity rather than assumed durations.
# n_trials is kept as an expected-count sanity check, not a hard duration.
#
# Colors: all four "open loop" conditions (3/5.5/8/10 Hz — fixed-frequency
# actuation with no phase-lock feedback) share ONE blue, since they're all
# the same underlying condition type just at different frequencies. Freq
# Tracked (closed loop) keeps its own green, since it's a fundamentally
# different condition (phase-locked feedback), not just another frequency.
FREQ_CONDITIONS = [
    ('3HZ',    '3 Hz',        '#2196F3',
     'off_frequency_tests/3hz/bags',   'off_freq_3hz_run*.bag',  False, 3),
    ('OL_55',  '5.5 Hz',      '#2196F3',
     'off_frequency_tests/OL_RDM',     'OPEN_LOOP_55_sample*.bag', False, None),
    ('TRACKED','Freq Tracked','#4CAF50',
     'off_frequency_tests/OL_RDM',     CLOSED_LOOP_PATTERN,      True, None),
    ('8HZ',    '8 Hz',        '#2196F3',
     'off_frequency_tests/8hz/bags',   'off_freq_8hz_run*.bag',  False, 3),
    ('10HZ',   '10 Hz',       '#2196F3',
     'off_frequency_tests/10hz/bags',  'off_freq_10hz_run*.bag', False, 3),
]


def find_first_transition(bag_path, window_size=2.0, percentile_split=(15, 85)):
    """
    Finds the timestamp of the first off->on transition using rolling
    standard deviation of /piezo_voltages. This is combined with the
    known fixed protocol timing from test_off_frequency_control.py (120s
    off / 120s on x3, off period first) to compute all six segment
    boundaries from one reliable anchor point, rather than trying to
    detect every subsequent transition independently.
    """
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
    v0 = np.array(v0)
    t_max = times[-1]

    bins = np.arange(0, t_max, window_size)
    stds = np.array([
        np.std(v0[(times >= b) & (times < b + window_size)])
        if np.any((times >= b) & (times < b + window_size)) else 0.0
        for b in bins
    ])

    lo, hi = np.percentile(stds, percentile_split[0]), np.percentile(stds, percentile_split[1])
    threshold = (lo + hi) / 2
    is_on = stds > threshold

    on_indices = np.where(is_on)[0]
    if len(on_indices) == 0:
        return None
    return bins[on_indices[0]]


def compute_protocol_segments(bag_path, cycle_duration=120, num_cycles=3):
    """
    Computes the 6 segment boundaries (off1, on1, off2, on2, off3, on3)
    for a bag recorded per test_off_frequency_control.py's protocol:
    control OFF first, then ON, each lasting cycle_duration seconds,
    repeated num_cycles times.

    Anchors the whole sequence to the bag's actual first detected
    off->on transition (find_first_transition) rather than assuming the
    protocol started exactly at the bag's t=0 — since there's typically
    some recording pre-roll before the script's internal timer begins
    (node init, the "press Enter to start" prompt, etc.).

    Returns a list of (label, t_start, t_end) tuples, or None if no
    transition could be detected.
    """
    first_on_start = find_first_transition(bag_path)
    if first_on_start is None:
        return None

    # first_on_start marks the end of off1 / start of on1
    t0 = first_on_start - cycle_duration

    segments = []
    for cycle in range(num_cycles):
        off_start = t0 + cycle * 2 * cycle_duration
        off_end   = off_start + cycle_duration
        on_start  = off_end
        on_end    = on_start + cycle_duration
        segments.append(('uncontrolled', off_start, off_end))
        segments.append(('controlled', on_start, on_end))

    return segments


def process_multi_trial_bag(bag_paths, expected_trials=3):
    """
    For bags recorded per the fixed off/on protocol (120s off / 120s on,
    repeated 3x), computes segment boundaries via compute_protocol_segments
    (anchored to the bag's actual first detected transition, then fixed
    120s durations for everything after) and extracts each as a trial.

    Returns the same 'uncontrolled'/'controlled' dict structure that
    process_condition_fft() produces, so it plugs into the same plotting
    functions without any changes on that end.
    """
    results = {
        'uncontrolled': {'freqs': [], 'mags': [], 'energy_total': [], 'energy_flutter': [], 'dates': []},
        'controlled':   {'freqs': [], 'mags': [], 'energy_total': [], 'energy_flutter': [], 'dates': []},
    }
    for bag_path in bag_paths:
        date_str = parse_sample_date(bag_path)
        segments = compute_protocol_segments(bag_path, cycle_duration=120, num_cycles=expected_trials)
        if segments is None:
            print(f"    [WARN] {os.path.basename(bag_path)}: could not detect first transition, skipping")
            continue

        print(f"    {os.path.basename(bag_path)}: anchored segments")
        for label, t0, t1 in segments:
            print(f"      {label}: {t0:.1f}s - {t1:.1f}s")

        for seg_name, t0, t1 in segments:
            if t0 < 0:
                print(f"      [SKIP] {seg_name} {t0:.1f}-{t1:.1f}s: starts before bag recording began")
                continue
            times, signal = read_optic_flow_segment(bag_path, t0, t1)
            if times is None or len(times) < 100:
                print(f"      [SKIP] {seg_name} {t0:.1f}-{t1:.1f}s: insufficient data")
                continue
            freqs, mags, fs = compute_fft(times, signal)
            if freqs is None:
                continue

            # Same switch as process_condition_fft in plot_fft_energy.py:
            # summary power numbers now use a proper Welch PSD estimate
            # rather than integrating raw FFT magnitude squared. The
            # visual spectrum curves (freqs, mags) are untouched.
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
    return results


def load_frequency_profile_data(base_path=None):
    """
    Loads and processes bags for all five conditions in FREQ_CONDITIONS,
    returning a dict keyed by condition key -> process_condition_fft()
    result dict (with 'uncontrolled'/'controlled' sub-dicts containing
    freqs, mags, energy_total, energy_flutter, dates).

    base_path: root directory containing figures_for_dissertation
    (defaults to the standard dissertation path).
    """
    if base_path is None:
        base_path = str(DATA_DIR)

    all_results = {}
    for key, label, color, rel_dir, pattern, is_closed_loop, n_trials in FREQ_CONDITIONS:
        bag_dir = os.path.join(base_path, rel_dir)
        bag_paths = sorted(glob.glob(os.path.join(bag_dir, pattern)))
        if not bag_paths:
            print(f'[{key}] No bags found in {bag_dir} matching {pattern}')
            continue
        print(f'[{key}] {len(bag_paths)} bags')
        if n_trials is not None:
            all_results[key] = process_multi_trial_bag(bag_paths, expected_trials=n_trials)
        else:
            all_results[key] = process_condition_fft(bag_paths, is_closed_loop=is_closed_loop)

    return all_results


# ---------------------------------------------------------------------------
# Portion 1: side-by-side FFT overlays
# ---------------------------------------------------------------------------
def draw_fft_row(axes_list, all_results, conditions=FREQ_CONDITIONS, f_max=25.0,
                  scale=1.0, normalize=True, show_inset=False, show_legend=True,
                  inset_freq_range=(4.0, 6.0), inset_bbox=(0.5, 0.5, 0.44, 0.44),
                  inset_show_ticks=True):
    """
    Draws one uncontrolled-vs-controlled FFT overlay per condition into
    axes_list (a list of matplotlib Axes, one per condition, in order).

    normalize: if True (default), each condition's spectra are divided by
    that condition's OWN uncontrolled peak magnitude before plotting. Since
    these bags were recorded across different sessions/dates, raw absolute
    magnitude may not be directly comparable (sensor calibration, lighting,
    etc. can drift over time) — normalizing to each condition's own
    baseline keeps the comparison about relative shape/reduction rather
    than absolute scale. Set False to plot raw magnitudes and share a
    single y-axis across panels if you've verified scales are comparable.

    show_inset: if True, adds a small zoomed-in inset to each panel (in
    its own upper-right corner) showing a magnified view of the flutter-
    band peak region, matching the inset style used in fig1's single-
    condition FFT overlay. Defaults to False.

    show_legend: if False, omits the Controlled/Uncontrolled legend on
    the first panel. Defaults to True.

    inset_freq_range: (f_low, f_high) for the inset's x-axis. Defaults to
    exactly the flutter band (4-6 Hz).

    inset_bbox: (x0, y0, width, height) in axes-fraction coordinates for
    where each inset sits.

    inset_show_ticks: if False, hides the tick number labels on the
    inset's axes (keeping the small tick marks themselves) — useful when
    the inset is small enough that numbers add clutter rather than
    clarity. Defaults to True.
    """
    center_idx = len(axes_list) // 2
    for i, (ax, (key, label, color, *_ )) in enumerate(zip(axes_list, conditions)):
        if key not in all_results:
            ax.text(0.5, 0.5, f'No data\n({label})', ha='center', va='center',
                    transform=ax.transAxes, color='#aaa', fontsize=6)
            continue

        res = all_results[key]
        norm_factor = 1.0

        unc_freqs = res['uncontrolled']['freqs']
        unc_mags  = res['uncontrolled']['mags']
        mean_unc = f_ref_unc = None
        if unc_freqs:
            f_ref_unc = unc_freqs[0][unc_freqs[0] <= f_max]
            mags_i = [np.interp(f_ref_unc, f, m) for f, m in zip(unc_freqs, unc_mags)]
            mean_unc = np.mean(mags_i, axis=0)
            if normalize:
                norm_factor = np.max(mean_unc) if np.max(mean_unc) > 0 else 1.0

        # Controlled drawn FIRST/behind, so the grey uncontrolled trace
        # ends up visually in front of it (matches plot_fft_energy.py's
        # draw_single_condition_fft_overlay convention).
        ctl_freqs = res['controlled']['freqs']
        ctl_mags  = res['controlled']['mags']
        mean_ctl = std_ctl = f_ref = None
        if ctl_freqs:
            f_ref = ctl_freqs[0][ctl_freqs[0] <= f_max]
            mags_i = [np.interp(f_ref, f, m) for f, m in zip(ctl_freqs, ctl_mags)]
            mean_ctl = np.mean(mags_i, axis=0)
            std_ctl  = np.std(mags_i, axis=0)
            ax.plot(f_ref, mean_ctl / norm_factor, color=color, lw=1.5 * scale,
                    label='Controlled', zorder=3)
            ax.fill_between(f_ref, (mean_ctl - std_ctl) / norm_factor,
                            (mean_ctl + std_ctl) / norm_factor,
                            color=color, alpha=0.15, zorder=4)

        # Uncontrolled — drawn LAST/on top, per request.
        if mean_unc is not None:
            ax.plot(f_ref_unc, mean_unc / norm_factor, color='#555', lw=1.5 * scale, ls='--',
                    label='Uncontrolled', zorder=5, alpha=0.8)

        ax.axvspan(*FLUTTER_BAND, color='#ffe0b2', alpha=0.3, zorder=0)
        ax.axvline(FLUTTER_FREQ, color='#e65100', lw=0.6 * scale, ls=':', alpha=0.7, zorder=1)

        # Peak-magnitude labels — same fixed x-position and dynamic
        # top/bottom ordering (larger peak gets the higher slot,
        # regardless of which trace it belongs to) as
        # draw_single_condition_fft_overlay in plot_fft_energy.py, so
        # both figures read consistently.
        traces = []
        if mean_ctl is not None:
            traces.append((color, f_ref, mean_ctl / norm_factor))
        if mean_unc is not None:
            traces.append(('#555', f_ref_unc, mean_unc / norm_factor))

        label_x = 0.8
        y_slots = [0.70, 0.60]
        peak_info = []
        for trace_color, f_arr, m_arr in traces:
            pi = np.argmax(m_arr)
            peak_info.append((trace_color, f_arr[pi], m_arr[pi]))
        peak_info.sort(key=lambda p: p[2], reverse=True)  # largest peak first
        for (trace_color, peak_f, peak_mag), y_frac in zip(peak_info, y_slots):
            ax.annotate(f'{peak_mag:.2f}',
                        xy=(peak_f, peak_mag), xycoords='data',
                        xytext=(label_x, y_frac), textcoords=('data', 'axes fraction'),
                        fontsize=6, color=trace_color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=trace_color, lw=0.8 * scale))

        ax.set_xlim(0, f_max)
        ax.set_ylim(bottom=0)
        ax.set_title(label, fontsize=8, color=color, fontweight='bold')
        if i == center_idx:
            ax.set_xlabel('Frequency (Hz)', fontsize=8, labelpad=2)
        ax.tick_params(axis='both', labelsize=7 * scale, pad=2)
        fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                         tick_length=2.5 * scale,
                                         spine_locations={'left': 5, 'bottom': 5},
                                         linewidth=0.5 * scale)
        ax.grid(True, alpha=0.2)

        if show_inset and mean_ctl is not None and mean_unc is not None:
            axins = ax.inset_axes(inset_bbox)
            mask_ctl = (f_ref >= inset_freq_range[0]) & (f_ref <= inset_freq_range[1])
            mask_unc = (f_ref_unc >= inset_freq_range[0]) & (f_ref_unc <= inset_freq_range[1])
            axins.plot(f_ref[mask_ctl], mean_ctl[mask_ctl] / norm_factor,
                      color=color, lw=1.2 * scale)
            axins.plot(f_ref_unc[mask_unc], mean_unc[mask_unc] / norm_factor,
                      color='#555', lw=1.2 * scale, ls='--')
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

    # Only the leftmost panel gets a y-label + legend, to avoid repeating
    # the same two labels five times across a tight side-by-side row.
    # Kept short (rather than the full "normalized to each condition's
    # own uncontrolled peak" explanation) so it doesn't overlap/clip
    # against the panel's left edge at small box sizes.
    if len(axes_list) > 0:
        ylabel = 'FFT Mag. (norm.)' if normalize else 'FFT Magnitude'
        axes_list[0].set_ylabel(ylabel, fontsize=8, labelpad=2)
        if show_legend:
            axes_list[0].legend(fontsize=6, loc='upper right', framealpha=0.9)


# ---------------------------------------------------------------------------
# Portion 2: triptych panels (total power / band power / % change)
# ---------------------------------------------------------------------------
def draw_frequency_triptych_panel(ax, all_results, conditions=FREQ_CONDITIONS,
                                   metric='energy_total', scale=1.0, normalize=True,
                                   xtick_fontsize=6):
    """
    Draws one panel of the triptych across all conditions in `conditions`.

    metric:
      'energy_total'   -> jittered scatter of total spectral energy, unc vs ctl
      'energy_flutter' -> same, but for the 4-6 Hz flutter band
      'pct_reduction'  -> scatter of % change in flutter-band (4-6 Hz)
                          energy (ctl vs unc) per condition, with significance stars (always self-normalized,
                          unaffected by the `normalize` flag). Despite the
                          parameter name (kept for backward compatibility),
                          this is genuinely a CHANGE metric, not strictly a
                          reduction — it can be positive (energy increased)
                          or negative (energy decreased), which is why the
                          panel's title/ylabel say "% Change" rather than
                          "% Reduction".

    normalize: for the two energy scatter metrics only, divides each
    condition's uncontrolled AND controlled values by that condition's own
    uncontrolled mean. Since these conditions were recorded in different
    sessions/dates, raw absolute energy units may not be directly
    comparable across conditions (calibration drift, lighting, etc.) —
    normalizing puts everything on a common "fraction of own baseline"
    scale so the comparison is about relative reduction, not absolute
    magnitude. pct_reduction is unaffected since it's already a ratio.

    xtick_fontsize: fontsize for the x-axis condition labels. Defaults to
    6 (matching the full condition names used in fig3). Bump this up for
    shorter abbreviated labels (e.g. fig4's "CL", "OL", "P") where there's
    room for larger text.
    """
    valid = [(key, label, color) for key, label, color, *_ in conditions if key in all_results]
    n = len(valid)
    np.random.seed(42)

    if metric in ('energy_total', 'energy_flutter'):
        for i, (key, label, color) in enumerate(valid):
            res = all_results[key]
            unc = np.array(res['uncontrolled'][metric])
            ctl = np.array(res['controlled'][metric])

            if normalize:
                baseline = np.mean(unc) if len(unc) and np.mean(unc) > 0 else 1.0
                unc = unc / baseline
                ctl = ctl / baseline

            x_unc = i - 0.18
            x_ctl = i + 0.18

            j_unc = np.random.normal(0, 0.04, len(unc))
            ax.scatter(x_unc + j_unc, unc, color='#888', alpha=0.7, s=30 * scale, zorder=3)
            ax.plot([x_unc - 0.1, x_unc + 0.1], [np.mean(unc), np.mean(unc)],
                    color='#444', lw=1.8 * scale, zorder=4)

            j_ctl = np.random.normal(0, 0.04, len(ctl))
            ax.scatter(x_ctl + j_ctl, ctl, color=color, alpha=0.7, s=30 * scale, zorder=3)
            ax.plot([x_ctl - 0.1, x_ctl + 0.1], [np.mean(ctl), np.mean(ctl)],
                    color=color, lw=1.8 * scale, zorder=4)

        ax.set_xticks(range(n))
        ax.set_xticklabels([label.replace('Freq Tracked', 'Freq\nTracked') for _, label, _ in valid], fontsize=xtick_fontsize, rotation=0)
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', labelsize=7 * scale, pad=2)
        if normalize:
            ylabel = ('Relative Total Power (PSD)' if metric == 'energy_total'
                      else f'Relative {FLUTTER_BAND[0]:.0f}\u2013{FLUTTER_BAND[1]:.0f} Hz Power (PSD)')
            if metric == 'energy_total':
                ax.axhline(1.0, color='black', lw=0.6 * scale, ls=':', alpha=0.5, zorder=1)
        else:
            ylabel = 'Total Energy (a.u.)' if metric == 'energy_total' else \
                     f'{FLUTTER_BAND[0]:.0f}\u2013{FLUTTER_BAND[1]:.0f} Hz Power (PSD, a.u.)'
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax.set_title(ylabel.split('\n')[0], fontsize=8, fontweight='bold')

    elif metric == 'pct_reduction':
        # Per-sample % change (paired unc[i]/ctl[i], matching the paired
        # t-test already used below) rather than a single aggregate number
        # per condition — this gives an actual scatter distribution,
        # matching the jittered-scatter style used in figs 4/5, instead of
        # one flat bar per condition.
        per_cond_pcts = []
        sigs = []
        for key, label, color in valid:
            res = all_results[key]
            unc = np.array(res['uncontrolled']['energy_flutter'])
            ctl = np.array(res['controlled']['energy_flutter'])
            if len(unc) == len(ctl) and len(unc) > 0:
                pcts_i = (ctl - unc) / unc * 100
            else:
                mean_unc = np.mean(unc) if len(unc) else np.nan
                mean_ctl = np.mean(ctl) if len(ctl) else np.nan
                pcts_i = np.array([(mean_ctl - mean_unc) / mean_unc * 100]) if mean_unc > 0 else np.array([np.nan])
            per_cond_pcts.append(pcts_i)

            if len(unc) > 1 and len(ctl) > 1:
                _, p = stats.ttest_rel(unc, ctl) if len(unc) == len(ctl) else stats.ttest_ind(unc, ctl)
                sigs.append('***' if p < 0.001 else '**' if p < 0.01 else
                            '*'   if p < 0.05  else 'ns')
            else:
                sigs.append('')

        for i, ((key, label, color), pcts_i) in enumerate(zip(valid, per_cond_pcts)):
            finite = pcts_i[np.isfinite(pcts_i)]
            if len(finite) == 0:
                continue
            jitter = np.random.normal(0, 0.06, size=len(finite))
            ax.scatter(i + jitter, finite, color=color, alpha=0.7, s=30 * scale, zorder=3)
            ax.plot([i - 0.15, i + 0.15], [np.mean(finite), np.mean(finite)],
                    color=color, lw=1.8 * scale, zorder=4)

        # All condition means + significance placed together in one line
        # near the TOP of the panel (shared height, axes-fraction y),
        # matching draw_energy_panel's convention above, instead of a
        # label floating above/below each individual bar.
        for i, ((key, label, color), pcts_i, sig) in enumerate(zip(valid, per_cond_pcts, sigs)):
            finite = pcts_i[np.isfinite(pcts_i)]
            if len(finite) == 0:
                continue
            mean_pct = np.mean(finite)
            ax.text(i, 0.96, f'{mean_pct:+.1f}%\n{sig}', ha='center', va='top',
                    fontsize=6, fontweight='bold', color=color, clip_on=True,
                    transform=ax.get_xaxis_transform())

        ax.axhline(0, color='black', lw=0.8 * scale)
        ax.set_xticks(range(n))
        ax.set_xticklabels([label.replace('Freq Tracked', 'Freq\nTracked') for _, label, _ in valid], fontsize=xtick_fontsize)
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', labelsize=7 * scale, pad=2)
        ax.set_ylabel('Power Change (%)', fontsize=8, labelpad=2)
        ax.set_title(f'% Change ({FLUTTER_BAND[0]:.0f}\u2013{FLUTTER_BAND[1]:.0f} Hz Flutter Power, PSD)',
                     fontsize=8, fontweight='bold')

        # Explicit headroom so the shared top-line labels (at 96% of axes
        # height) never overlap the highest actual scatter point.
        all_finite = np.concatenate([p[np.isfinite(p)] for p in per_cond_pcts if np.any(np.isfinite(p))])
        if len(all_finite):
            y_min, y_max = np.min(all_finite), np.max(all_finite)
            span = y_max - y_min if y_max > y_min else 1.0
            ax.set_ylim(y_min - span * 0.15, y_max + span * 0.35)

    ax.grid(True, axis='y', alpha=0.25, linestyle=':')
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)
    ax.set_xlim(-0.6, n - 0.4)


def main():
    all_results = load_frequency_profile_data()
    if not all_results:
        print('[ERROR] No data loaded for any condition.')
        return

    output_dir = str(OUTPUT_DIR)

    # Portion 1: side-by-side FFT row
    fig1, axes1 = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    draw_fft_row(axes1, all_results, FREQ_CONDITIONS)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_fft_row.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_fft_row.pdf'), bbox_inches='tight')
    print('[SAVED] fig3_fft_row.png / .pdf')
    plt.close(fig1)

    # Portion 2: triptych
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    draw_frequency_triptych_panel(axes2[0], all_results, FREQ_CONDITIONS, metric='energy_total')
    draw_frequency_triptych_panel(axes2[1], all_results, FREQ_CONDITIONS, metric='energy_flutter')
    draw_frequency_triptych_panel(axes2[2], all_results, FREQ_CONDITIONS, metric='pct_reduction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_triptych.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_triptych.pdf'), bbox_inches='tight')
    print('[SAVED] fig3_triptych.png / .pdf')
    plt.close(fig2)


if __name__ == '__main__':
    main()