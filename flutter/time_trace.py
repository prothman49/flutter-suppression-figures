#!/usr/bin/env python3
"""
Updated Fig 4.10 - reads piezo voltage directly from ROS1 bag file
Generates full time trace + zoomed inset showing square wave character

Notes:
    draw_magnitude_panel() is factored out so it can be imported and drawn
    directly into a figurefirst-tagged axis (e.g. layout.axes['RT_Control'])
    from a notebook, using the exact same plotting logic as the top panel
    of the standalone 4-panel PNG/PDF this script produces. The other three
    panels (tracked frequency, piezo voltage, zoom inset) are standalone-only
    for now.
"""

import numpy as np
import figurefirst as fifi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from .config import TIME_TRACE_DIR, OUTPUT_DIR

BASE_PATH = TIME_TRACE_DIR
BAG_PATH  = TIME_TRACE_DIR / 'control_demo_4phase__2026-02-02-13-52-29.bag'
JSON_PATH = TIME_TRACE_DIR / 'advisor_demo_20260202_135712.json'

# Alignment: main control burst starts at bag relative 154.35s = JSON compressed 90s
BAG_UNIX_START     = 1770069156.82
BAG_CONTROL_START  = 154.35
JSON_CONTROL_START = 90.0
BAG_ARTIFACT_CUTOFF = 10.0

# Phase region colours — No Wind left white (no shading), Flutter (both
# uncontrolled segments) shown in a neutral gray as the "null/no control"
# condition, Control ON kept green — matching the gray-for-null,
# green-for-control convention used consistently across every other
# figure now (previously No Wind used a light gray fill here too).
PHASE_INFO = {
    1: ('No Wind (Baseline)',          'white'),
    2: ('Wind ON (Flutter)',           'gray'),
    3: ('Wind + Control (Suppression)','#4CAF50'),
    4: ('Wind ON (Flutter Returns)',   'gray'),
}


# ---------------------------------------------------------------------------
# Reusable top panel — magnitude trace with phase shading + avg annotations
# ---------------------------------------------------------------------------
def draw_magnitude_panel(ax, json_time, json_magnitudes, json_labels, phase_offset_deg,
                         scale=1.0, show_avg_labels=True, show_summary_box=True,
                         show_phase_label_box=True):
    """
    Draws the flutter-magnitude time trace (phase-shaded regions, per-phase
    average lines, phase labels) into whatever matplotlib Axes you give it.

    This is the reusable core of the top panel in the standalone 4-panel
    figure below — used both for the full PNG/PDF this script produces and
    for drawing the identical panel directly into a figurefirst-tagged axis
    from a notebook (e.g. layout.axes['RT_control']).

    scale: multiplier applied to all fontsizes, line widths, and text-box
    padding. Defaults to 1.0.

    show_avg_labels: if True (default), shows the "Avg: X.X" text box next
    to each phase's average line. Set False to hide these (e.g. per
    advisor request) while keeping the average line itself.

    show_summary_box: if True (default), shows the yellow summary box
    (baseline/control/reduction numbers) in the upper right. Set False to
    hide it.

    show_phase_label_box: if True (default), draws a white bordered box
    behind each phase label ("No Wind", "Flutter", "Control ON"). Set
    False to show just the colored text with no box.
    """
    # Phase shading + boundary lines
    for phase_num, (pname, color) in PHASE_INFO.items():
        mask = json_labels == phase_num
        if not np.any(mask):
            continue
        t_s = json_time[mask][0]
        t_e = json_time[mask][-1]
        ax.axvspan(t_s, t_e, alpha=0.3, color=color)
        ax.axvline(t_s, color='black', linestyle='--', linewidth=1.5 * scale, alpha=0.7)

    # Average lines and labels
    flutter_mask = (json_labels == 2) | (json_labels == 4)
    control_mask = json_labels == 3

    avg_flutter = np.mean(json_magnitudes[flutter_mask])
    avg_control = np.mean(json_magnitudes[control_mask])
    pct_red     = (avg_flutter - avg_control) / avg_flutter * 100

    for phase_num, (pname, color) in PHASE_INFO.items():
        mask = json_labels == phase_num
        if not np.any(mask):
            continue
        t_s   = json_time[mask][0]
        t_e   = json_time[mask][-1]
        avg   = np.mean(json_magnitudes[mask])
        mid_t = (t_s + t_e) / 2
        ax.hlines(avg, t_s, t_e, colors='red', linestyles='--', linewidth=2 * scale, alpha=0.8)
        if show_avg_labels:
            ax.text(mid_t, avg + 10, f'Avg: {avg:.1f}',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle=f'round,pad={0.4 * scale}', facecolor='white', edgecolor='red'))

    if show_summary_box:
        textstr = (f'Avg Magnitude Baseline Flutter: {avg_flutter:.1f}\n'
                   f'Avg Magnitude With Control: {avg_control:.1f}\n'
                   f'Flutter Reduction: {pct_red:.1f}%')
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes,
                fontsize=6, fontweight='bold', va='top', ha='right',
                bbox=dict(boxstyle=f'round,pad={0.8 * scale}', facecolor='yellow',
                          edgecolor='black', linewidth=2 * scale))

    ax.plot(json_time, json_magnitudes, 'b-', linewidth=0.8 * scale, alpha=0.8)
    ax.set_ylabel('Magnitude', fontsize=8, labelpad=2)
    ax.set_xlabel('Time (s)', fontsize=8, labelpad=2)
    ax.set_title(f'Flutter Control Demonstration (Phase Offset = {phase_offset_deg}°)',
                 fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(json_magnitudes) * 1.05])
    # Match xlim exactly to the same per-phase boundaries axvspan uses
    # above (first sample of the lowest present phase, last sample of the
    # highest present phase) rather than the raw valid-label mask over the
    # whole array — a few stray samples sharing a label but sitting past
    # the actual contiguous block (e.g. a sparse trailing straggler) could
    # otherwise stretch t_end well past where the shading/data visually end.
    present_phases = [p for p in PHASE_INFO if np.any(json_labels == p)]
    if present_phases:
        first_phase, last_phase = min(present_phases), max(present_phases)
        t_start = json_time[json_labels == first_phase][0]
        t_end   = json_time[json_labels == last_phase][-1]
    else:
        t_start, t_end = json_time[0], json_time[-1]
    ax.set_xlim([t_start, t_end])

    # Phase labels — anchored near the top of the AXES (fraction
    # coordinates for y), not a fixed data value like y=85. That fixed
    # value only worked because the data happened to peak below it; using
    # axes-fraction y keeps the labels just under the top border
    # regardless of the data's actual magnitude range.
    for phase_num in [1, 2, 3, 4]:
        mask = json_labels == phase_num
        if not np.any(mask):
            continue
        mid_t = (json_time[mask][0] + json_time[mask][-1]) / 2
        labels_map = {1: ('No Wind',    'black'),
                      2: ('Flutter',    'black'),
                      3: ('Control ON', 'green'),
                      4: ('Flutter',    'black')}
        txt, fc = labels_map[phase_num]
        label_kwargs = dict(
            bbox=dict(boxstyle=f'round,pad={0.3 * scale}', facecolor='white', edgecolor=fc)
        ) if show_phase_label_box else {}
        ax.text(mid_t, 0.92, txt, ha='center', va='top', color=fc,
                fontsize=6,
                transform=ax.get_xaxis_transform(), clip_on=True,
                **label_kwargs)

    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'],
                                     tick_length=2.5 * scale,
                                     spine_locations={'left': 5, 'bottom': 5},
                                     linewidth=0.5 * scale)

    # A spine's drawn LENGTH is tied to the Inkscape box's physical width,
    # not to xlim — xlim only controls what data range maps across that
    # fixed box, it doesn't shrink the spine itself. set_bounds explicitly
    # restricts where the spine is actually drawn to match the real data
    # range, regardless of how wide the box was drawn.
    ax.spines['bottom'].set_bounds(t_start, t_end)
    ax.spines['left'].set_bounds(*ax.get_ylim())


def load_time_trace_data():
    """
    Reads the bag file + JSON, does the time alignment, and returns
    everything needed to call draw_magnitude_panel() (or the other
    panels' plotting code) — separated out so a notebook can load the
    data once without re-running main()'s full plotting + save logic.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)

    piezo_times, piezo_v0, piezo_v1 = [], [], []
    env_times, env_vals = [], []
    flow_times, flow_vals = [], []
    phase_times, phase_vals = [], []

    print("Reading bag file...")
    with Reader(BAG_PATH) as reader:
        connections = {c.topic: c for c in reader.connections}
        print(f"Available topics: {list(connections.keys())}")

        for connection, timestamp, rawdata in reader.messages():
            t = timestamp / 1e9  # nanoseconds to seconds

            if connection.topic == '/piezo_voltages':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                if len(msg.data) >= 2:
                    piezo_times.append(t)
                    piezo_v0.append(msg.data[0])
                    piezo_v1.append(msg.data[1])

            elif connection.topic == '/amplitude_envelope':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                env_times.append(t)
                env_vals.append(msg.data)

            elif connection.topic == '/optic_flow_mean_vx':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                flow_times.append(t)
                flow_vals.append(msg.data)

            elif connection.topic == '/phase_mod':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                phase_times.append(t)
                phase_vals.append(msg.data)

    piezo_times = np.array(piezo_times)
    piezo_v0    = np.array(piezo_v0)
    piezo_v1    = np.array(piezo_v1)
    env_times   = np.array(env_times)
    env_vals    = np.array(env_vals)
    flow_times  = np.array(flow_times)
    flow_vals   = np.array(flow_vals)

    print(f"Piezo messages: {len(piezo_times)}, rate ~{len(piezo_times)/(piezo_times[-1]-piezo_times[0]):.1f} Hz")
    print(f"Envelope messages: {len(env_times)}")

    # Load JSON for phase labels and FFT magnitude
    import json
    with open(JSON_PATH, 'r') as f:
        jdata = json.load(f)

    json_timestamps  = np.array(jdata['timestamps'])
    json_magnitudes  = np.array(jdata['magnitudes'])
    json_labels      = np.array(jdata['phase_labels'])
    phase_offset_deg = jdata['phase_offset']

    # Build compressed time axis from JSON
    time_compressed = np.zeros_like(json_timestamps)
    offset = 0
    for phase_num in [1, 2, 3, 4]:
        mask = json_labels == phase_num
        if np.any(mask):
            pt = json_timestamps[mask]
            pd = pt[-1] - pt[0]
            time_compressed[mask] = np.linspace(offset, offset + pd, np.sum(mask))
            offset += pd

    json_time = time_compressed

    def bag_to_compressed(bag_t):
        """Map bag Unix timestamp to compressed JSON time axis."""
        bag_rel  = bag_t - BAG_UNIX_START
        json_compressed_t = bag_rel - BAG_CONTROL_START + JSON_CONTROL_START
        idx = np.searchsorted(json_time, json_compressed_t).clip(0, len(json_time) - 1)
        return json_time[idx]

    # Filter out early artifact burst
    piezo_rel   = piezo_times - BAG_UNIX_START
    valid_mask  = piezo_rel >= BAG_ARTIFACT_CUTOFF
    piezo_times = piezo_times[valid_mask]
    piezo_v0    = piezo_v0[valid_mask]
    piezo_v1    = piezo_v1[valid_mask]

    piezo_compressed = np.array([bag_to_compressed(t) for t in piezo_times])

    return {
        'json_time': json_time,
        'json_magnitudes': json_magnitudes,
        'json_labels': json_labels,
        'phase_offset_deg': phase_offset_deg,
        'piezo_compressed': piezo_compressed,
        'piezo_v0': piezo_v0,
        'piezo_v1': piezo_v1,
        'flow_times': flow_times,
        'flow_vals': flow_vals,
        'bag_to_compressed': bag_to_compressed,
    }


def main():
    data = load_time_trace_data()

    json_time        = data['json_time']
    json_magnitudes  = data['json_magnitudes']
    json_labels      = data['json_labels']
    phase_offset_deg = data['phase_offset_deg']
    piezo_compressed = data['piezo_compressed']
    piezo_v0         = data['piezo_v0']
    flow_times       = data['flow_times']
    flow_vals        = data['flow_vals']
    bag_to_compressed = data['bag_to_compressed']

    control_mask = json_labels == 3

    # Compute FFT magnitude from optic flow (same method as original)
    window_size = 800
    hop_size    = 50
    dom_freq, dom_mag, dom_time = [], [], []

    flow_vals_arr  = np.array(flow_vals)
    flow_times_arr = np.array(flow_times)

    for i in range(0, len(flow_vals_arr) - window_size, hop_size):
        window    = flow_vals_arr[i:i+window_size]
        t_window  = flow_times_arr[i:i+window_size]
        dt        = np.mean(np.diff(t_window))
        if dt <= 0:
            continue
        freqs = np.fft.rfftfreq(len(window), dt)
        fft   = np.abs(np.fft.rfft(window))
        mask  = (freqs > 1.0) & (freqs < 15.0)
        if np.any(mask):
            pi    = np.argmax(fft[mask])
            dom_freq.append(freqs[mask][pi])
            dom_mag.append(fft[mask][pi])
            dom_time.append(bag_to_compressed(np.mean(t_window)))

    dom_freq = np.array(dom_freq)
    dom_mag  = np.array(dom_mag)
    dom_time = np.array(dom_time)

    # Pre-compute zoom window
    ctrl_mask_pre = json_labels == 3
    t_ctrl_start  = json_time[ctrl_mask_pre][0]
    t_ctrl_end    = json_time[ctrl_mask_pre][-1]
    zoom_start    = t_ctrl_start + (t_ctrl_end - t_ctrl_start) * 0.3
    zoom_end      = zoom_start + 6.0

    # Build figure — 4 panels
    fig = plt.figure(figsize=(16, 14))
    gs_outer = gridspec.GridSpec(4, 1, figure=fig,
                                  height_ratios=[3, 1.5, 1, 1], hspace=0.45)

    ax1 = fig.add_subplot(gs_outer[0])                    # FFT magnitude
    ax3 = fig.add_subplot(gs_outer[1], sharex=ax1)        # tracked frequency
    ax2 = fig.add_subplot(gs_outer[2], sharex=ax1)        # piezo voltage
    ax4 = fig.add_subplot(gs_outer[3])                    # zoomed inset (no sharex)

    # Phase shading on ax2/ax3 only (ax1 gets it inside draw_magnitude_panel)
    for phase_num, (pname, color) in PHASE_INFO.items():
        mask = json_labels == phase_num
        if not np.any(mask):
            continue
        t_s = json_time[mask][0]
        t_e = json_time[mask][-1]
        for ax in [ax2, ax3]:
            ax.axvspan(t_s, t_e, alpha=0.3, color=color)

    # ── Top panel (reusable) ──────────────────────────────────────────────
    draw_magnitude_panel(ax1, json_time, json_magnitudes, json_labels, phase_offset_deg)

    # ── Tracked flutter frequency panel ──────────────────────────────────
    ax3.plot(dom_time, dom_freq, color='purple', linewidth=1.2, alpha=0.8,
             label='FFT-tracked flutter frequency')
    ax3.axhline(5.3, color='g', linestyle='--', linewidth=1.5,
                alpha=0.7, label='5.3 Hz reference')
    ax3.axhline(5.6, color='red', linestyle=':', linewidth=1.0,
                alpha=0.5, label='5.6 Hz reference')
    ax3.set_ylabel('Flutter Freq (Hz)', fontsize=8, labelpad=2)
    ax3.set_ylim([0, 12])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=6)

    ctrl_freq_mask = (dom_time >= json_time[control_mask][0]) & \
                     (dom_time <= json_time[control_mask][-1])
    if np.any(ctrl_freq_mask):
        mean_ctrl_freq = np.mean(dom_freq[ctrl_freq_mask])
        ctrl_mid_t = (json_time[control_mask][0] + json_time[control_mask][-1]) / 2
        ax3.text(ctrl_mid_t, 0.8, f'Mean: {mean_ctrl_freq:.2f} Hz',
                 ha='center', va='bottom', fontsize=6,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='purple'))

    # ── Piezo voltage panel ────────────────────────────────────────────────
    ax2.plot(piezo_compressed, piezo_v0, color='green', linewidth=1.0,
             alpha=0.9, label='Piezo 0 (0–4.2 V)', drawstyle='steps-post')
    ax2.set_ylabel('Control Voltage (V)', fontsize=8, labelpad=2)
    ax2.set_ylim([-0.5, 5.0])
    ax2.set_yticks([0, 2.1, 4.2])
    ax2.axvspan(zoom_start, zoom_end, alpha=0.4, color='yellow', zorder=5)
    ax2.annotate('zoom', xy=((zoom_start + zoom_end)/2, 4.0),
                 ha='center', va='center', fontsize=6, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', edgecolor='black'))
    ax2.set_xlabel('Time (seconds)', fontsize=8, labelpad=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=6)

    # ── Zoomed panel — reconstructed high-fidelity control signal ─────────
    zoom_mask = (piezo_compressed >= zoom_start) & (piezo_compressed <= zoom_end)

    if np.sum(zoom_mask) > 5:
        t_recon = np.arange(0, 6.0, 1/1000.0)
        t_recon_abs = t_recon + zoom_start

        freq_recon = np.interp(t_recon_abs, dom_time, dom_freq,
                               left=dom_freq[0], right=dom_freq[-1])

        gate_recon = np.interp(t_recon_abs, piezo_compressed,
                               (piezo_v0 > 0.1).astype(float))
        gate_recon = gate_recon > 0.5

        phase_offset_rad = phase_offset_deg * np.pi / 180.0
        dt_recon = 1/1000.0
        cum_phase = np.cumsum(2 * np.pi * freq_recon * dt_recon) + phase_offset_rad
        square = (np.sin(cum_phase) >= 0).astype(float) * 4.2

        square_gated = np.where(gate_recon, square, 0.0)

        ax4.plot(t_recon, square_gated, color='#4CAF50', linewidth=1.5,
                 drawstyle='steps-post', label='Piezo 0 (reconstructed)')
        ax4.set_xlim(0, 6.0)
        ax4.set_ylim([-0.3, 4.8])
        ax4.set_xlabel('Time (s)', fontsize=8, labelpad=2)
        ax4.set_ylabel('Voltage (V)', fontsize=8, labelpad=2)
        ax4.set_title('Control Signal Zoom — 6-second window (active control period)',
                      fontsize=8, fontweight='bold')
        ax4.axhline(0,   color='gray', linewidth=0.5, linestyle=':')
        ax4.axhline(4.2, color='gray', linewidth=0.5, linestyle=':')
        ax4.set_yticks([0, 2.1, 4.2])
        ax4.grid(True, alpha=0.3)

        zoom_freq_mask = (dom_time >= zoom_start) & (dom_time <= zoom_end)
        if np.any(zoom_freq_mask):
            mean_zoom_freq = np.mean(dom_freq[zoom_freq_mask])
            ax4.text(0.98, 0.85, f'Mean flutter freq: {mean_zoom_freq:.2f} Hz',
                    transform=ax4.transAxes, fontsize=6, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='green'))
    else:
        ax4.text(0.5, 0.5, 'Insufficient zoom data', transform=ax4.transAxes,
                 ha='center', va='center', fontsize=6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_4_10_updated.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_4_10_updated.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/fig_4_10_updated.png / .pdf")
    plt.show()


if __name__ == '__main__':
    main()