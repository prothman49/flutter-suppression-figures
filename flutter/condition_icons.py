#!/usr/bin/env python3
"""
condition_icons.py
====================
Generates small, clean waveform "icon" sketches for each of the six
experimental conditions, matching the timing/polarity logic actually
defined by each condition's equation (see Experimental Conditions table),
rather than just being decorative squiggles.

Each icon is a short synthetic square-wave-style trace styled minimally
(no axis ticks/labels, colored border matching the condition's color) so
it reads as a compact visual reference — the fig4 equivalent of the
"Visual Example" column in the conditions table.

Notes:
    draw_condition_icon() draws ONE icon into a given axis.
    draw_condition_icons_row() draws all six into a list of axes, in
    CONDITIONS order — reusable in a figurefirst-tagged row of 6 boxes.

Usage (standalone):
    python -m flutter.condition_icons
"""

import numpy as np
import matplotlib
import figurefirst as fifi
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import OUTPUT_DIR

CONDITIONS = ['CLOSED_LOOP', 'OPEN_LOOP_54', 'OPEN_LOOP_55', 'POISSON',
              'UNIFORM_JITTER', 'PHASE_RANDOMIZED', 'REPLAY']

# Trim to the six actually used in your dissertation figures (OPEN_LOOP_54
# doesn't appear in your six-condition set — swap in/out here if needed)
ICON_CONDITIONS = ['CLOSED_LOOP', 'OPEN_LOOP_55', 'POISSON',
                    'UNIFORM_JITTER', 'PHASE_RANDOMIZED', 'REPLAY']

CONDITION_LABELS = {
    'CLOSED_LOOP':      'Closed Loop',
    'OPEN_LOOP_54':     'Open Loop 5.4 Hz',
    'OPEN_LOOP_55':     'Open Loop',
    'POISSON':          'Poisson',
    'UNIFORM_JITTER':   'Uniform Jitter',
    'PHASE_RANDOMIZED': 'Polarity\nRandomized',
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

T_NOMINAL = 1 / 5.4  # ~185 ms nominal period, per the conditions table


def _bipolar_wave_from_boundaries(boundaries, t_end, polarities=None):
    """
    Builds a CONTINUOUS bipolar square wave (always at +1 or -1, never
    resting at zero) from a list of segment boundary times. Between each
    pair of consecutive boundaries, the signal holds one flat level; by
    default consecutive segments alternate sign, matching a true
    V(t) = Vmax*sgn(sin(...)) square wave. `polarities` lets a condition
    override this with an explicit per-segment sign instead of strict
    alternation (used for Polarity Randomized).

    This matches the actual reference icon style: irregular-timing
    conditions (Poisson, Uniform Jitter) are irregular WIDTHS of
    high/low segments, not gaps down to a zero baseline.
    """
    if polarities is None:
        polarities = [1 if i % 2 == 0 else -1 for i in range(len(boundaries))]

    t = np.linspace(0, t_end, 3000)
    y = np.zeros_like(t)
    for i, onset in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else t_end
        mask = (t >= onset) & (t < end)
        y[mask] = polarities[i]
    return t, y


def generate_closed_loop(seed=0, t_end=1.0, T=T_NOMINAL):
    """Regular continuous bipolar square wave, frequency-locked."""
    half_period = T / 2
    boundaries = np.arange(0, t_end, half_period)
    return _bipolar_wave_from_boundaries(boundaries, t_end)


def generate_open_loop(seed=0, t_end=1.0, T=1 / 5.5):
    """Fixed-frequency regular continuous bipolar square wave (no feedback)."""
    half_period = T / 2
    boundaries = np.arange(0, t_end, half_period)
    return _bipolar_wave_from_boundaries(boundaries, t_end)


def generate_poisson(seed=1, t_end=1.0, T=T_NOMINAL):
    """
    IPI_n = max(T, Exponential(1/T)) — segment widths (not gaps) drawn
    from this distribution, so an occasional very wide flat segment
    represents the long-gap tail, matching the reference icon.
    """
    rng = np.random.default_rng(seed)
    boundaries = [0.0]
    while boundaries[-1] < t_end:
        width = max(T, rng.exponential(T)) / 2  # /2 so it reads as a half-cycle width
        boundaries.append(boundaries[-1] + width)
    return _bipolar_wave_from_boundaries(boundaries, t_end)


def generate_uniform_jitter(seed=2, t_end=1.0, T=T_NOMINAL):
    """IPI_n = max(T, Uniform(0.5T, 1.5T)) — bounded segment-width jitter."""
    rng = np.random.default_rng(seed)
    boundaries = [0.0]
    while boundaries[-1] < t_end:
        width = max(T, rng.uniform(0.5 * T, 1.5 * T)) / 2
        boundaries.append(boundaries[-1] + width)
    return _bipolar_wave_from_boundaries(boundaries, t_end)


def generate_polarity_randomized(seed=3, t_end=1.0, T=T_NOMINAL):
    """
    IPI = T exactly (regular timing, same as Closed/Open Loop); polarity
    of each segment chosen randomly rather than strictly alternating.
    """
    half_period = T / 2
    boundaries = np.arange(0, t_end, half_period)
    rng = np.random.default_rng(seed)
    polarities = rng.choice([1, -1], size=len(boundaries))
    return _bipolar_wave_from_boundaries(boundaries, t_end, polarities=polarities)


def generate_replay(seed=4, t_end=1.0, T=T_NOMINAL):
    """
    Pre-recorded waveform stand-in: identical regular timing to Closed
    Loop (since Replay literally re-plays a previously recorded
    closed-loop signal) — matches the reference icon, which renders it
    as the same clean regular square wave, distinguished only by color.
    """
    half_period = T / 2
    boundaries = np.arange(0, t_end, half_period)
    return _bipolar_wave_from_boundaries(boundaries, t_end)


ICON_GENERATORS = {
    'CLOSED_LOOP':      generate_closed_loop,
    'OPEN_LOOP_54':     generate_open_loop,
    'OPEN_LOOP_55':     generate_open_loop,
    'POISSON':          generate_poisson,
    'UNIFORM_JITTER':   generate_uniform_jitter,
    'PHASE_RANDOMIZED': generate_polarity_randomized,
    'REPLAY':           generate_replay,
}


def draw_condition_icon(ax, condition, scale=1.0, linewidth=1.5):
    """
    Draws one condition's waveform icon into the given axis: a clean,
    minimal unipolar-pulse sketch with no ticks/axis labels, colored
    border matching the condition's color, a faint dashed reference sine
    wave (the underlying oscillation being targeted/suppressed) drawn
    behind it, and the condition name as a title.

    scale: multiplier for title fontsize and line widths, matching the
    convention used across the other reusable panel functions in this
    project. Defaults to 1.0.
    """
    color = CONDITION_COLORS[condition]
    generator = ICON_GENERATORS[condition]
    t, y = generator()

    ylim = (-1.3, 1.3)

    # Reference sine wave — smooth, same nominal frequency as the square
    # wave, sized to be clearly visible without clipping. Matches the
    # reference icon's dashed background sine.
    ref_sine = 0.85 * np.sin(2 * np.pi * t / T_NOMINAL)
    ax.plot(t, ref_sine, color='#aaaaaa', linewidth=1.2 * scale,
            linestyle='--', alpha=0.9, zorder=1)

    ax.plot(t, y, color=color, linewidth=linewidth * scale,
            drawstyle='steps-post', zorder=2)

    ax.set_ylim(*ylim)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    # No axis spines at all here — unlike the data-bearing panels
    # elsewhere (which use the standardized left+bottom adjust_spines
    # style), these are pure schematic sketches with no real axes to
    # show, so a full box/frame would just add visual clutter.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(CONDITION_LABELS[condition], fontsize=8,
                 color=color, fontweight='bold', pad=4 * scale)


def draw_condition_icons_row(axes_list, conditions=ICON_CONDITIONS, scale=1.0):
    """
    Draws all six condition icons into axes_list (a list of matplotlib
    Axes, one per condition, in `conditions` order) — the reusable core
    for the fig4 top-row cartoon representations.
    """
    for ax, condition in zip(axes_list, conditions):
        draw_condition_icon(ax, condition, scale=scale)


def main():
    fig, axes = plt.subplots(1, len(ICON_CONDITIONS), figsize=(3 * len(ICON_CONDITIONS), 2.2))
    draw_condition_icons_row(axes, ICON_CONDITIONS)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'condition_icons_row.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'condition_icons_row.pdf', bbox_inches='tight')
    print('[SAVED] condition_icons_row.png / .pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()