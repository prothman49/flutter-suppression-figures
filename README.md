# Flutter Suppression Figures

Code to reproduce the figures in:

> Rothman, P., et al. *TODO title.* Journal of Fluids and Structures (in preparation).

Real-time aeroelastic flutter suppression using optical flow sensing and
phase-locked piezoelectric actuation.

## Setup

    git clone https://github.com/prothman49/flutter-suppression-figures.git
    cd flutter-suppression-figures
    conda env create -f environment.yml
    conda activate flutter-figures
    nbstripout --install    # strips notebook outputs on commit

## Data

Data is hosted on Zenodo (DOI TODO). See [`data/README.md`](data/README.md).

    python scripts/download_data.py

## Make figures

Each figure is a notebook in `figures/figN/` next to its figurefirst SVG
template. Open it in Jupyter and run all cells, or run everything at once:

    python make_all_figures.py          # all figures
    python make_all_figures.py 3        # just fig3

Filled SVGs and PDFs go to `output/`.

| Figure | Notebook | Data | Content |
|---|---|---|---|
| 1 | `figures/fig1/fig1.ipynb` | OL_RDM, time trace | Setup, Logic block diagram, real-time control trace, FFT, total power (closed loop) |
| 2 | `figures/fig2/fig2.ipynb` | phase sweep | Reduction vs. phase offset, 0–330° |
| 3 | `figures/fig3/fig3.ipynb` | OL_RDM, 3/8/10 Hz | Off-frequency open loop vs. frequency-tracked closed loop |
| 4 | `figures/fig4/fig4.ipynb` | OL_RDM | Six-condition comparison: icons, FFTs, trial order, power |
| 5 | `figures/fig5/fig5.ipynb` | jitter sweep, OL_RDM, 3/8/10 Hz | Uniform jitter frequency 

## Analysis code

Shared code is in the `flutter` package (installed by the environment).
Each module also runs standalone to make its own diagnostic plots:

    python -m flutter.fft_energy

| Module | Was | What it does |
|---|---|---|
| `flutter/config.py` | (new) | All data and output paths |
| `flutter/plotting.py` | (new) | Shared rcParams (`set_style()`) |
| `flutter/fft_energy.py` | `plot_fft_energy.py` | Bag reading, FFT/Welch PSD, energy, core panels |
| `flutter/frequency_profile.py` | `Fig3_frequency_profile.py` | Off-frequency multi-trial bags, FFT row, triptych |
| `flutter/jitter_sweep.py` | `fig5_jitter_freq_sweep.py` | Jitter frequency sweep loading and panels |
| `flutter/phase_sweep.py` | `phase_sweep_comprehensive.py` | Phase sweep JSON loading, violin panel |
| `flutter/condition_comparison.py` | `plot_CL_RDM.py` | Per-sample JSON reductions, CI box plots, stats |
| `flutter/trial_order.py` | `plot_trial_order.py` | Trial order timeline from bag timestamps |
| `flutter/condition_icons.py` | `condition_icons.py` | Condition waveform icons |
| `flutter/time_trace.py` | `plot_time_trace_updated.py` | Real-time control trace (fig1 panel B) |

## Experimental conditions

| Code key | Name | Description |
|---|---|---|
| `CLOSED_LOOP` | Closed Loop | Actuation locked to live flutter frequency estimate, 255° phase offset |
| `OPEN_LOOP_55` | Open Loop 5.5 Hz | Fixed 5.5 Hz actuation, no feedback |
| `POISSON` | Poisson | Pulse intervals drawn from a Poisson process |
| `UNIFORM_JITTER` | Uniform Jitter | Pulse intervals jittered uniformly around the nominal period |
| `PHASE_RANDOMIZED` | Polarity Randomized | Correct frequency, actuator polarity randomly flipped, no phase relationship to wing state |
| `REPLAY` | Replay | Previously recorded closed-loop command played back open loop |

## Repo layout

    flutter/             analysis and plotting package
    figures/figN/        figure notebook + figurefirst SVG template
    notebooks/           exploratory notebooks (see notebooks/README.md)
    scripts/             data download
    data/                (not in git) downloaded data
    output/              (not in git) generated figures

## Citation

TODO

## License

TODO: confirm with advisor (MIT is common for code; CC-BY for data).
