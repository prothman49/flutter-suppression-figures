# Data

Data is not stored in this repo. It is on Dryad:
https://doi.org/10.5061/dryad.1vhhmgrb6

The dataset is forthcoming. The DOI will resolve once the associated paper is published.

Download the three `.tar.gz` files from that page and unpack them here:

    for f in ~/Downloads/*.tar.gz; do tar -xzf "$f" -C data; done

The unpacked layout:

    data/
    ├── off_frequency_tests/
    │   ├── OL_RDM/                  six-condition randomized bags + per-sample JSON
    │   │                            ({CONDITION}_sample{NN}_{YYYYMMDD}_{HHMMSS}.bag/.json)
    │   ├── 3hz/  8hz/  10hz/        fixed open-loop off-frequency runs (bags/ and json/)
    │   └── jitter_freq_sweep/       UNIFORM_JITTER_{freq}hz_sample*.bag/.json
    ├── full_360_phase_test/
    │   └── feb24/                   flutter_on_off_phase{deg}_*.json
    └── time_trace_no_wind_flutter_control/

| Folder | Used by |
|---|---|
| `off_frequency_tests/OL_RDM` | fig1, fig3, fig4, fig5 |
| `off_frequency_tests/3hz`, `8hz`, `10hz` | fig3, fig5 |
| `off_frequency_tests/jitter_freq_sweep` | fig5 |
| `full_360_phase_test/feb24` | fig2 |
| `time_trace_no_wind_flutter_control` | fig1 |

To keep the data somewhere else, unpack it there and set `FLUTTER_DATA`:

    export FLUTTER_DATA=/path/to/data

Full descriptions of the files are in the README on the Dryad dataset page.
