"""Every data and output path lives here.

By default data is read from <repo>/data. To use data somewhere else
(e.g. your existing figures_for_dissertation folder), set FLUTTER_DATA:

    export FLUTTER_DATA=~/Desktop/Flutter_analysis/figures_for_dissertation

The folder layout under DATA_DIR matches figures_for_dissertation.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.path.expanduser(os.environ.get("FLUTTER_DATA", REPO_ROOT / "data")))

OFF_FREQ_DIR = DATA_DIR / "off_frequency_tests"
OL_RDM_DIR = OFF_FREQ_DIR / "OL_RDM"                       # six-condition randomized bags + JSON
JITTER_DIR = OFF_FREQ_DIR / "jitter_freq_sweep"            # uniform jitter at 3/5.5/8/10 Hz
PHASE_SWEEP_DIR = DATA_DIR / "full_360_phase_test" / "feb24"
TIME_TRACE_DIR = DATA_DIR / "time_trace_no_wind_flutter_control"

OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
