"""Run every figure notebook top to bottom. Output goes to output/.

    python make_all_figures.py          # all figures
    python make_all_figures.py 3 5      # just fig3 and fig5
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
which = sys.argv[1:] or [p.name[3:] for p in sorted((ROOT / "figures").glob("fig*"))]

with tempfile.TemporaryDirectory() as tmp:
    for n in which:
        nb = ROOT / "figures" / f"fig{n}" / f"fig{n}.ipynb"
        if not nb.exists():
            print(f"--- fig{n}: no notebook, skipping")
            continue
        print(f"--- fig{n}")
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=-1",
                        "--output-dir", tmp, str(nb)], check=True)
