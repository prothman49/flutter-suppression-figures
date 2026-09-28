"""Download the dataset from Zenodo and unpack it into data/.

Fill in FILES after you create the Zenodo record. Each archive should
unpack into the folder layout described in data/README.md.
"""
import sys
import tarfile
import urllib.request

from flutter.config import DATA_DIR

FILES = {
    # "off_frequency_tests.tar.gz":  "https://zenodo.org/records/XXXXXXX/files/off_frequency_tests.tar.gz",
    # "full_360_phase_test.tar.gz":  "https://zenodo.org/records/XXXXXXX/files/full_360_phase_test.tar.gz",
    # "time_trace_no_wind_flutter_control.tar.gz": "https://zenodo.org/records/XXXXXXX/files/time_trace_no_wind_flutter_control.tar.gz",
}


def main(names):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = DATA_DIR / name
        print(f"downloading {name}")
        urllib.request.urlretrieve(FILES[name], dest)
        with tarfile.open(dest) as tar:
            tar.extractall(DATA_DIR)
        dest.unlink()
    print(f"data in {DATA_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(FILES))
