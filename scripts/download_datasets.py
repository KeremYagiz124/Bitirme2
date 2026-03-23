"""
Dataset download helper.

PKLot: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
UA-DETRAC: http://detrac-db.rit.albany.edu/

This script downloads sample data for quick testing.
Run full downloads manually from the links above.
"""

import urllib.request
import zipfile
from pathlib import Path


PKLOT_SAMPLE_URL = "https://github.com/visualDetection/PKLot/archive/refs/heads/master.zip"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done.")


def extract_zip(src: Path, dest: Path) -> None:
    print(f"Extracting {src} -> {dest}")
    with zipfile.ZipFile(src) as z:
        z.extractall(dest)
    print("Done.")


if __name__ == "__main__":
    print("Dataset Download Helper")
    print("=" * 40)
    print("PKLot (full):  https://web.inf.ufpr.br/vri/databases/parking-lot-database/")
    print("UA-DETRAC:     http://detrac-db.rit.albany.edu/")
    print("COCO (cars):   https://cocodataset.org/#download")
    print()
    print("Place downloaded files under data/raw/<dataset_name>/")
    print("Expected structure:")
    print("  data/raw/pklot/")
    print("  data/raw/ua_detrac/")
    print("  data/raw/coco/")
