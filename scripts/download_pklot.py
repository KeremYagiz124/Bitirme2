"""
PKLot dataset indirme scripti (Roboflow üzerinden).

Kullanım:
    python scripts/download_pklot.py --api-key YOUR_KEY
    python scripts/download_pklot.py --api-key YOUR_KEY --output data/pklot
"""

import argparse
import sys
from pathlib import Path


DATASETS = {
    "pklot-small": ("parking-space-detection", "pklot-8w36e", 1),
    "pklot-large": ("parking-lot-detection-system", "parking-lot-detection-yolov8", 1),
    "parking-detection": ("universe-datasets", "parking-space-detection-gzedm", 4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="Roboflow API anahtarı")
    ap.add_argument("--output",  default="data/pklot", help="İndirme dizini")
    ap.add_argument("--workspace", default="", help="Roboflow workspace slug")
    ap.add_argument("--project",   default="", help="Roboflow project slug")
    ap.add_argument("--version",   type=int, default=1)
    args = ap.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("HATA: roboflow paketi yüklü değil. pip install roboflow")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace = args.workspace or "parking-space-detection"
    project   = args.project   or "pklot-8w36e"
    version   = args.version

    print(f"Dataset indiriliyor: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=args.api_key)
    proj = rf.workspace(workspace).project(project)
    ver  = proj.version(version)
    ver.download("yolov8", location=str(out_dir))

    yaml_candidates = list(out_dir.rglob("data.yaml"))
    if not yaml_candidates:
        print("UYARI: data.yaml bulunamadı.")
        sys.exit(1)

    yaml_path = yaml_candidates[0]
    train_dir = yaml_path.parent / "train" / "images"
    count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    print(f"\nDataset hazır: {yaml_path.parent}")
    print(f"Eğitim görüntüsü: {count}")
    print(f"\nModeli eğitmek için:")
    print(f"  python scripts/train_parking_detector.py --data-dir {yaml_path.parent}")


if __name__ == "__main__":
    main()
