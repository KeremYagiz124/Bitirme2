"""
CLI pipeline: YOLOv8 detection + IoU zone analysis.

Usage:
    # Resim
    python scripts/run_parking.py --source data/raw/araba1.jpeg --zones data/raw/araba1.json

    # Video (ESC ile çık)
    python scripts/run_parking.py --source data/raw/video.mp4 --zones data/raw/araba1.json

    # Webcam
    python scripts/run_parking.py --source 0 --zones data/raw/araba1.json

    # Çıktıyı kaydet
    python scripts/run_parking.py --source data/raw/araba1.jpeg --zones data/raw/araba1.json --save
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer
from src.parking import STATUS_AVAILABLE, STATUS_OCCUPIED, STATUS_FORBIDDEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="Resim/video yolu veya webcam index (0)")
    p.add_argument("--zones", required=True,
                   help="Zone JSON dosyası")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Model ağırlığı (varsayılan: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold")
    p.add_argument("--iou-thresh", type=float, default=0.25,
                   help="IoU eşiği (zone analizi)")
    p.add_argument("--save", action="store_true",
                   help="Çıktıyı outputs/ klasörüne kaydet")
    return p.parse_args()


def print_summary(result, total_frames: int = 1):
    print(f"\n{'─'*40}")
    print(f"  Boş slot    : {result.available}")
    print(f"  Dolu slot   : {result.occupied}")
    print(f"  Yasak araç  : {result.forbidden_vehicles}")
    print(f"{'─'*40}\n")


def run_image(source: str, detector, analyzer, save: bool):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Hata: resim açılamadı — {source}")
        sys.exit(1)

    detections = detector.detect(frame)
    result = analyzer.analyze(detections)
    out = analyzer.draw(frame, result, detections)

    print_summary(result)

    if save:
        out_dir = Path("outputs/parking")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(source).name
        cv2.imwrite(str(out_path), out)
        print(f"Kaydedildi: {out_path}")

    cv2.imshow("Parking Analysis", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(source, detector, analyzer, save: bool):
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Hata: kaynak açılamadı — {source}")
        sys.exit(1)

    writer = None
    if save and isinstance(src, str):
        out_dir = Path("outputs/parking")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(src).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Çıktı: {out_path}")

    print("ESC veya Q ile çıkış.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        result = analyzer.analyze(detections)
        out = analyzer.draw(frame, result, detections)

        # HUD
        hud = (f"Bos:{result.available}  Dolu:{result.occupied}"
               f"  Yasak:{result.forbidden_vehicles}")
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)

        if writer:
            writer.write(out)

        cv2.imshow("Parking Analysis", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    print("Model yükleniyor...")
    detector = VehicleDetector(model_path=args.model, conf=args.conf)

    print("Zone yükleniyor...")
    loader = ZoneLoader(args.zones)
    analyzer = ParkingAnalyzer(loader, iou_threshold=args.iou_thresh)
    print(f"  {len(loader.parking_zones)} park zonu, "
          f"{len(loader.forbidden_zones)} yasak zon")

    source = args.source
    ext = Path(source).suffix.lower() if not source.isdigit() else ""

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        run_image(source, detector, analyzer, args.save)
    else:
        run_video(source, detector, analyzer, args.save)


if __name__ == "__main__":
    main()
