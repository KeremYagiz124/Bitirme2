"""
Ground truth karşılaştırma scripti.

Zone JSON'una her park slotu için "expected" alanı ekle:
    "expected": "occupied"  veya  "expected": "available"

Kullanım:
    python scripts/compare_ground_truth.py --image data/raw/araba1.jpeg --zones data/raw/araba1.json
    python scripts/compare_ground_truth.py --image data/raw/araba1.jpeg --zones data/raw/araba1.json --conf 0.4
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.parking.zone_loader import ZoneLoader
from src.parking.parking_analyzer import ParkingAnalyzer, STATUS_OCCUPIED, STATUS_AVAILABLE


def load_expected(json_path: str) -> dict[int, str]:
    """zone_id → expected status ("occupied"/"available")"""
    with open(json_path) as f:
        data = json.load(f)
    expected = {}
    for z in data.get("zones", []):
        if z.get("type") == "parking" and "expected" in z:
            expected[z["id"]] = z["expected"].lower()
    return expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True, help="Kaynak görüntü")
    ap.add_argument("--zones",  required=True, help="Zone JSON (expected alanı içermeli)")
    ap.add_argument("--conf",   type=float, default=0.5)
    ap.add_argument("--iou-thresh", type=float, default=0.25, dest="iou_thresh")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"HATA: Görüntü açılamadı: {args.image}")
        sys.exit(1)

    expected_map = load_expected(args.zones)
    if not expected_map:
        print("UYARI: JSON'da 'expected' alanı bulunan park zonu yok.")
        print("  Örnek ekleme: her 'parking' zonuna  \"expected\": \"occupied\"  ekleyin.")
        sys.exit(0)

    loader   = ZoneLoader(args.zones)
    detector = VehicleDetector(conf=args.conf)
    analyzer = ParkingAnalyzer(loader, iou_threshold=args.iou_thresh)

    detections = detector.detect(frame)
    result     = analyzer.analyze(detections)

    total = correct = 0
    rows  = []

    for zs in result.zone_statuses:
        if zs.zone.id not in expected_map:
            continue
        exp = expected_map[zs.zone.id]
        pred = "occupied" if zs.status == STATUS_OCCUPIED else "available"
        ok   = exp == pred
        total   += 1
        correct += int(ok)
        rows.append((zs.zone.id, exp, pred, "✓" if ok else "✗"))

    print(f"\n{'ID':>4}  {'Beklenen':<12}  {'Sistem':<12}  {'Sonuç'}")
    print("-" * 40)
    for zone_id, exp, pred, mark in rows:
        print(f"{zone_id:>4}  {exp:<12}  {pred:<12}  {mark}")

    if total == 0:
        print("\nEşleşen zone bulunamadı.")
        return

    accuracy = correct / total * 100
    print("-" * 40)
    print(f"Accuracy: {correct}/{total}  (%{accuracy:.1f})\n")


if __name__ == "__main__":
    main()
