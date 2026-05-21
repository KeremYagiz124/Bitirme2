"""Sokak modu (StreetParkingDetector) değerlendirme scripti.

Ground truth JSON'daki beklenen boş alan sayısıyla karşılaştırır;
precision, recall ve F1 hesaplar.

Kullanım:
    python scripts/evaluate_street.py
    python scripts/evaluate_street.py --gt data/ground_truth/street_gt.json
    python scripts/evaluate_street.py --conf 0.4 --verbose
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.detection.street_parking_detector import StreetParkingDetector


def evaluate_image(image_path: Path, expected_empty: int,
                   detector: VehicleDetector, conf: float,
                   verbose: bool) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {"error": f"Görüntü açılamadı: {image_path}"}

    det = StreetParkingDetector()
    detections = detector.detect(frame)
    result = det.analyze(frame, detections)

    detected_empty = result["empty_count"]
    detected_parked = result["occupied_count"]

    # Sayı tabanlı TP/FP/FN hesabı:
    # TP = doğru tespit edilen boş alan sayısı = min(detected, expected)
    # FP = fazla tespit (detected > expected)
    # FN = kaçırılan (detected < expected)
    tp = min(detected_empty, expected_empty)
    fp = max(0, detected_empty - expected_empty)
    fn = max(0, expected_empty - detected_empty)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    if verbose:
        print(f"  Araç tespiti: {detected_parked} araç")
        print(f"  Beklenen boş: {expected_empty}  |  Tespit edilen: {detected_empty}")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    return {
        "image":    image_path.name,
        "expected": expected_empty,
        "detected": detected_empty,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",      default="data/ground_truth/street_gt.json")
    ap.add_argument("--conf",    type=float, default=0.35)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gt_path = ROOT / args.gt
    if not gt_path.exists():
        print(f"HATA: Ground truth bulunamadı: {gt_path}")
        sys.exit(1)

    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    detector = VehicleDetector(conf=args.conf)
    results  = []

    print("\n" + "=" * 60)
    print("SOKAK MODU DEĞERLENDİRME")
    print("=" * 60)

    for entry in gt_data["images"]:
        img_path = ROOT / entry["path"]
        expected = entry["expected_empty_count"]
        note     = entry.get("note", "")

        print(f"\n[{entry['path']}]  (Beklenen boş: {expected})")
        if note:
            print(f"  Not: {note}")

        r = evaluate_image(img_path, expected, detector, args.conf, args.verbose or True)
        if "error" in r:
            print(f"  HATA: {r['error']}")
            continue
        results.append(r)

    if not results:
        print("\nHiç sonuç yok.")
        return

    # Özet tablo
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    macro_p  = sum(r["precision"] for r in results) / len(results)
    macro_r  = sum(r["recall"]    for r in results) / len(results)
    macro_f1 = sum(r["f1"]        for r in results) / len(results)

    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    print("\n" + "=" * 60)
    print(f"{'Görüntü':<16} {'Beklenen':>8} {'Tespit':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['image']:<16} {r['expected']:>8} {r['detected']:>8} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
    print("-" * 60)
    print(f"{'Makro Ort.':<16} {'':>8} {'':>8} "
          f"{macro_p:>7.3f} {macro_r:>7.3f} {macro_f1:>7.3f}")
    print(f"{'Mikro Ort.':<16} {'':>8} {'':>8} "
          f"{micro_p:>7.3f} {micro_r:>7.3f} {micro_f1:>7.3f}")
    print("=" * 60)
    print(f"Toplam: TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Mikro F1: %{micro_f1 * 100:.1f}\n")


if __name__ == "__main__":
    main()
