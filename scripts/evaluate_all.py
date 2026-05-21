"""Kapsamlı sistem değerlendirme scripti.

Her iki modu da test eder ve birleşik rapor üretir:
  1. Sabit Kamera Modu  — zona dayalı doluluk doğruluğu
  2. Sokak Modu         — boş alan tespiti precision/recall/F1

Kullanım:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --conf 0.4 --iou-thresh 0.25
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.detection.vehicle_detector import VehicleDetector
from src.parking.zone_loader import ZoneLoader
from src.parking.parking_analyzer import ParkingAnalyzer, STATUS_OCCUPIED, STATUS_AVAILABLE
from src.detection.street_parking_detector import StreetParkingDetector


# ──────────────────────────────────────────────
# Sabit kamera modu
# ──────────────────────────────────────────────

FIXED_TESTS = [
    ("data/raw/araba1.jpeg", "data/raw/araba1.json"),
    ("data/raw/araba2.jpg",  "data/raw/araba2.json"),
    ("data/raw/sample_bus.jpg", "data/raw/sample_bus.json"),
]


def _load_expected(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        z["id"]: z["expected"].lower()
        for z in data.get("zones", [])
        if z.get("type") == "parking" and "expected" in z
    }


def evaluate_fixed(conf: float, iou_thresh: float) -> dict:
    detector = VehicleDetector(conf=conf)
    total = correct = 0
    rows = []

    for img_rel, json_rel in FIXED_TESTS:
        img_path  = ROOT / img_rel
        json_path = ROOT / json_rel

        if not img_path.exists() or not json_path.exists():
            continue

        expected_map = _load_expected(json_path)
        if not expected_map:
            continue

        frame    = cv2.imread(str(img_path))
        if frame is None:
            continue

        loader   = ZoneLoader(str(json_path))
        analyzer = ParkingAnalyzer(loader, iou_threshold=iou_thresh)
        dets     = detector.detect(frame)
        result   = analyzer.analyze(dets)

        for zs in result.zone_statuses:
            if zs.zone.id not in expected_map:
                continue
            exp  = expected_map[zs.zone.id]
            pred = "occupied" if zs.status == STATUS_OCCUPIED else "available"
            ok   = exp == pred
            total   += 1
            correct += int(ok)
            rows.append({
                "file":  img_path.name,
                "zone":  zs.zone.id,
                "exp":   exp,
                "pred":  pred,
                "ok":    ok,
            })

    accuracy = correct / total if total > 0 else 0.0
    return {"rows": rows, "total": total, "correct": correct, "accuracy": accuracy}


# ──────────────────────────────────────────────
# Sokak modu
# ──────────────────────────────────────────────

def evaluate_street(conf: float, gt_path: Path) -> dict:
    if not gt_path.exists():
        return {"error": f"GT bulunamadı: {gt_path}"}

    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    detector = VehicleDetector(conf=conf)
    rows = []

    for entry in gt_data["images"]:
        img_path = ROOT / entry["path"]
        expected = entry["expected_empty_count"]
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        det    = StreetParkingDetector()
        dets   = detector.detect(frame)
        result = det.analyze(frame, dets)
        detected = result["empty_count"]

        tp = min(detected, expected)
        fp = max(0, detected - expected)
        fn = max(0, expected - detected)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "image":     img_path.name,
            "expected":  expected,
            "detected":  detected,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })

    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0

    return {
        "rows":     rows,
        "micro_p":  micro_p,  "micro_r":  micro_r,  "micro_f1":  micro_f1,
        "macro_f1": macro_f1,
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
    }


# ──────────────────────────────────────────────
# Rapor yazdırma
# ──────────────────────────────────────────────

def print_report(fixed: dict, street: dict):
    sep = "=" * 62

    print(f"\n{sep}")
    print("  AKILLI PARK SİSTEMİ — TAM DEĞERLENDİRME RAPORU")
    print(sep)

    # Sabit kamera
    print("\n── 1. SABİT KAMERA MODU (Zona Tabanlı Doluluk Tespiti) ──\n")
    if fixed.get("total", 0) == 0:
        print("  Test verisi bulunamadı.")
    else:
        print(f"  {'Dosya':<20} {'Zon':>4} {'Beklenen':<12} {'Tespit':<12} {'Sonuç'}")
        print("  " + "-" * 54)
        for r in fixed["rows"]:
            mark = "✓" if r["ok"] else "✗"
            print(f"  {r['file']:<20} {r['zone']:>4} {r['exp']:<12} {r['pred']:<12} {mark}")
        print("  " + "-" * 54)
        acc = fixed["accuracy"]
        print(f"  Doğruluk: {fixed['correct']}/{fixed['total']}  (%{acc * 100:.1f})\n")

    # Sokak modu
    print("── 2. SOKAK MODU (Boş Alan Tespiti Precision/Recall/F1) ──\n")
    if "error" in street:
        print(f"  HATA: {street['error']}")
    elif not street.get("rows"):
        print("  Test görüntüsü bulunamadı.")
    else:
        print(f"  {'Görüntü':<16} {'Beklenen':>8} {'Tespit':>8} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "-" * 58)
        for r in street["rows"]:
            print(f"  {r['image']:<16} {r['expected']:>8} {r['detected']:>8} "
                  f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
        print("  " + "-" * 58)
        print(f"  {'Mikro Ort.':<16} {'':>8} {'':>8} "
              f"{street['micro_p']:>7.3f} {street['micro_r']:>7.3f} "
              f"{street['micro_f1']:>7.3f}")
        print(f"  TP={street['total_tp']}  FP={street['total_fp']}  "
              f"FN={street['total_fn']}")
        print(f"  Makro F1: %{street['macro_f1'] * 100:.1f}  |  "
              f"Mikro F1: %{street['micro_f1'] * 100:.1f}\n")

    # Genel özet
    print(sep)
    print("  GENEL ÖZET")
    print(sep)
    fixed_acc  = fixed.get("accuracy", 0.0)
    street_f1  = street.get("micro_f1", 0.0)
    combined   = (fixed_acc + street_f1) / 2
    print(f"  Sabit Kamera Doğruluğu : %{fixed_acc * 100:.1f}")
    print(f"  Sokak Modu Mikro-F1    : %{street_f1 * 100:.1f}")
    print(f"  Genel Skor (ort.)      : %{combined * 100:.1f}")
    print(sep + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf",       type=float, default=0.35)
    ap.add_argument("--iou-thresh", type=float, default=0.25, dest="iou_thresh")
    ap.add_argument("--gt",         default="data/ground_truth/street_gt.json")
    args = ap.parse_args()

    print("Değerlendirme başlıyor...")

    print("  [1/2] Sabit kamera modu test ediliyor...")
    fixed  = evaluate_fixed(args.conf, args.iou_thresh)

    print("  [2/2] Sokak modu test ediliyor...")
    street = evaluate_street(args.conf, ROOT / args.gt)

    print_report(fixed, street)


if __name__ == "__main__":
    main()
