"""Uçtan uca değerlendirme çalıştırıcısı.

Bir görüntü kümesi + ground truth üzerinde park tespitini çalıştırır,
metrikleri hesaplar ve grafik (PNG) + tablo (CSV) olarak kaydeder.

Ground truth biçimi (JSON):
    {
        "1.png": {"empty": 0},
        "2.png": {"empty": 2},
        "3.png": {"empty": 3}
    }

Komut satırı:
    python -m src.evaluation.runner --gt data/ground_truth.json --images . \
        --out outputs/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from src.evaluation import metrics as M
from src.evaluation import plots as P


def load_ground_truth(gt_path: str) -> dict:
    """Ground truth JSON dosyasını yükle."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_detectors(conf: float):
    """VehicleDetector + StreetParkingDetector örnekleri oluştur (lazy import)."""
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.street_parking_detector import StreetParkingDetector
    vehicle = VehicleDetector(conf=conf)
    street = StreetParkingDetector(max_edge_extension_ratio=0.20)
    return vehicle, street


def run_evaluation(gt, images_dir: str, out_dir: str = "outputs/evaluation",
                   conf: float = 0.35, vehicle_detector=None,
                   street_detector=None) -> M.CountMetrics:
    """Değerlendirmeyi çalıştır ve sonuçları kaydet.

    gt: dict (filename -> {"empty": n}) veya JSON dosya yolu.
    vehicle_detector / street_detector: enjekte edilebilir (test için).
    Döner: CountMetrics
    """
    if isinstance(gt, str):
        gt = load_ground_truth(gt)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vehicle_detector is None or street_detector is None:
        vehicle_detector, street_detector = _build_detectors(conf)

    names, predicted, expected = [], [], []
    for fname, spec in gt.items():
        img_path = images_dir / fname
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        street_detector.reset_history()
        dets = vehicle_detector.detect(frame)
        result = street_detector.analyze(frame, dets)
        names.append(fname)
        predicted.append(int(result["empty_count"]))
        expected.append(int(spec["empty"]))

    cm = M.evaluate_counts(predicted, expected)

    # ─ CSV: özet ─
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        row = cm.as_row()
        w.writerow(list(row.keys()))
        w.writerow(list(row.values()))

    # ─ CSV: görüntü bazında ─
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "predicted", "expected", "TP", "FP", "FN"])
        for name, rec in zip(names, cm.per_image):
            w.writerow([name, rec["predicted"], rec["expected"],
                        rec["TP"], rec["FP"], rec["FN"]])

    # ─ Grafikler ─
    P.plot_confusion_matrix(cm.tp, cm.fp, cm.fn, 0,
                            out_dir / "confusion_matrix.png")
    P.plot_metric_bars(
        {"Precision": cm.micro_precision, "Recall": cm.micro_recall,
         "F1": cm.micro_f1},
        out_dir / "metrics_bars.png", title="Sayim-Temelli Metrikler")
    if names:
        P.plot_per_image_counts(predicted, expected,
                                out_dir / "per_image_counts.png")

    return cm


def run_adaptive_evaluation(gt, images_dir: str,
                            out_dir: str = "outputs/evaluation_adaptive",
                            conf: float = 0.35, ipm=None):
    """Adaptif dedektörü etiketli görüntülerde değerlendir (sayım-temelli).

    gt: dict (filename -> {"empty": n}) veya JSON yolu. ipm verilirse kuş
    bakışında çalışır (IPM katkısını ölçmek için aynı gt'yle ipm=None ve
    ipm=kalibre çağırıp karşılaştırılabilir).
    Döner: CountMetrics.
    """
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
    if isinstance(gt, str):
        gt = load_ground_truth(gt)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vehicle = VehicleDetector(conf=conf)
    adaptive = AdaptiveSlotDetector(use_voting=False)  # tek kare: oylama yok

    names, predicted, expected = [], [], []
    for fname, spec in gt.items():
        frame = cv2.imread(str(images_dir / fname))
        if frame is None:
            continue
        adaptive.reset()
        dets = vehicle.detect(frame)
        res = adaptive.analyze(frame, dets, ipm=ipm)
        names.append(fname)
        predicted.append(int(res["empty_count"]))
        expected.append(int(spec["empty"]))

    cm = M.evaluate_counts(predicted, expected)
    with open(out_dir / "adaptive_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        row = cm.as_row()
        w.writerow(list(row.keys()))
        w.writerow(list(row.values()))
    if names:
        P.plot_per_image_counts(predicted, expected,
                                out_dir / "adaptive_per_image.png")
    return cm


def run_synthetic_evaluation(n_scenes=120, seed=42,
                             out_dir="outputs/evaluation_synthetic",
                             iou_threshold=0.4, detector=None):
    """Sentetik sahnelerde slot-tespit algoritmasını değerlendir.

    YOLO'dan bağımsız: doğru araç tespitleri doğrudan beslenir, böylece
    geometrik boş-yer bulma algoritmasının kalitesi 100+ sahnede ölçülür.
    Hem tespit-temelli (bbox IoU) hem sayım-temelli metrikler üretir.

    Döner: (DetectionMetrics, CountMetrics)
    """
    from src.evaluation import synthetic
    from src.detection.street_parking_detector import StreetParkingDetector

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = synthetic.make_dataset(n_scenes=n_scenes, seed=seed)

    preds_per_image, gts_per_image = [], []
    pred_counts, exp_counts = [], []
    for frame, dets, gt_empty in scenes:
        det = detector or StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False, max_edge_extension_ratio=0.0,
        )
        det.reset_history()
        result = det.analyze(frame, dets)
        empty = [tuple(map(float, s)) for s in result["empty_spaces"]]
        preds_per_image.append(empty)
        gts_per_image.append([tuple(map(float, g)) for g in gt_empty])
        pred_counts.append(len(empty))
        exp_counts.append(len(gt_empty))

    dm = M.evaluate_detections(preds_per_image, gts_per_image,
                               iou_threshold=iou_threshold)
    cm = M.evaluate_counts(pred_counts, exp_counts)

    # CSV
    with open(out_dir / "detection_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        row = dm.as_row()
        w.writerow(list(row.keys()))
        w.writerow(list(row.values()))
    with open(out_dir / "count_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        row = cm.as_row()
        w.writerow(list(row.keys()))
        w.writerow(list(row.values()))

    # Grafikler
    P.plot_confusion_matrix(dm.tp, dm.fp, dm.fn, 0,
                            out_dir / "confusion_matrix.png",
                            title=f"Sentetik ({n_scenes} sahne) — Slot Tespiti")
    P.plot_metric_bars(
        {"Precision": dm.precision, "Recall": dm.recall,
         "F1": dm.f1, "AP": dm.ap},
        out_dir / "detection_metrics.png",
        title="Tespit-Temelli Metrikler")
    return dm, cm


def main():
    ap = argparse.ArgumentParser(description="Park tespiti değerlendirmesi")
    ap.add_argument("--mode", choices=["real", "synthetic"], default="real",
                    help="real: görüntü+GT; synthetic: prosedürel 100+ sahne")
    ap.add_argument("--gt", help="Ground truth JSON yolu (real mod)")
    ap.add_argument("--images", default=".", help="Görüntü klasörü (real mod)")
    ap.add_argument("--out", default="outputs/evaluation", help="Çıktı klasörü")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO conf eşiği")
    ap.add_argument("--scenes", type=int, default=120, help="Sentetik sahne sayısı")
    args = ap.parse_args()

    if args.mode == "synthetic":
        out = args.out if args.out != "outputs/evaluation" else "outputs/evaluation_synthetic"
        dm, cm = run_synthetic_evaluation(n_scenes=args.scenes, out_dir=out)
        print(f"Sentetik değerlendirme tamamlandı ({args.scenes} sahne).")
        print(f"  Precision: {dm.precision:.4f}")
        print(f"  Recall:    {dm.recall:.4f}")
        print(f"  F1:        {dm.f1:.4f}")
        print(f"  AP:        {dm.ap:.4f}")
        print(f"  Çıktılar:  {out}")
        return

    if not args.gt:
        ap.error("real mod için --gt zorunlu")
    cm = run_evaluation(args.gt, args.images, args.out, args.conf)
    print("Değerlendirme tamamlandı.")
    print(f"  Micro-Precision: {cm.micro_precision:.4f}")
    print(f"  Micro-Recall:    {cm.micro_recall:.4f}")
    print(f"  Micro-F1:        {cm.micro_f1:.4f}")
    print(f"  MAE:             {cm.mae:.4f}")
    print(f"  Çıktılar:        {args.out}")


if __name__ == "__main__":
    main()
