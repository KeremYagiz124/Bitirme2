"""Nicel değerlendirme modülü.

Park tespiti sonuçlarını ground truth ile karşılaştırarak Precision, Recall,
F1, Average Precision (AP/mAP) ve karışıklık matrisi üretir; sonuçları
grafik (PNG) ve tablo (CSV) olarak kaydeder.

Kullanım:
    from src.evaluation import evaluate_counts, evaluate_detections
    from src.evaluation.runner import run_evaluation
"""

from src.evaluation.metrics import (
    bbox_iou,
    match_detections,
    precision_recall_f1,
    average_precision,
    confusion_counts,
    evaluate_counts,
    evaluate_detections,
    DetectionMetrics,
    CountMetrics,
)

__all__ = [
    "bbox_iou",
    "match_detections",
    "precision_recall_f1",
    "average_precision",
    "confusion_counts",
    "evaluate_counts",
    "evaluate_detections",
    "DetectionMetrics",
    "CountMetrics",
]
