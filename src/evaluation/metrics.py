"""Değerlendirme metrikleri — saf fonksiyonlar (grafik/IO bağımlılığı yok).

İki değerlendirme türü desteklenir:

1. Tespit-temelli (detection): tahmin ve ground truth bbox'ları IoU ile
   eşleştirilir → TP/FP/FN → Precision, Recall, F1, Average Precision.

2. Sayım-temelli (count): görüntü başına tahmin edilen boş slot sayısı,
   beklenen sayı ile karşılaştırılır → mikro Precision/Recall/F1.

Tüm fonksiyonlar dış kütüphane (matplotlib, sklearn) gerektirmez; yalnızca
numpy kullanır. Bu sayede hızlı ve bağımsız test edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BBox = "tuple[float, float, float, float]"


# ─── Temel geometri ──────────────────────────────────────────────────────────

def bbox_iou(a, b) -> float:
    """İki bbox (x1, y1, x2, y2) arasındaki Intersection-over-Union."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─── Tespit eşleştirme ───────────────────────────────────────────────────────

def match_detections(preds, gts, iou_threshold: float = 0.5,
                     scores=None):
    """Tahminleri ground truth ile greedy IoU eşleştirmesiyle eşle.

    Tahminler (varsa) skora göre azalan sırada işlenir; her tahmin en yüksek
    IoU'lu ve henüz eşleşmemiş GT ile eşleştirilir. Eşik altı eşleşmeler
    sayılmaz.

    Döner: (tp, fp, fn, matches)
      tp, fp, fn: tam sayılar
      matches: [(pred_idx, gt_idx, iou), ...] eşleşen çiftler
    """
    n_pred = len(preds)
    n_gt = len(gts)
    if n_pred == 0:
        return 0, 0, n_gt, []
    if n_gt == 0:
        return 0, n_pred, 0, []

    order = list(range(n_pred))
    if scores is not None:
        order = sorted(order, key=lambda i: -scores[i])

    gt_used = [False] * n_gt
    matches = []
    tp = 0
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi in range(n_gt):
            if gt_used[gi]:
                continue
            iou = bbox_iou(preds[pi], gts[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_used[best_gi] = True
            matches.append((pi, best_gi, best_iou))
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, matches


# ─── Skaler metrikler ────────────────────────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int):
    """TP/FP/FN'den Precision, Recall, F1 üçlüsü."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def confusion_counts(tp: int, fp: int, fn: int, tn: int = 0):
    """Karışıklık matrisi sözlüğü (2x2 ikili sınıflandırma)."""
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


# ─── Average Precision (PR eğrisi altındaki alan) ────────────────────────────

def average_precision(preds_per_image, gts_per_image,
                      scores_per_image=None, iou_threshold: float = 0.5):
    """COCO-tarzı 101-nokta interpolasyonlu Average Precision.

    preds_per_image / gts_per_image: görüntü başına bbox listeleri
    scores_per_image: görüntü başına güven skorları (None → hepsi 1.0)

    Döner: AP (0..1)
    """
    all_scores = []
    all_tp = []
    total_gt = 0

    for idx, (preds, gts) in enumerate(zip(preds_per_image, gts_per_image)):
        total_gt += len(gts)
        if len(preds) == 0:
            continue
        scores = (scores_per_image[idx] if scores_per_image is not None
                  else [1.0] * len(preds))
        order = sorted(range(len(preds)), key=lambda i: -scores[i])
        gt_used = [False] * len(gts)
        for pi in order:
            best_iou, best_gi = 0.0, -1
            for gi in range(len(gts)):
                if gt_used[gi]:
                    continue
                iou = bbox_iou(preds[pi], gts[gi])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            is_tp = best_gi >= 0 and best_iou >= iou_threshold
            if is_tp:
                gt_used[best_gi] = True
            all_scores.append(scores[pi])
            all_tp.append(1 if is_tp else 0)

    if total_gt == 0:
        return 0.0
    if not all_scores:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp_arr = np.array(all_tp)[order]
    fp_arr = 1 - tp_arr
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # 101-nokta interpolasyon (COCO)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 101.0
    return float(ap)


# ─── Sonuç kapsayıcıları ─────────────────────────────────────────────────────

@dataclass
class DetectionMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap: float = 0.0
    per_image: list = field(default_factory=list)

    def as_row(self):
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "AP": round(self.ap, 4),
        }


@dataclass
class CountMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    mae: float = 0.0
    per_image: list = field(default_factory=list)

    def as_row(self):
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "Micro-Precision": round(self.micro_precision, 4),
            "Micro-Recall": round(self.micro_recall, 4),
            "Micro-F1": round(self.micro_f1, 4),
            "MAE": round(self.mae, 4),
        }


# ─── Yüksek seviye değerlendirme ─────────────────────────────────────────────

def evaluate_detections(preds_per_image, gts_per_image,
                        scores_per_image=None,
                        iou_threshold: float = 0.5) -> DetectionMetrics:
    """Tespit-temelli tam değerlendirme (bbox eşleştirme + AP)."""
    total_tp = total_fp = total_fn = 0
    per_image = []
    for idx, (preds, gts) in enumerate(zip(preds_per_image, gts_per_image)):
        scores = (scores_per_image[idx] if scores_per_image is not None
                  else None)
        tp, fp, fn, _ = match_detections(preds, gts, iou_threshold, scores)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p, r, f = precision_recall_f1(tp, fp, fn)
        per_image.append({"image": idx, "TP": tp, "FP": fp, "FN": fn,
                          "Precision": p, "Recall": r, "F1": f})
    p, r, f = precision_recall_f1(total_tp, total_fp, total_fn)
    ap = average_precision(preds_per_image, gts_per_image,
                           scores_per_image, iou_threshold)
    return DetectionMetrics(total_tp, total_fp, total_fn, p, r, f, ap, per_image)


def evaluate_counts(predicted_counts, expected_counts) -> CountMetrics:
    """Sayım-temelli değerlendirme (görüntü başına boş slot sayısı).

    Her görüntü için tahmin edilen sayı ile beklenen sayı karşılaştırılır:
      TP = min(tahmin, beklenen)
      FP = max(0, tahmin - beklenen)
      FN = max(0, beklenen - tahmin)
    Ayrıca ortalama mutlak hata (MAE) hesaplanır.
    """
    if len(predicted_counts) != len(expected_counts):
        raise ValueError("Tahmin ve beklenen sayı listeleri eşit uzunlukta olmalı")
    total_tp = total_fp = total_fn = 0
    abs_errors = []
    per_image = []
    for idx, (pred, exp) in enumerate(zip(predicted_counts, expected_counts)):
        tp = min(pred, exp)
        fp = max(0, pred - exp)
        fn = max(0, exp - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        abs_errors.append(abs(pred - exp))
        per_image.append({"image": idx, "predicted": pred, "expected": exp,
                          "TP": tp, "FP": fp, "FN": fn})
    p, r, f = precision_recall_f1(total_tp, total_fp, total_fn)
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    return CountMetrics(total_tp, total_fp, total_fn, p, r, f, mae, per_image)
