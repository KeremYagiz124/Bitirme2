"""Otomatik boş park yeri tespiti — sokak/yan kamera perspektifi.

Yaklaşım:
  1. Tespit edilen araçların bottom-y medyanından bir "park bandı" belirlenir.
  2. Bu banda dahil araçlar x'e göre sıralanır (tek sıra mantığı).
  3. Bitişik araçlar arası boşluklar park yeri adayı olarak değerlendirilir.
  4. Genişlik [min_gap_ratio, max_gap_ratio] × medyan_araç_genişliği aralığında
     ve alt-kenarlar yeterince hizalı olmalı.
  5. Geniş boşluklar kaç aracın sığabileceğine göre eşit parçalara bölünür.
  6. Frame'ler arası tutarlılık: son N karenin en az M'inde görülmüş olmalı.
"""

from collections import deque

import cv2
import numpy as np

COLOR_EMPTY = (0, 220, 80)


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    union    = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class StreetParkingDetector:
    def __init__(
        self,
        # Yan kamera için ayarlanmış varsayılanlar:
        # Park boşluğu ≈ araç uzunluğunun %40-180'i (yan perspektifte)
        min_gap_ratio: float       = 0.40,
        max_gap_ratio: float       = 2.0,
        # Tek sıra tespiti: bottom-y band genişliği (medyan yüksekliğe oran)
        row_band_ratio: float      = 0.80,
        # Alt kenar hizalama toleransı (yan görüşte daha gevşek)
        bottom_align_tol: float    = 0.65,
        min_cars_per_row: int      = 2,
        ignore_top_ratio: float    = 0.20,
        min_car_width_ratio: float = 0.025,
        max_spaces_per_gap: int    = 3,
        # Temporal smoothing
        smoothing_frames: int      = 5,
        smoothing_min_hits: int    = 3,
        match_iou: float           = 0.35,
    ):
        self.min_gap_ratio       = min_gap_ratio
        self.max_gap_ratio       = max_gap_ratio
        self.row_band_ratio      = row_band_ratio
        self.bottom_align_tol    = bottom_align_tol
        self.min_cars_per_row    = min_cars_per_row
        self.ignore_top_ratio    = ignore_top_ratio
        self.min_car_width_ratio = min_car_width_ratio
        self.max_spaces_per_gap  = max_spaces_per_gap
        self.smoothing_frames    = smoothing_frames
        self.smoothing_min_hits  = smoothing_min_hits
        self.match_iou           = match_iou

        self._history: deque = deque(maxlen=smoothing_frames)

    @staticmethod
    def _bbox_size(b):
        return (b[2] - b[0]), (b[3] - b[1])

    def reset_history(self):
        self._history.clear()

    def _filter_candidates(self, frame_shape, detections):
        h, w = frame_shape[:2]
        min_w = self.min_car_width_ratio * w
        y_min = self.ignore_top_ratio * h
        out = []
        for d in detections:
            b = d["bbox"]
            bw, bh = self._bbox_size(b)
            cy = (b[1] + b[3]) / 2
            if bw < min_w:
                continue
            if cy < y_min:
                continue
            # Yan kamera: park eden araç yandan geniş görünür (en/boy > 1.0)
            if bh > 0 and bw / bh < 0.8:
                continue
            out.append(b)
        return out

    def _build_parking_row(self, boxes):
        """Yan kamera için tek sıra: bottom-y band etrafındaki araçlar.

        Outlier boyutlu kutuları (çok büyük/küçük) filtreler — ağaç/direk vb.
        """
        if not boxes:
            return []
        bottoms = [b[3] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        widths  = [b[2] - b[0] for b in boxes]
        med_bottom = float(np.median(bottoms))
        med_h      = float(np.median(heights))
        med_w      = float(np.median(widths))

        # Boyut tutarlılığı: medyandan çok sapan kutuları çıkar
        band = [b for b in boxes
                if abs(b[3] - med_bottom) <= self.row_band_ratio * med_h
                and 0.3 * med_w <= (b[2] - b[0]) <= 3.0 * med_w
                and 0.3 * med_h <= (b[3] - b[1]) <= 3.0 * med_h]

        if len(band) < self.min_cars_per_row:
            band = list(boxes)
        return sorted(band, key=lambda b: b[0])

    def _detect_raw(self, frame, detections):
        h, w = frame.shape[:2]
        candidates = self._filter_candidates(frame.shape, detections)
        row        = self._build_parking_row(candidates)

        empty_spaces: list[tuple[int, int, int, int]] = []

        if len(row) < self.min_cars_per_row:
            return empty_spaces, row, [row] if row else []

        widths  = [b[2] - b[0] for b in row]
        heights = [b[3] - b[1] for b in row]
        med_w   = float(np.median(widths))
        med_h   = float(np.median(heights))
        min_gap = self.min_gap_ratio * med_w
        max_gap = self.max_gap_ratio * med_w

        frame_edge_margin = 0.05 * w  # frame kenarına çok yakın boşlukları yok say

        for i in range(len(row) - 1):
            left  = row[i]
            right = row[i + 1]
            gap_x1 = left[2]
            gap_x2 = right[0]
            gap_w  = gap_x2 - gap_x1

            if gap_w < min_gap or gap_w > max_gap:
                continue

            # Frame kenarına çok yakın boşluklar = sahte tespit
            if gap_x1 < frame_edge_margin or gap_x2 > w - frame_edge_margin:
                continue

            bot_left  = left[3]
            bot_right = right[3]
            if abs(bot_left - bot_right) > self.bottom_align_tol * med_h:
                continue

            n = max(1, int(round(gap_w / med_w)))
            n = min(n, self.max_spaces_per_gap)

            base_bot = (bot_left + bot_right) / 2
            y2 = int(min(h, base_bot))
            y1 = int(max(0, base_bot - med_h))

            step = gap_w / n
            for j in range(n):
                sx1 = int(gap_x1 + j * step)
                sx2 = int(gap_x1 + (j + 1) * step)
                empty_spaces.append((sx1, y1, sx2, y2))

        return empty_spaces, row, [row]

    def _confirm(self, candidates):
        self._history.append(list(candidates))

        if len(self._history) < self.smoothing_min_hits:
            return list(candidates)

        confirmed = []
        for cur in candidates:
            hits = sum(
                1 for past_frame in self._history
                if any(_bbox_iou(cur, past) >= self.match_iou for past in past_frame)
            )
            if hits >= self.smoothing_min_hits:
                confirmed.append(cur)
        return confirmed

    def analyze(self, frame: np.ndarray, detections: list[dict]) -> dict:
        raw_empty, parked, rows = self._detect_raw(frame, detections)
        confirmed = self._confirm(raw_empty)

        return {
            "parked":         parked,
            "rows":           rows,
            "empty_spaces":   confirmed,
            "raw_candidates": raw_empty,
            "empty_count":    len(confirmed),
            "occupied_count": len(parked),
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        out = frame.copy()
        spaces = result["empty_spaces"]
        if not spaces:
            return out

        overlay = out.copy()
        for (x1, y1, x2, y2) in spaces:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_EMPTY, -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        for (x1, y1, x2, y2) in spaces:
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_EMPTY, 3)
            label = "BOS"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.rectangle(out,
                          (cx - tw // 2 - 5, cy - th // 2 - 5),
                          (cx + tw // 2 + 5, cy + th // 2 + 5),
                          COLOR_EMPTY, -1)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        return out
