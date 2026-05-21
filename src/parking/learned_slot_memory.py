"""Long-term öğrenilmiş park slot kütüğü.

Bir araç yeterli süre statik kaldıysa bbox'ı kalıcı slot olarak kaydedilir.
Sonraki frame'lerde slot durumu (BOS/DOLU) mevcut araç bbox'larıyla IoU
karşılaştırması yapılarak güncellenir.

Bu sayede gerçek park yerlerinin konumu zamanla otomatik öğrenilir —
heuristik tahmin (perspektif fit + boşluk bölme) bir zemin verir; öğrenme
zemini gözlemle "kalibre" eder.
"""

from dataclasses import dataclass

import numpy as np


def _slot_in_mask(mask, slot, min_ratio: float = 0.25) -> bool:
    """Slot bbox'ın road/drivable maskesinde %min_ratio kaplama oranı."""
    if mask is None:
        return True
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = slot
    sx1 = max(0, int(x1)); sy1 = max(0, int(y1))
    sx2 = min(w, int(x2)); sy2 = min(h, int(y2))
    if sx2 - sx1 < 2 or sy2 - sy1 < 2:
        return False
    sub = mask[sy1:sy2, sx1:sx2]
    if sub.size == 0:
        return False
    return (float(np.count_nonzero(sub)) / float(sub.size)) >= min_ratio


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class LearnedSlot:
    bbox: tuple
    learned_frame: int
    last_dolu_frame: int = 0
    seen_count: int = 1   # kaç kez bir araç bu slotta görüldü


class LearnedSlotMemory:
    def __init__(
        self,
        slot_iou_match: float = 0.40,
        dedup_iou: float      = 0.55,
        max_slots: int        = 200,
    ):
        self.slot_iou_match = slot_iou_match
        self.dedup_iou      = dedup_iou
        self.max_slots      = max_slots
        self._slots: list[LearnedSlot] = []
        self._frame = 0

    def reset(self):
        self._slots.clear()
        self._frame = 0

    def update(self, static_track_bboxes: list[tuple],
               current_vehicle_bboxes: list,
               road_mask=None) -> list[dict]:
        """Statik track'leri yeni slot olarak öğren ve mevcut durumları döndür.

        road_mask verilirse:
          - Yeni slot kaydı: konum drivable area içinde olmalı, aksi halde
            kaldırım/refüj üzerinde sahte slot oluşmaz.
          - Periyodik temizlik: zaten kaydedilmiş ama drivable dışında kalan
            slot'lar (kamera açısı değişimi, sahne değişimi) silinir.
        """
        self._frame += 1

        # 1) Yeni statik track'leri slot olarak kaydet (dedup + drivable kontrolü)
        for tb in static_track_bboxes:
            tb_t = tuple(float(v) for v in tb)
            if any(_iou(tb_t, s.bbox) > self.dedup_iou for s in self._slots):
                continue
            if not _slot_in_mask(road_mask, tb_t, min_ratio=0.25):
                continue  # drivable area dışında → sahte slot adayı, reddet
            self._slots.append(LearnedSlot(
                bbox=tb_t,
                learned_frame=self._frame,
                last_dolu_frame=self._frame,
            ))

        # Periyodik temizlik: drivable dışına çıkmış mevcut slot'lar
        if road_mask is not None and self._frame % 30 == 0:
            self._slots = [
                s for s in self._slots
                if _slot_in_mask(road_mask, s.bbox, min_ratio=0.20)
            ]

        # FIFO sınırı
        if len(self._slots) > self.max_slots:
            self._slots = self._slots[-self.max_slots:]

        # 2) Slot durumları
        out = []
        for s in self._slots:
            occ = any(
                _iou(s.bbox, vb) >= self.slot_iou_match
                for vb in current_vehicle_bboxes
            )
            if occ:
                s.last_dolu_frame = self._frame
                s.seen_count += 1
            out.append({
                "bbox":     s.bbox,
                "occupied": occ,
                "age":      self._frame - s.learned_frame,
                "learned":  True,
            })
        return out

    @property
    def slots(self):
        return list(self._slots)
