"""Zamansal slot oylama — video titremesini azaltır.

Çizgi-tabanlı slot tespiti kare-kare hafif kayabilir veya doluluk anlık
gürültüden dolayı yanıp sönebilir. Bu sınıf, slotları karelerde IoU ile
eşleştirip son N karenin oy çoğunluğuyla boş/dolu durumunu yumuşatır.

Statik kamerada slotlar sabit kalır; kamera hafif oynasa bile IoU eşleştirmesi
küçük kaymaları tolere eder. Kararlı (yeterli geçmişe sahip) slotlar
işaretlenir, böylece çağıran isterse yalnızca kararlı slotları gösterebilir.
"""

from __future__ import annotations

from collections import deque


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


class TemporalSlotVoter:
    def __init__(self, history: int = 7, iou_thresh: float = 0.3,
                 max_miss: int = 8, stable_min: int = 3):
        self.history = history
        self.iou_thresh = iou_thresh
        self.max_miss = max_miss
        self.stable_min = min(stable_min, history)
        self.tracks: list[dict] = []

    def reset(self):
        self.tracks = []

    def update(self, classified: list[dict]) -> list[dict]:
        """Mevcut karenin slotlarını oylayıp yumuşatılmış listeyi döndür.

        classified: [{"bbox": (x1,y1,x2,y2), "occupied": bool, ...}]
        Döner: aynı sırada [{"bbox", "occupied"(yumuşatılmış), "stable", "score"}]
        """
        result = []
        used = set()
        for s in classified:
            bbox = s["bbox"]
            occ = bool(s["occupied"])
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue
                iou = _iou(bbox, t["bbox"])
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= self.iou_thresh:
                t = self.tracks[best_i]
                used.add(best_i)
                t["bbox"] = bbox
                t["votes"].append(occ)
                t["miss"] = 0
            else:
                t = {"bbox": bbox,
                     "votes": deque([occ], maxlen=self.history), "miss": 0}
                self.tracks.append(t)
                used.add(len(self.tracks) - 1)
            votes = t["votes"]
            smoothed = sum(votes) > len(votes) / 2.0
            result.append({
                "bbox": bbox,
                "occupied": smoothed,
                "stable": len(votes) >= self.stable_min,
                "score": s.get("score"),
            })

        # Eşleşmeyen track'leri yaşlandır, çok eskiyenleri at
        for i, t in enumerate(self.tracks):
            if i not in used:
                t["miss"] = t.get("miss", 0) + 1
        self.tracks = [t for t in self.tracks if t.get("miss", 0) <= self.max_miss]
        return result
