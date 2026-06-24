"""IoU tabanlı basit araç tracker.

Frame'ler arası bbox eşleştirmesi yaparak her aracın merkez geçmişini tutar;
son N kare boyunca yer değiştirmesi düşük olanları "statik" (park etmiş)
olarak işaretler.

Park alanı tespiti için: yalnızca statik araçlar park sırası oluşturur,
böylece hareket halindeki araçlar yanlış slot tespitlerine neden olmaz.
"""

import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    union    = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("id", "bbox", "history", "misses", "frames_seen", "first_seen", "class_id", "confidence", "duration")

    def __init__(self, tid: int, bbox, history_len: int, class_id: int, confidence: float):
        self.id          = tid
        self.bbox        = bbox
        self.history     = deque([self._center(bbox)], maxlen=history_len)
        self.misses      = 0
        self.frames_seen = 1
        self.first_seen  = time.time()
        self.class_id    = class_id
        self.confidence  = confidence
        self.duration    = 0.0

    @staticmethod
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def update(self, bbox, confidence: float, dt: float = 0.0):
        self.bbox = bbox
        self.history.append(self._center(bbox))
        self.misses = 0
        self.frames_seen += 1
        self.confidence = confidence
        self.duration += dt

    def miss(self):
        self.misses += 1

    def is_static(self, min_history: int, max_disp_ratio: float, check_window: int = 15) -> bool:
        history_to_use = list(self.history)[-check_window:]
        if len(history_to_use) < min_history:
            return False  # Require min_history to confirm state (tests require False initially)
        xs = [p[0] for p in history_to_use]
        ys = [p[1] for p in history_to_use]
        
        # Trim outliers: remove the top and bottom 10% of coordinates to handle YOLO jitter robustly.
        n_trim = int(len(history_to_use) * 0.10)
        if n_trim > 0 and len(history_to_use) - 2 * n_trim >= min_history:
            xs_sorted = sorted(xs)
            ys_sorted = sorted(ys)
            xs_trimmed = xs_sorted[n_trim:-n_trim]
            ys_trimmed = ys_sorted[n_trim:-n_trim]
            disp = ((max(xs_trimmed) - min(xs_trimmed)) ** 2 + (max(ys_trimmed) - min(ys_trimmed)) ** 2) ** 0.5
        else:
            disp = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            
        ref = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])
        if ref <= 0:
            return False
        return disp <= max_disp_ratio * ref


class VehicleTracker:
    def __init__(
        self,
        iou_threshold: float   = 0.30,
        max_misses: int        = 10,
        history_len: int       = 15,
        min_history: int       = 5,
        max_disp_ratio: float  = 0.20,
        # Ego-motion compensation (sürücü kameralarında kritik):
        # Kamera hareketinden gelen global translation tahmin edilip
        # track displacement'tan çıkarılır → araç gerçek hareketi izole edilir.
        ego_motion: bool       = True,
        ego_max_features: int  = 80,
    ):
        self.iou_threshold    = iou_threshold
        self.max_misses       = max_misses
        self.history_len      = history_len
        self.min_history      = min_history
        self.max_disp_ratio   = max_disp_ratio
        self.ego_motion       = ego_motion
        self.ego_max_features = ego_max_features
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._prev_gray: Optional[np.ndarray] = None
        self._last_ego: tuple[float, float] = (0.0, 0.0)

    @property
    def last_ego_motion(self) -> tuple[float, float]:
        """Son `update()` çağrısında tahmin edilen kamera (dx, dy)."""
        return self._last_ego

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._prev_gray = None
        self._last_ego = (0.0, 0.0)

    def _estimate_ego_motion(self, frame, ignore_boxes) -> tuple[float, float]:
        """Frame-to-frame median translation (dx, dy) — kamera hareketi.

        Sparse LK optical flow: Shi-Tomasi köşeleri (araç bbox'ları dışında) →
        prev'ten curr'a takip → her vektörün medyanı. Araç hareketleri outlier
        olarak medyan tarafından bastırılır.
        """
        if frame is None or not self.ego_motion:
            return 0.0, 0.0

        h, w = frame.shape[:2]
        # Target width of 320px reduces pixel operations by ~9x compared to 960px
        target_w = 320
        scale = target_w / float(w)
        target_h = int(h * scale)

        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0, 0.0

        # Araç bbox'larını mask ile dışla
        h_s, w_s = gray.shape
        mask = np.full((h_s, w_s), 255, dtype=np.uint8)
        for b in ignore_boxes:
            x1 = max(0, int(b[0] * scale))
            y1 = max(0, int(b[1] * scale))
            x2 = min(w_s, int(b[2] * scale))
            y2 = min(h_s, int(b[3] * scale))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0

        pts0 = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self.ego_max_features,
            qualityLevel=0.01, minDistance=6, mask=mask, blockSize=5,
        )
        if pts0 is None or len(pts0) < 8:
            self._prev_gray = gray
            return 0.0, 0.0

        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts0, None,
            winSize=(11, 11), maxLevel=2,
        )
        self._prev_gray = gray
        if pts1 is None:
            return 0.0, 0.0
        good = (status.reshape(-1) == 1)
        if good.sum() < 6:
            return 0.0, 0.0

        dx = pts1[good, 0, 0] - pts0[good, 0, 0]
        dy = pts1[good, 0, 1] - pts0[good, 0, 1]
        
        # Scale the translation vector back to original frame coordinates
        ego = (float(np.median(dx) / scale), float(np.median(dy) / scale))
        self._last_ego = ego
        return ego

    def get_static_tracks(self, min_frames: int = 20) -> list[tuple]:
        """Yeterince uzun süredir görülen ve statik olan track bbox'ları."""
        out = []
        for tr in self._tracks.values():
            if tr.frames_seen >= min_frames and tr.is_static(
                self.min_history, self.max_disp_ratio
            ):
                out.append(tuple(tr.bbox))
        return out

    def get_static_tracks_with_duration(self, min_frames: int = 20) -> list[tuple]:
        """Statik track'leri (bbox, duration_sec) çiftleri olarak döner."""
        out = []
        for tr in self._tracks.values():
            if tr.frames_seen >= min_frames and tr.is_static(
                self.min_history, self.max_disp_ratio
            ):
                out.append((tuple(tr.bbox), tr.duration))
        return out

    def update(self, detections: list[dict],
               frame: Optional[np.ndarray] = None,
               dt: float = 0.033) -> List[bool]:
        """Tracker'ı detections ile güncelle, her detection için is_static döner.

        `frame` verilirse ego-motion (optik akış) düzeltmesi uygulanır:
        kamera hareketinden gelen global translation track history'sinden
        çıkarılır → sürücü kamerasında bile park etmiş araçlar statik gözükür.
        """
        # 1) Ego-motion tahmin: kameranın global hareketi
        det_boxes = [d["bbox"] for d in detections]
        ego_dx, ego_dy = self._estimate_ego_motion(frame, det_boxes)

        # 2) Mevcut track geçmişlerinde kamera hareketini telafi et
        # (geçmiş merkezleri ego-motion ile kaydır → yeni merkezle aynı uzayda)
        if ego_dx != 0.0 or ego_dy != 0.0:
            for tr in self._tracks.values():
                tr.history = deque(
                    ((px + ego_dx, py + ego_dy) for (px, py) in tr.history),
                    maxlen=tr.history.maxlen,
                )
                # Ayrıca tr.bbox'ı da kaydır (ego-motion compensation)
                b = tr.bbox
                tr.bbox = [b[0] + ego_dx, b[1] + ego_dy, b[2] + ego_dx, b[3] + ego_dy]

        n = len(detections)
        assigned: list[_Track | None] = [None] * n

        for tr in list(self._tracks.values()):
            best_i, best_iou = -1, 0.0
            for i, d in enumerate(detections):
                if assigned[i] is not None:
                    continue
                v = _iou(tr.bbox, d["bbox"])
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0 and best_iou >= self.iou_threshold:
                tr.update(detections[best_i]["bbox"], detections[best_i].get("confidence", 0.8), dt)
                assigned[best_i] = tr
            else:
                tr.miss()

        for i in range(n):
            if assigned[i] is None:
                d = detections[i]
                tr = _Track(
                    self._next_id,
                    d["bbox"],
                    self.history_len,
                    d.get("class_id", 2),
                    d.get("confidence", 0.8)
                )
                tr.duration = dt
                self._tracks[self._next_id] = tr
                assigned[i] = tr
                self._next_id += 1

        for tid in [t for t, x in self._tracks.items() if x.misses > self.max_misses]:
            del self._tracks[tid]

        return [
            tr.is_static(self.min_history, self.max_disp_ratio) if tr else False
            for tr in assigned
        ]

    def get_missed_detections(self) -> list[dict]:
        """YOLO'nun kaçırdığı fakat tracker'ın hala aktif tuttuğu araçları döner."""
        out = []
        active_boxes = [tr.bbox for tr in self._tracks.values() if tr.misses == 0]
        for tr in self._tracks.values():
            if tr.misses > 0:
                # Suppress missed tracks that overlap significantly with any active track to prevent duplicates
                if any(_iou(tr.bbox, ab) > 0.40 for ab in active_boxes):
                    continue
                out.append({
                    "bbox": list(tr.bbox),
                    "class_id": tr.class_id,
                    "confidence": tr.confidence,
                    "tracker_miss": True,
                    "is_static": tr.is_static(self.min_history, self.max_disp_ratio)
                })
        return out
