"""Zamansal araç olasılığı haritası (Occupancy Heatmap).

Her piksel için son N karede "araç bbox'ı içinde olma" sıklığını biriktirir.
Sürücü kamerasında ego-motion ile birlikte kaydırılır → kamera hareketi
telafi edilir.

Kullanım: bir slot adayının altındaki pikseller geçmişte hiç araç görmemişse
(yol kenarı kaldırımı, refüj, yolun ortası) o slot büyük olasılıkla yanlış
pozitiftir. Heuristik çıktısını bu sinyalle süzeriz.

Performans: downsample edilmiş float32 array üzerinde çalışır — 1280×720
→ 320×180 → ~60 KB; update ~0.2-0.5 ms.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class OccupancyHeatmap:
    def __init__(
        self,
        downsample: float = 0.25,
        decay: float      = 0.96,
        add_per_frame: float = 0.18,
        max_value: float  = 1.0,
        warmup_frames: int = 10,
    ):
        self.downsample    = float(downsample)
        self.decay         = float(decay)
        self.add_per_frame = float(add_per_frame)
        self.max_value     = float(max_value)
        self.warmup_frames = int(warmup_frames)

        self._heatmap: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None  # (h, w) küçük
        self._update_count = 0

    def reset(self):
        self._heatmap = None
        self._shape = None
        self._update_count = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._update_count >= self.warmup_frames

    def _ensure(self, frame_shape):
        h, w = frame_shape[:2]
        sh = max(8, int(h * self.downsample))
        sw = max(8, int(w * self.downsample))
        if self._heatmap is None or self._shape != (sh, sw):
            self._heatmap = np.zeros((sh, sw), dtype=np.float32)
            self._shape = (sh, sw)

    def update(self, frame_shape, vehicle_bboxes,
               ego_dx: float = 0.0, ego_dy: float = 0.0):
        """Heatmap'i ego-motion + decay + bbox eklemeleriyle ilerlet."""
        self._ensure(frame_shape)
        sh, sw = self._shape

        # 1) Ego-motion: heatmap'i kamera hareketinin TERSİ yönünde kaydır.
        # (kamera ileri ⇒ sahnedeki pikseller ekrana göre geride kalır)
        if (abs(ego_dx) > 0.1 or abs(ego_dy) > 0.1):
            sdx = ego_dx * self.downsample
            sdy = ego_dy * self.downsample
            M = np.float32([[1.0, 0.0, sdx], [0.0, 1.0, sdy]])
            self._heatmap = cv2.warpAffine(
                self._heatmap, M, (sw, sh),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        # 2) Decay (zamansal sönüm)
        self._heatmap *= self.decay

        # 3) Bbox'ları ekle (clip ile satüre)
        for b in vehicle_bboxes:
            x1 = max(0, int(b[0] * self.downsample))
            y1 = max(0, int(b[1] * self.downsample))
            x2 = min(sw, int(b[2] * self.downsample))
            y2 = min(sh, int(b[3] * self.downsample))
            if x2 > x1 and y2 > y1:
                np.add(self._heatmap[y1:y2, x1:x2],
                       self.add_per_frame, out=self._heatmap[y1:y2, x1:x2])
        if self._heatmap.max() > self.max_value:
            np.clip(self._heatmap, 0.0, self.max_value, out=self._heatmap)

        self._update_count += 1

    def slot_probability(self, slot, frame_shape) -> float:
        """Slot alanı içindeki ortalama olasılık (0-1)."""
        if self._heatmap is None:
            return 0.0
        sh, sw = self._shape
        x1, y1, x2, y2 = slot
        sx1 = max(0, int(x1 * self.downsample))
        sy1 = max(0, int(y1 * self.downsample))
        sx2 = min(sw, int(x2 * self.downsample))
        sy2 = min(sh, int(y2 * self.downsample))
        if sx2 - sx1 < 1 or sy2 - sy1 < 1:
            return 0.0
        return float(np.mean(self._heatmap[sy1:sy2, sx1:sx2]))

    def slot_max(self, slot, frame_shape) -> float:
        """Slot içindeki en yüksek olasılık (dilatasyon benzeri kontrol)."""
        if self._heatmap is None:
            return 0.0
        sh, sw = self._shape
        x1, y1, x2, y2 = slot
        sx1 = max(0, int(x1 * self.downsample))
        sy1 = max(0, int(y1 * self.downsample))
        sx2 = min(sw, int(x2 * self.downsample))
        sy2 = min(sh, int(y2 * self.downsample))
        if sx2 - sx1 < 1 or sy2 - sy1 < 1:
            return 0.0
        return float(np.max(self._heatmap[sy1:sy2, sx1:sx2]))

    def slot_neighborhood_max(self, slot, frame_shape,
                              expand: float = 0.4) -> float:
        """Slot bbox'ını expand oranında genişleterek yakın bölgenin en
        yüksek olasılığını döner. Park sırasındaki bir aracın hemen yanındaki
        gerçek slot için bu değer yüksek; izole hayalî slot için düşük olur.
        """
        x1, y1, x2, y2 = slot
        sw = x2 - x1; sh_ = y2 - y1
        ex1 = x1 - expand * sw
        ey1 = y1 - expand * sh_
        ex2 = x2 + expand * sw
        ey2 = y2 + expand * sh_
        return self.slot_max((ex1, ey1, ex2, ey2), frame_shape)
