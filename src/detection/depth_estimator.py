"""Monoküler derinlik tahmini — MiDaS / Depth Anything.

Tek kameradan (stereo veya LiDAR olmadan) her piksel için göreli derinlik
haritası üretir. Park uygulamasında iki işe yarar:

  1. Aynı yatay hizada görünen ama farklı uzaklıktaki nesneleri ayırmak —
     IPM'i tamamlayarak çapraz açı belirsizliğini azaltır.
  2. Bir referans mesafe verildiğinde göreli derinliği metrik ölçeğe çevirmek.

Model ağırdır ve indirme gerektirir; bu yüzden DrivableAreaSegmenter ile aynı
desen kullanılır: model yüklenemezse `available=False` döner ve infer() None
verir (graceful degradation) — sistem asla çökmez.

Model kaynağı önceliği:
  1. local_model (TorchScript .pt) verilmişse onu yükler.
  2. Aksi halde torch.hub'dan MiDaS (intel-isl/MiDaS) indirmeyi dener.
İkisi de başarısızsa devre dışı kalır.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class DepthEstimator:
    def __init__(self, local_model: str | None = None,
                 model_type: str = "MiDaS_small",
                 device: str | None = None,
                 allow_download: bool = False):
        self.available = False
        self.model = None
        self.device = "cpu"
        self._torch = None
        self._transform = None
        self.model_type = model_type
        self._running_min = None
        self._running_max = None
        try:
            import torch
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if local_model and os.path.exists(local_model):
                self.model = torch.jit.load(local_model, map_location=self.device)
                if self.device == "cuda":
                    self.model.half()
                self.model.eval()
                self.available = True
            elif allow_download:
                self.model = torch.hub.load("intel-isl/MiDaS", model_type)
                self.model.to(self.device)
                if self.device == "cuda":
                    self.model.half()
                self.model.eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self._transform = (transforms.small_transform
                                   if "small" in model_type
                                   else transforms.dpt_transform)
                self.available = True
        except Exception:
            self.available = False
            self.model = None

    # ── Çıkarım ──────────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> np.ndarray | None:
        """Göreli derinlik haritası (frame boyutunda, float32). Yoksa None.

        Değerler göreli: yüksek = yakın (MiDaS ters derinlik üretir).
        0..1 aralığına normalize edilir.
        """
        if not self.available or frame is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            torch = self._torch
            with torch.inference_mode():
                if self._transform is not None:
                    inp = self._transform(rgb).to(self.device)
                else:
                    # TorchScript yerel model: basit normalize + CHW
                    img = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
                    inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if self.device == "cuda":
                    inp = inp.half()
                pred = self.model(inp)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                depth = pred.squeeze().float().cpu().numpy()
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            return self._normalize(depth)
        except Exception:
            return None

    def reset(self):
        """Reset the running temporal min/max values."""
        self._running_min = None
        self._running_max = None

    def _normalize(self, depth: np.ndarray = None) -> np.ndarray:
        """Derinliği 0..1 aralığına getir (zamansal EMA min-max normalizasyonu).
        
        Destek: Hem nesne metodu hem de statik metot (testlerin geriye dönük uyumluluğu için) olarak çağrılabilir.
        """
        if depth is None:
            # Statik metot olarak çağrıldığında: DepthEstimator._normalize(d)
            # Burada numpy dizisi 'self' parametresine bağlanır.
            d = self.astype(np.float32)
            lo, hi = float(d.min()), float(d.max())
            diff = hi - lo
            if diff < 1e-6:
                return np.zeros_like(d)
            return np.clip((d - lo) / diff, 0.0, 1.0)
            
        d = depth.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if self._running_min is None:
            self._running_min = lo
            self._running_max = hi
        else:
            self._running_min = 0.95 * self._running_min + 0.05 * lo
            self._running_max = 0.95 * self._running_max + 0.05 * hi
        diff = self._running_max - self._running_min
        if diff < 1e-6:
            return np.zeros_like(d)
        return np.clip((d - self._running_min) / diff, 0.0, 1.0)

    # ── Yardımcılar ──────────────────────────────────────────────────────────

    @staticmethod
    def depth_to_colormap(depth_norm: np.ndarray):
        """0..1 derinlik haritasını görselleştirme için BGR ısı haritasına çevir."""
        if depth_norm is None:
            return None
        d8 = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)

    @staticmethod
    def region_depth(depth_map: np.ndarray, bbox) -> float | None:
        """Bbox bölgesinin medyan göreli derinliği (0..1). Yoksa None."""
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return float(np.median(depth_map[y1:y2, x1:x2]))

    @classmethod
    def same_plane(cls, depth_map, bbox_a, bbox_b, tol: float = 0.12) -> bool | None:
        """İki bbox aynı derinlik düzleminde mi (göreli derinlik farkı < tol).

        Çapraz açıda 'aynı sıradaymış gibi görünen' ama farklı uzaklıktaki
        nesneleri ayırmak için kullanılır. depth_map yoksa None döner.
        """
        da = cls.region_depth(depth_map, bbox_a)
        db = cls.region_depth(depth_map, bbox_b)
        if da is None or db is None:
            return None
        return abs(da - db) <= tol
