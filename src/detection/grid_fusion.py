"""Kalıcı ızgara füzyonu — slot çizgi geometrisini zaman içinde kararlı kılar.

Zamansal oylama slot DOLULUĞUNU yumuşatır; bu modül slot GEOMETRİSİNİ
(ızgara çizgi konumlarını) karelerde biriktirir. Her eksen (dikey x, yatay y)
için çizgiler 1B takip edilir: konumlar üstel hareketli ortalama (EMA) ile
yumuşatılır, görülme sayısı (hits) ve kaçırma (miss) tutulur.

Faydası: bir karede çizgi tespiti kaçsa bile (gölge, blur, oklüzyon) ızgara
haritası ayakta kalır → titremesiz, eksiksiz slot ızgarası. Video sabitleme ve
zamansal oylama ile birlikte canlı kararlılığı tamamlar.
"""

from __future__ import annotations


class _AxisTracker:
    def __init__(self, match_tol: float, alpha: float,
                 max_miss: int, min_hits: int):
        self.match_tol = match_tol
        self.alpha = alpha
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.tracks: list[dict] = []

    def reset(self):
        self.tracks = []

    def update(self, positions):
        used = set()
        for p in positions:
            best_i, best_d = -1, self.match_tol + 1.0
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue
                d = abs(p - t["pos"])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0 and best_d <= self.match_tol:
                t = self.tracks[best_i]
                t["pos"] = self.alpha * p + (1 - self.alpha) * t["pos"]
                t["hits"] += 1
                t["miss"] = 0
                used.add(best_i)
            else:
                self.tracks.append({"pos": float(p), "hits": 1, "miss": 0})
                used.add(len(self.tracks) - 1)
        # Eşleşmeyenleri yaşlandır, çok eskiyenleri at
        for i, t in enumerate(self.tracks):
            if i not in used:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        # Güvenilir (yeterince görülmüş) çizgileri sıralı döndür
        confident = [t["pos"] for t in self.tracks if t["hits"] >= self.min_hits]
        return sorted(confident)


class GridLineFusion:
    def __init__(self, match_tol: float = 14.0, alpha: float = 0.4,
                 max_miss: int = 12, min_hits: int = 1):
        self._x = _AxisTracker(match_tol, alpha, max_miss, min_hits)
        self._y = _AxisTracker(match_tol, alpha, max_miss, min_hits)

    def reset(self):
        self._x.reset()
        self._y.reset()

    def update(self, xs, ys):
        """Mevcut karenin çizgi konumlarını füzyonla; kararlı (xs, ys) döndür."""
        return self._x.update(list(xs)), self._y.update(list(ys))
