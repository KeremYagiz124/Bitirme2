"""Park çizgisi (boyalı şerit) tespiti ve ızgara çıkarımı.

Çizgili otoparklarda zemindeki boyalı şeritlerden park ızgarasını çıkarır.
En iyi kuş bakışı (IPM sonrası) görüntüde çalışır: perspektif kalktığında
şeritler düz ve eksen-hizalı olur, Hough ile kolay bulunur.

Adaptif kullanım: has_lines() ile çizgi olup olmadığına bakılır.
  - Çizgi varsa → build_slots() ızgara hücrelerini döndürür (çizgi-tabanlı).
  - Çizgi yoksa (yol kenarı vb.) → çağıran mevcut geometri yöntemine düşer.

Saf OpenCV (Canny + HoughLinesP); ek bağımlılık/patent gerektirmez.
"""

from __future__ import annotations

import cv2
import numpy as np


class ParkingLineDetector:
    def __init__(self, canny_lo: int = 60, canny_hi: int = 180,
                 hough_thresh: int = 50, min_line_frac: float = 0.12,
                 max_gap: int = 25, white_thresh: int = 170,
                 cluster_tol_frac: float = 0.025, angle_tol_deg: float = 22.0,
                 use_color: bool = True, white_v_min: int = 165,
                 white_s_max: int = 70, yellow_h: tuple = (18, 38)):
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.hough_thresh = hough_thresh
        self.min_line_frac = min_line_frac
        self.max_gap = max_gap
        self.white_thresh = white_thresh
        self.cluster_tol_frac = cluster_tol_frac
        self.angle_tol_deg = angle_tol_deg
        # Renk segmentasyonu: boyalı şeritler beyaz veya sarıdır.
        self.use_color = use_color
        self.white_v_min = white_v_min   # beyaz: yüksek parlaklık
        self.white_s_max = white_s_max   # beyaz: düşük doygunluk
        self.yellow_h = yellow_h         # sarı ton aralığı (HSV H)

    # ── Boyalı şerit renk maskesi ────────────────────────────────────────────
    def _color_line_mask(self, img: np.ndarray) -> np.ndarray:
        """Beyaz ve sarı boyalı şeritleri HSV'de vurgula.

        Asfalt dokusu kenar üretip Canny'yi yanıltabilir; renk maskesi gerçek
        boya işaretlerini hedefler → çizgi tespiti çok daha sağlam olur.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.white_v_min),
                            (180, self.white_s_max, 255))
        yellow = cv2.inRange(hsv, (self.yellow_h[0], 60, 80),
                             (self.yellow_h[1], 255, 255))
        return cv2.bitwise_or(white, yellow)

    # ── Kenar/çizgi maskesi ──────────────────────────────────────────────────
    def _edge_mask(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Boyalı şeritler asfalttan açıktır: gri eşik + Canny kenarları
        _, white = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        mask = cv2.bitwise_or(white, edges)
        # Renkli giriş varsa beyaz/sarı boya maskesini de ekle
        if self.use_color and img.ndim == 3:
            mask = cv2.bitwise_or(mask, self._color_line_mask(img))
        return mask

    # ── Ham segmentler ───────────────────────────────────────────────────────
    def detect_segments(self, img: np.ndarray, mask: np.ndarray | None = None):
        """HoughLinesP ile çizgi segmentleri. Döner: [(x1,y1,x2,y2), ...]."""
        h, w = img.shape[:2]
        if mask is None:
            mask = self._edge_mask(img)
        min_len = int(self.min_line_frac * max(h, w))
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, self.hough_thresh,
                                minLineLength=min_len, maxLineGap=self.max_gap)
        if lines is None:
            return []
        return [tuple(int(v) for v in l[0]) for l in lines]

    @staticmethod
    def _angle_deg(seg) -> float:
        x1, y1, x2, y2 = seg
        return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    def _split_orientation(self, segments):
        """Segmentleri dikey ve yatay olarak ayır (BEV'de slot bölücüler dikey)."""
        verticals, horizontals = [], []
        for s in segments:
            a = self._angle_deg(s)
            if abs(a - 90) <= self.angle_tol_deg:
                verticals.append(s)
            elif a <= self.angle_tol_deg or abs(a - 180) <= self.angle_tol_deg:
                horizontals.append(s)
        return verticals, horizontals

    @staticmethod
    def _cluster_positions(values, tol: float):
        """1B konumları tol içinde kümele, küme merkezlerini döndür (sıralı)."""
        if not values:
            return []
        vals = sorted(values)
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if v - clusters[-1][-1] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [float(np.mean(c)) for c in clusters]

    @staticmethod
    def _refine_position(pos: float, profile: np.ndarray, window: float) -> float:
        """Konumu, ±window penceresindeki yoğunluk profilinin ağırlık merkezine
        snap et (alt-piksel hassasiyet). Boş pencerede konum değişmez."""
        lo = max(0, int(round(pos - window)))
        hi = min(len(profile), int(round(pos + window)) + 1)
        if hi - lo < 1:
            return pos
        seg = profile[lo:hi]
        s = float(seg.sum())
        if s <= 0:
            return pos
        idx = np.arange(lo, hi, dtype=np.float64)
        return float((idx * seg).sum() / s)

    # ── Izgara çizgileri ─────────────────────────────────────────────────────
    def grid_lines(self, img: np.ndarray, refine: bool = True):
        h, w = img.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / float(w)
            nh = int(h * scale)
            nw = max_w
            img_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            xs_small, ys_small = self._grid_lines_raw(img_small, refine)
            return [x / scale for x in xs_small], [y / scale for y in ys_small]
        return self._grid_lines_raw(img, refine)

    def _grid_lines_raw(self, img: np.ndarray, refine: bool = True):
        h, w = img.shape[:2]
        tol = self.cluster_tol_frac * max(h, w)
        mask = self._edge_mask(img)
        verticals, horizontals = self._split_orientation(
            self.detect_segments(img, mask=mask))
        xs = self._cluster_positions(
            [(s[0] + s[2]) / 2 for s in verticals], tol)
        ys = self._cluster_positions(
            [(s[1] + s[3]) / 2 for s in horizontals], tol)
        if refine and (xs or ys):
            binm = (mask > 0)
            col_profile = binm.sum(axis=0).astype(np.float64)  # x ekseni
            row_profile = binm.sum(axis=1).astype(np.float64)  # y ekseni
            xs = [self._refine_position(x, col_profile, tol) for x in xs]
            ys = [self._refine_position(y, row_profile, tol) for y in ys]
        return xs, ys

    def has_lines(self, img: np.ndarray, min_vertical: int = 3,
                  min_horizontal: int = 0) -> bool:
        """Yeterli sayıda paralel şerit var mı (çizgi-tabanlı yönteme geç)?"""
        xs, ys = self.grid_lines(img)
        return len(xs) >= min_vertical and len(ys) >= min_horizontal

    # ── Slot hücreleri ───────────────────────────────────────────────────────
    def build_slots_from_positions(self, xs, ys, shape):
        """Verilen çizgi konumlarından slot hücrelerini kur.

        Füzyon sonrası kararlı çizgilerle de çağrılabilir.
        Döner: [(x1, y1, x2, y2), ...].
        """
        h, w = shape[:2]
        xs = sorted(xs)
        if len(xs) < 2:
            return []
        if len(ys) >= 2:
            y_top, y_bot = min(ys), max(ys)
        else:
            # yatay referans yoksa orta bandı kullan (üst/alt %15 kırp)
            y_top, y_bot = 0.15 * h, 0.85 * h
        slots = []
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            if x2 - x1 < 0.02 * w:   # çok dar aralıkları ele
                continue
            slots.append((int(x1), int(y_top), int(x2), int(y_bot)))
        return self.filter_by_size_consistency(slots)

    def build_slots(self, img: np.ndarray):
        """Çizgi ızgarasından slot hücrelerini (bbox) çıkar.

        Dikey bölücüler arasındaki her aralık bir slot; dikey kapsam yatay
        çizgiler (varsa) ile, yoksa görüntü yüksekliğinin orta bandı ile
        sınırlanır. Döner: [(x1, y1, x2, y2), ...].
        """
        xs, ys = self.grid_lines(img)
        return self.build_slots_from_positions(xs, ys, img.shape)

    @staticmethod
    def filter_by_size_consistency(slots, width_tol: float = 0.55):
        """Gerçek slotlar eşit boyutludur; medyan genişlikten çok sapan
        adayları (yanlış birleşmiş/bölünmüş hücreler) ele.

        3'ten az slot varsa dokunulmaz (medyan güvenilmez).
        """
        if len(slots) < 3:
            return slots
        widths = [s[2] - s[0] for s in slots]
        med = float(np.median(widths))
        if med <= 0:
            return slots
        return [s for s in slots if abs((s[2] - s[0]) - med) <= width_tol * med]

    @staticmethod
    def classify_slots(slots, vehicle_bboxes, overlap_thresh: float = 0.15):
        """Her slotu boş/dolu işaretle: içinde yeterli araç örtüşmesi varsa dolu.

        Döner: [{"bbox": slot, "occupied": bool, "score": örtüşme_oranı}, ...].
        Eğitim gerektirmez; YOLO araç tespitleriyle çalışır.
        """
        out = []
        for slot in slots:
            sx1, sy1, sx2, sy2 = slot
            slot_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
            best_score = 0.0
            occupied = False
            for vb in vehicle_bboxes:
                vx1, vy1, vx2, vy2 = vb
                ix1, iy1 = max(sx1, vx1), max(sy1, vy1)
                ix2, iy2 = min(sx2, vx2), min(sy2, vy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                if inter > 0:
                    overlap_slot = inter / slot_area
                    veh_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
                    overlap_veh = inter / veh_area
                    best_score = max(best_score, overlap_slot)
                    if overlap_slot >= overlap_thresh or overlap_veh >= 0.45:
                        occupied = True
            out.append({"bbox": slot, "occupied": occupied,
                        "score": round(float(best_score), 3)})
        return out
