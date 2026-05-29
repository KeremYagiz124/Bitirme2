"""Otomatik boş park yeri tespiti — sokak/yan kamera perspektifi.

Geliştirilmiş yaklaşım (yüksek kesinlik):
  1. Araçlar bottom-y koordinatına göre 1B kümelenir → birden çok park sırası.
  2. Her sıra kendi içinde değerlendirilir; aralarda perspektif (x → genişlik)
     lineer fit ile slot boyu lokal olarak tahmin edilir.
  3. Bitişik araç-arası + sıra-uçlarındaki yol kenarı boşlukları slot adayı.
  4. Engel maskeleme: bir slot içinde araç-dışı obje (kişi/bisiklet/hidran/
     direk/bank/park sayacı) varsa slot elenir.
  5. Frame'ler arası tutarlılık: son N karenin en az M'inde görülmüş olmalı.
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


def _point_in_box(px: float, py: float, box) -> bool:
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


class StreetParkingDetector:
    def __init__(
        self,
        min_gap_ratio: float       = 0.20,
        max_gap_ratio: float       = 5.0,
        row_band_ratio: float      = 0.80,
        bottom_align_tol: float    = 0.35,
        min_cars_per_row: int      = 1,
        ignore_top_ratio: float    = 0.20,
        min_car_width_ratio: float = 0.025,
        max_spaces_per_gap: int    = 3,
        # Sıra uçları (yol kenarı uzun boş alanlar)
        detect_row_edges: bool     = True,
        max_edge_extension_ratio: float = 0.40,
        frame_edge_margin_ratio: float  = 0.0,
        # Multi-row & perspektif
        multi_row: bool            = True,
        max_rows: int              = 4,
        # Lateral split: aynı bottom-y row'unda yol genişliği kadar yatay
        # boşluk varsa AYRI sublane'lere böl (sol/sağ park şeritleri yolun
        # ortasındaki boşluk üstünden birleşmesin).
        lateral_split_ratio: float = 3.5,
        # Yol merkezi reddi: 0.0 = devre dışı
        road_center_reject_ratio: float = 0.0,
        # Engel maskeleme (slot içinde araç-dışı obje varsa eler)
        obstacle_overlap_min: float = 0.05,
        # Yol yüzeyi (asfalt) renk doğrulaması: park etmiş araçların altındaki
        # şerit referans alınır; slot içinde benzer renk yoksa slot elenir.
        road_color_check: bool     = True,
        road_color_tol_h: float    = 25.0,   # 30.0 → 25.0 (daha sıkı)
        road_color_tol_s: float    = 60.0,   # 70.0 → 60.0
        road_color_tol_v: float    = 60.0,   # 70.0 → 60.0
        # Performans: yol maskesi downsample oranı (0.5 = %50 küçült → ~4x hız).
        road_mask_scale: float     = 0.5,
        # Yol maskesi her N analiz'de bir yenilenir (cache).
        road_mask_period: int      = 5,
        # Temporal smoothing
        smoothing_frames: int      = 5,
        smoothing_min_hits: int    = 3,
        match_iou: float           = 0.35,
    ):
        self.min_gap_ratio            = min_gap_ratio
        self.max_gap_ratio            = max_gap_ratio
        self.row_band_ratio           = row_band_ratio
        self.bottom_align_tol         = bottom_align_tol
        self.min_cars_per_row         = min_cars_per_row
        self.ignore_top_ratio         = ignore_top_ratio
        self.min_car_width_ratio      = min_car_width_ratio
        self.max_spaces_per_gap       = max_spaces_per_gap
        self.detect_row_edges         = detect_row_edges
        self.max_edge_extension_ratio = max_edge_extension_ratio
        self.frame_edge_margin_ratio  = frame_edge_margin_ratio
        self.multi_row                = multi_row
        self.max_rows                 = max_rows
        self.lateral_split_ratio      = float(lateral_split_ratio)
        self.road_center_reject_ratio = float(road_center_reject_ratio)
        self.obstacle_overlap_min     = obstacle_overlap_min
        self.road_color_check         = road_color_check
        self.road_color_tol_h         = road_color_tol_h
        self.road_color_tol_s         = road_color_tol_s
        self.road_color_tol_v         = road_color_tol_v
        self.road_mask_scale          = float(road_mask_scale)
        self.road_mask_period         = max(1, int(road_mask_period))
        self._road_mask_cache         = None
        self._analyze_count           = 0
        self.smoothing_frames         = smoothing_frames
        self.smoothing_min_hits       = smoothing_min_hits
        self.match_iou                = match_iou

        self._history: deque = deque(maxlen=smoothing_frames)

    @staticmethod
    def _bbox_size(b):
        return (b[2] - b[0]), (b[3] - b[1])

    def reset_history(self):
        self._history.clear()
        self._road_mask_cache = None
        self._analyze_count   = 0

    # ── Aday filtre ───────────────────────────────────────────────
    def _filter_candidates(self, frame_shape, detections, static_mask=None):
        h, w = frame_shape[:2]
        min_w = self.min_car_width_ratio * w
        y_min = self.ignore_top_ratio * h
        out = []
        for i, d in enumerate(detections):
            # Tracker varsa: hareketli araçlar park sırasına dahil edilmez
            if static_mask is not None and i < len(static_mask) and not static_mask[i]:
                continue
            b = d["bbox"]
            bw, bh = self._bbox_size(b)
            cy = (b[1] + b[3]) / 2
            if bw < min_w:
                continue
            if cy < y_min:
                continue
            if bh > 0 and bw / bh < 0.8:
                continue
            out.append(b)
        return out

    # ── Sıra kümeleme (1B bottom-y) ───────────────────────────────
    def _cluster_rows(self, boxes):
        """Bottom-y'ye göre 1B kümele; her küme bir park sırası.

        Basit aglomeratif: sıralanmış bottom-y'lerde komşular medyan-h * band
        eşiğinden uzaksa yeni küme başlar.
        """
        if not boxes:
            return []
        if not self.multi_row:
            return [self._build_single_row(boxes)] if boxes else []

        heights = [b[3] - b[1] for b in boxes]
        med_h   = float(np.median(heights))
        # Kullanıcı slider'ı 0'a çekse bile kümeleme bozulmamalı.
        # Minimum eşik 0.30 × medyan araç yüksekliği olarak zorlanır.
        effective_ratio = max(0.30, self.row_band_ratio)
        threshold = effective_ratio * med_h

        sorted_b = sorted(boxes, key=lambda b: b[3])
        rows = [[sorted_b[0]]]
        for b in sorted_b[1:]:
            if b[3] - rows[-1][-1][3] <= threshold:
                rows[-1].append(b)
            else:
                rows.append([b])

        # Geçerli sıralar: min_cars_per_row üzeri
        rows = [sorted(r, key=lambda b: b[0])
                for r in rows if len(r) >= self.min_cars_per_row]

        # Her bottom-y row'u yatay gap'lere göre alt-lanelere böl
        # (yol genişliği üzerinden sol/sağ park sıraları birleşmesin).
        split_rows: list = []
        for r in rows:
            split_rows.extend(self._split_row_by_lateral_gaps(r))
        rows = [r for r in split_rows if len(r) >= self.min_cars_per_row]

        # En kalabalık max_rows sırayı tut, alt-y'ye göre sırala
        rows.sort(key=lambda r: -len(r))
        rows = rows[: self.max_rows]
        rows.sort(key=lambda r: float(np.median([b[3] for b in r])))
        return rows

    def _split_row_by_lateral_gaps(self, row):
        """Aynı bottom-y row'unu yatay gap'lere göre alt-lanelere böl.

        Sokak/ön kamera sahnesinde sol ve sağ taraf park sıraları benzer
        bottom-y'ye sahip ama aralarında YOL GENİŞLİĞİ kadar yatay boşluk
        vardır. Bu boşluk park boşluğu sayılmamalı — ayrı sublane.
        """
        if len(row) < 2:
            return [row]
        sorted_row = sorted(row, key=lambda b: (b[0] + b[2]) / 2)
        widths = [b[2] - b[0] for b in sorted_row]
        med_w = float(np.median(widths))
        threshold = self.lateral_split_ratio * med_w

        sublanes = [[sorted_row[0]]]
        for cur in sorted_row[1:]:
            prev = sublanes[-1][-1]
            gap = cur[0] - prev[2]
            if gap > threshold:
                sublanes.append([cur])
            else:
                sublanes[-1].append(cur)
        return sublanes

    def _build_single_row(self, boxes):
        if not boxes:
            return []
        bottoms = [b[3] for b in boxes]
        heights = [b[3] - b[1] for b in boxes]
        widths  = [b[2] - b[0] for b in boxes]
        med_bottom = float(np.median(bottoms))
        med_h      = float(np.median(heights))
        med_w      = float(np.median(widths))
        band = [b for b in boxes
                if abs(b[3] - med_bottom) <= self.row_band_ratio * med_h
                and 0.3 * med_w <= (b[2] - b[0]) <= 3.0 * med_w
                and 0.3 * med_h <= (b[3] - b[1]) <= 3.0 * med_h]
        if len(band) < self.min_cars_per_row:
            band = list(boxes)
        return sorted(band, key=lambda b: b[0])

    # ── Perspektif: x → genişlik lineer fit ───────────────────────
    @staticmethod
    def _perspective_fit(row):
        """Row içindeki araç merkezlerinden (cx → width) lineer fit.

        Yan kamerada kaybolma noktasına doğru araçlar büyür/küçülür.
        En az 3 örnek varsa lineer fit; yoksa medyan (sabit) döner.
        Çağıran: width_at(x) ve height_at(x).
        """
        widths  = np.array([b[2] - b[0] for b in row], dtype=float)
        heights = np.array([b[3] - b[1] for b in row], dtype=float)
        cxs     = np.array([(b[0] + b[2]) / 2 for b in row], dtype=float)
        med_w   = float(np.median(widths))
        med_h   = float(np.median(heights))

        if len(row) >= 3 and np.ptp(cxs) > 1e-3:
            kw, bw = np.polyfit(cxs, widths, 1)
            kh, bh = np.polyfit(cxs, heights, 1)
        else:
            kw, bw, kh, bh = 0.0, med_w, 0.0, med_h

        def width_at(x: float) -> float:
            v = kw * x + bw
            return float(np.clip(v, 0.4 * med_w, 2.5 * med_w))

        def height_at(x: float) -> float:
            v = kh * x + bh
            return float(np.clip(v, 0.4 * med_h, 2.5 * med_h))

        return width_at, height_at, med_w, med_h

    # ── Yol yüzeyi maskesi (connected component tabanlı) ──────────
    def _build_road_mask(self, frame, parked):
        """Tüm frame için yol/asfalt binary maskesi (downsample + cache).

        Performans:
          - Frame `road_mask_scale` ile küçültülür → cvtColor/inRange/morph/CCL
            örn. %50 oranında 4x daha hızlı.
          - Mask `road_mask_period` analizde bir yenilenir (sahne yavaş değişir).
        """
        self._analyze_count += 1
        # Cache geçerli mi?
        if (self._road_mask_cache is not None
                and self._analyze_count % self.road_mask_period != 0):
            return self._road_mask_cache

        if not parked:
            return self._road_mask_cache  # eski cache kalsın (yoksa None)

        H, W = frame.shape[:2]
        s = self.road_mask_scale
        if s <= 0 or s >= 1.0:
            small = frame
            scale = 1.0
        else:
            small = cv2.resize(frame, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA)
            scale = s
        h_s, w_s = small.shape[:2]

        # LAB renk uzayı: L (luminance) ayrı; a/b (renk) aydınlatma-bağımsız.
        # L'ye CLAHE uygula (lokal kontrast normalizasyonu) → gölge/parlaklık
        # değişimlerine karşı çok daha kararlı maskeleme.
        lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab_small)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        L_eq = clahe.apply(L)
        lab_eq = cv2.merge((L_eq, A, B))

        samples_lab = []
        for b in parked:
            bh = b[3] - b[1]
            strip_h = max(4.0, 0.30 * bh)
            sy1 = int(min(H - 1, b[3]) * scale)
            sy2 = int(min(H, b[3] + strip_h) * scale)
            sx1 = int(max(0, b[0] + 0.1 * (b[2] - b[0])) * scale)
            sx2 = int(min(W, b[2] - 0.1 * (b[2] - b[0])) * scale)
            sy1 = max(0, min(h_s - 1, sy1))
            sy2 = max(sy1 + 1, min(h_s, sy2))
            sx1 = max(0, min(w_s - 1, sx1))
            sx2 = max(sx1 + 1, min(w_s, sx2))
            if sy2 - sy1 < 2 or sx2 - sx1 < 2:
                continue
            roi = lab_eq[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            samples_lab.append(roi.reshape(-1, 3))

        if not samples_lab:
            return self._road_mask_cache

        all_lab = np.concatenate(samples_lab, axis=0)
        if all_lab.shape[0] > 4000:
            idx = np.random.choice(all_lab.shape[0], 4000, replace=False)
            all_lab = all_lab[idx]
        med = np.median(all_lab, axis=0)
        std = np.std(all_lab, axis=0)
        # L: geniş tolerans — gölgedeki aracu00e7 altı ile güneşli boş alan arası
        # ~30-50 L birimi fark olabilir (iç gesti test: 32 optimum)
        # a/b: dar — renksiz asfalt a/b ≈ 0, tu011ffı kızıl/yeşill yolu diye
        tol_L = max(32, min(65, std[0] * 2.2 + 18))
        tol_a = max(8,  min(20, std[1] * 1.8 + 5))
        tol_b = max(8,  min(20, std[2] * 1.8 + 5))
        lo = np.array([max(0, med[0] - tol_L),
                       max(0, med[1] - tol_a),
                       max(0, med[2] - tol_b)], dtype=np.uint8)
        hi = np.array([min(255, med[0] + tol_L),
                       min(255, med[1] + tol_a),
                       min(255, med[2] + tol_b)], dtype=np.uint8)

        mask = cv2.inRange(lab_eq, lo, hi)

        # Morfolojik temizleme küçük frame'de (hızlı)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return self._road_mask_cache
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        small_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

        # Orijinal boyuta NEAREST ile geri aç (binary mask — anti-alias gereksiz)
        if scale < 1.0:
            road_mask = cv2.resize(small_mask, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            road_mask = small_mask

        self._road_mask_cache = road_mask
        return road_mask

    def _slot_at_road_center(self, road_mask, slot) -> bool:
        """Slot center x, drivable area'nın yatay merkezine çok yakınsa True.

        Yol merkezi park yeri olamaz — slot'un alt yarısı yüksekliğinde
        drivable maske içinde X-aralığının medyanı yolun merkezini verir;
        slot center bu merkeze `road_center_reject_ratio × W` kadar yakınsa
        reddedilir.
        """
        if road_mask is None or self.road_center_reject_ratio <= 0:
            return False
        h, w = road_mask.shape[:2]
        x1, y1, x2, y2 = slot
        sh = y2 - y1
        sy1 = int(max(0, y2 - 0.5 * sh))
        sy2 = int(min(h, y2))
        if sy2 - sy1 < 2:
            return False
        band = road_mask[sy1:sy2, :]
        if band.size == 0:
            return False
        col_sum = (band > 0).sum(axis=0)
        if col_sum.sum() == 0:
            return False
        xs = np.arange(band.shape[1], dtype=np.float64)
        center_x = float((col_sum * xs).sum() / col_sum.sum())
        slot_cx = 0.5 * (x1 + x2)
        return abs(slot_cx - center_x) < self.road_center_reject_ratio * w

    def _slot_in_road_mask(self, road_mask, slot, min_ratio: float = 0.20) -> bool:
        """Slot alt yarısı road maskesinin %min_ratio kadarını içermeli."""
        if road_mask is None:
            return True
        h, w = road_mask.shape[:2]
        x1, y1, x2, y2 = slot
        sh = y2 - y1
        sy1 = int(max(0, y2 - 0.55 * sh))
        sy2 = int(min(h, y2))
        sx1 = int(max(0, x1 + 0.05 * (x2 - x1)))
        sx2 = int(min(w, x2 - 0.05 * (x2 - x1)))
        if sy2 - sy1 < 2 or sx2 - sx1 < 2:
            return True
        sub = road_mask[sy1:sy2, sx1:sx2]
        if sub.size == 0:
            return True
        ratio = float(np.count_nonzero(sub)) / float(sub.size)
        return ratio >= min_ratio

    # ── Yol yüzeyi renk referansı (eski yöntem, fallback) ─────────
    def _road_color_ref(self, frame, parked):
        """Park etmiş araçların hemen altındaki şeritlerin HSV-medyanı.

        Yol yüzeyinin (asfalt/parke) tipik renk dağılımını yakalar.
        Hiç örnek yoksa None döner — bu durumda renk kontrolü atlanır.
        """
        if not parked:
            return None
        h, w = frame.shape[:2]
        samples = []
        for b in parked:
            bh = b[3] - b[1]
            strip_h = max(4.0, 0.25 * bh)
            sy1 = int(min(h - 1, b[3]))
            sy2 = int(min(h, b[3] + strip_h))
            sx1 = int(max(0, b[0] + 0.1 * (b[2] - b[0])))
            sx2 = int(min(w, b[2] - 0.1 * (b[2] - b[0])))
            if sy2 - sy1 < 2 or sx2 - sx1 < 2:
                continue
            roi = frame[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            samples.append(np.median(hsv.reshape(-1, 3), axis=0))
        if not samples:
            return None
        return np.median(np.array(samples), axis=0)

    def _slot_looks_like_road(self, frame, slot, ref) -> bool:
        """Slot'un alt yarısında yol benzeri renk var mı?

        Kontrol 1: Açık yeşill bitki/çim piksel oranı (bahçe/bollard reddi)
        Kontrol 2: Renk benzerliği — sadece ref verilmişse
        """
        if not self.road_color_check:
            return True
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = slot
        sh = y2 - y1
        sy1 = int(max(0, y2 - 0.45 * sh))
        sy2 = int(min(h, y2))
        sx1 = int(max(0, x1 + 0.08 * (x2 - x1)))
        sx2 = int(min(w, x2 - 0.08 * (x2 - x1)))
        if sy2 - sy1 < 2 or sx2 - sx1 < 2:
            return True
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return True
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        flat = hsv.reshape(-1, 3).astype(np.float32)

        # --- Kontrol 1: Yeşil bitki tespiti ---
        # Hue 35-85 (yeşil tonlar) VE saturation >= 55 (doygun yeşil)
        # Bu çift koşul: beyaz çizgiler (S≈0), gri asfalt (S≈0-20) ve
        # gölgeli zeminleri YANLIŞLIKLA bitki saymaz.
        green_mask = (flat[:, 0] >= 35) & (flat[:, 0] <= 85) & (flat[:, 1] >= 70)
        if np.mean(green_mask) > 0.40:   # >%40 doygun yeşil piksel → bitki alanı
            return False

        # --- Kontrol 2: Renk benzerliği (yalnızca referans verilmişse) ---
        if ref is None:
            return True
        med = np.median(flat, axis=0)
        dh = abs(med[0] - ref[0])
        dh = min(dh, 180 - dh)
        ds = abs(med[1] - ref[1])
        dv = abs(med[2] - ref[2])
        return (dh <= self.road_color_tol_h
                and ds <= self.road_color_tol_s
                and dv <= self.road_color_tol_v)

    # ── Engel kontrolü ────────────────────────────────────────────
    def _slot_blocked(self, slot, obstacles) -> bool:
        """Slot içinde engel merkezi var mı veya yeterli alan örtüşmesi mi?"""
        if not obstacles:
            return False
        sx1, sy1, sx2, sy2 = slot
        slot_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
        for o in obstacles:
            ox1, oy1, ox2, oy2 = o["bbox"]
            cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            if _point_in_box(cx, cy, (sx1, sy1, sx2, sy2)):
                return True
            ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
            ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            if (iw * ih) / slot_area >= self.obstacle_overlap_min:
                return True
        return False

    # ── Ham tespit ────────────────────────────────────────────────
    def _detect_in_row(self, row, frame, obstacles, road_ref, road_mask):
        h, w = frame.shape[:2]
        empty = []
        if not row:
            return empty

        width_at, height_at, med_w, med_h = self._perspective_fit(row)
        frame_edge_margin = self.frame_edge_margin_ratio * w
        max_edge_ext      = self.max_edge_extension_ratio * w

        def _add(gap_x1: float, gap_x2: float, base_bot: float) -> None:
            gap_w = gap_x2 - gap_x1
            if gap_w <= 0:
                return
            mid_x = (gap_x1 + gap_x2) / 2
            local_w = width_at(mid_x)
            local_h = height_at(mid_x)
            if gap_w < self.min_gap_ratio * local_w:
                return
            n = max(1, int(round(gap_w / local_w)))
            n = min(n, self.max_spaces_per_gap)
            y2 = int(min(h, base_bot))
            y1 = int(max(0, base_bot - local_h))
            step = gap_w / n
            for j in range(n):
                sx1 = int(gap_x1 + j * step)
                sx2 = int(gap_x1 + (j + 1) * step)
                # Slot en-boy oranı: yan görüş için 0.25-6.0 aralığı dışı
                # son derece dar veya çok geniş kutular sahte pozitiftir.
                slot_w = sx2 - sx1
                slot_h = y2 - y1
                if slot_h > 0 and not (0.25 <= slot_w / slot_h <= 6.0):
                    continue
                slot = (sx1, y1, sx2, y2)
                if self._slot_blocked(slot, obstacles):
                    continue
                # Bitki/çim alanı kontrolü (bahçe, park, yeşil alan reddi)
                if not self._slot_looks_like_road(frame, slot, None):
                    continue
                empty.append(slot)

        # Bitişik araç-arası
        for i in range(len(row) - 1):
            left  = row[i]
            right = row[i + 1]
            gap_x1 = left[2]
            gap_x2 = right[0]
            gap_w  = gap_x2 - gap_x1
            mid_x  = (gap_x1 + gap_x2) / 2
            local_w = width_at(mid_x)
            local_h = height_at(mid_x)

            if gap_w < self.min_gap_ratio * local_w:
                continue
            if gap_w > self.max_gap_ratio * local_w:
                continue
            if frame_edge_margin > 0 and (
                gap_x1 < frame_edge_margin or gap_x2 > w - frame_edge_margin
            ):
                continue
            if abs(left[3] - right[3]) > self.bottom_align_tol * local_h:
                continue

            base_bot = (left[3] + right[3]) / 2
            _add(gap_x1, gap_x2, base_bot)

        # Sıra uçları
        if self.detect_row_edges and row:
            leftmost  = row[0]
            rightmost = row[-1]

            left_gap_x2 = leftmost[0]
            left_gap_x1 = max(0.0, left_gap_x2 - max_edge_ext)
            _add(left_gap_x1, left_gap_x2, leftmost[3])

            right_gap_x1 = rightmost[2]
            right_gap_x2 = min(float(w), right_gap_x1 + max_edge_ext)
            _add(right_gap_x1, right_gap_x2, rightmost[3])

        return empty

    def _detect_raw(self, frame, detections, obstacles, static_mask,
                    external_road_mask=None):
        candidates = self._filter_candidates(frame.shape, detections, static_mask)
        # Fallback: static_mask çok sıkı veya tracker henüz olgunlaşmadıysa
        # tüm araç tespitlerini kullan ki park sırası yine de oluşsun.
        if not candidates and detections:
            candidates = self._filter_candidates(frame.shape, detections, None)
        rows       = self._cluster_rows(candidates)

        # Yol yüzeyi öncelik sırası:
        #   1) external_road_mask (YOLOPv2 drivable area — en güvenilir)
        #   2) klasik LAB+CLAHE connected-component maskesi
        #   3) renk referansı (fallback)
        all_parked: list = []
        for r in rows:
            all_parked.extend(r)
        if external_road_mask is not None:
            road_mask = external_road_mask
            # Çift doğrulama: Hem dış maske hem de renk referansı hesaplanır.
            # road_mask geçmesi gerekli ama AYRICA renk benzerliği de kontrol edilir.
            road_ref = (self._road_color_ref(frame, all_parked)
                        if self.road_color_check else None)
        else:
            road_mask = (self._build_road_mask(frame, all_parked)
                         if self.road_color_check else None)
            road_ref  = (self._road_color_ref(frame, all_parked)
                         if self.road_color_check else None)
        self._last_road_mask = road_mask  # debug/görsel için

        empty_spaces: list[tuple[int, int, int, int]] = []
        parked: list = []
        for row in rows:
            parked.extend(row)
            empty_spaces.extend(
                self._detect_in_row(row, frame, obstacles, road_ref, road_mask)
            )

        # Aynı slotu birden çok sıra üretebilir: yüksek IoU duplikalarını ele
        deduped: list = []
        for s in empty_spaces:
            if not any(_bbox_iou(s, e) > 0.6 for e in deduped):
                deduped.append(s)

        return deduped, parked, rows

    # ── Smoothing ─────────────────────────────────────────────────
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

    # ── Ölçek tahmini (park edilmiş araçlar → piksel/metre) ──────
    @staticmethod
    def estimate_scale(rows: list, ref_car_length_m: float = 4.5) -> float | None:
        """Park sırasındaki araç piksel genişliklerinden m/px ölçeği hesapla.

        Yan kamera görüşünde araç piksel genişliği ≈ araç uzunluğu.
        ref_car_length_m: referans araç uzunluğu (varsayılan: ortalama otomobil 4.5m).
        Döner: m/px (None = yeterli veri yok).
        """
        widths_px = []
        for row in rows:
            for b in row:
                widths_px.append(b[2] - b[0])
        if not widths_px:
            return None
        med_px = float(np.median(widths_px))
        if med_px < 1:
            return None
        return ref_car_length_m / med_px  # metre / piksel

    # ── Public API ────────────────────────────────────────────────
    def analyze(
        self,
        frame: np.ndarray,
        detections: list[dict],
        obstacles: list[dict] | None = None,
        static_mask: list[bool] | None = None,
        external_road_mask: np.ndarray | None = None,
        ref_car_length_m: float = 4.5,
    ) -> dict:
        raw_empty, parked, rows = self._detect_raw(
            frame, detections, obstacles or [], static_mask,
            external_road_mask=external_road_mask,
        )
        confirmed = self._confirm(raw_empty)

        # Ölçek tahmini ve slot boyutları (metre cinsinden)
        scale = self.estimate_scale(rows, ref_car_length_m)
        slot_sizes_m: list[tuple[float, float]] = []
        for (x1, y1, x2, y2) in confirmed:
            if scale is not None:
                w_m = (x2 - x1) * scale
                h_m = (y2 - y1) * scale
            else:
                w_m = h_m = 0.0
            slot_sizes_m.append((round(w_m, 2), round(h_m, 2)))

        return {
            "parked":         parked,
            "rows":           rows,
            "empty_spaces":   confirmed,
            "raw_candidates": raw_empty,
            "empty_count":    len(confirmed),
            "occupied_count": len(parked),
            "obstacles":      obstacles or [],
            "slot_sizes_m":   slot_sizes_m,   # [(uzunluk_m, derinlik_m), ...]
            "scale_m_per_px": scale,           # None = kalibre edilemedi
        }

    def draw(
        self,
        frame: np.ndarray,
        result: dict,
        car_length_m: float | None = None,
    ) -> np.ndarray:
        """Boş slotları çiz; araç uzunluğu verilmişse sığma durumunu göster."""
        out = frame.copy()
        spaces     = result["empty_spaces"]
        sizes_m    = result.get("slot_sizes_m", [])
        scale      = result.get("scale_m_per_px")
        check_fit  = (car_length_m is not None and scale is not None and scale > 0)

        if not spaces:
            return out

        COLOR_FIT   = (0, 220, 80)    # yeşil  — sığar
        COLOR_NOFIT = (0, 60, 220)    # kırmızı — sığmaz
        COLOR_UNK   = COLOR_EMPTY     # ölçek bilinmiyor

        overlay = out.copy()
        for i, (x1, y1, x2, y2) in enumerate(spaces):
            if check_fit and i < len(sizes_m):
                w_m = sizes_m[i][0]
                fits = w_m >= car_length_m
                color = COLOR_FIT if fits else COLOR_NOFIT
            else:
                color = COLOR_UNK
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        for i, (x1, y1, x2, y2) in enumerate(spaces):
            if check_fit and i < len(sizes_m):
                w_m = sizes_m[i][0]
                fits = w_m >= car_length_m
                color = COLOR_FIT if fits else COLOR_NOFIT
                label = f"{'SIGAR' if fits else 'SIGMAZ'} {w_m:.1f}m"
            else:
                color = COLOR_UNK
                label = "BOS"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            font_scale = max(0.4, min(0.7, (x2 - x1) / 200))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.rectangle(out,
                          (cx - tw // 2 - 4, cy - th // 2 - 4),
                          (cx + tw // 2 + 4, cy + th // 2 + 4),
                          color, -1)
            cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out
