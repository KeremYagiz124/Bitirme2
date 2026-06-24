"""Adaptif slot tespiti — çizgi varsa ızgara, yoksa geometri.

Tek bir giriş noktası altında iki yöntemi birleştirir:

  - Çizgili otoparklar: park şeritlerinden ızgara çıkarılır (yüksek doğruluk).
    IPM verilirse kuş bakışında çalışır; perspektif kalktığı için şeritler
    temiz bulunur, slotlar geri perspektife haritalanıp canlı kareye çizilir.
  - Çizgisiz alanlar (yol kenarı vb.): mevcut geometri tabanlı
    StreetParkingDetector'a düşülür.

Karar her karede has_lines() ile verilir → tamamen adaptif, canlı çalışır.

Sonuç sözlüğü (tüm koordinatlar KAYNAK görüntüde):
    method:         "line" | "geometry"
    empty_spaces:   [(x1,y1,x2,y2), ...]   eksen-hizalı, sayım/sığma/çizim için
    empty_polys:    [(4,2) köşe, ...]       perspektif poligon (şık çizim)
    occupied_polys: [(4,2) köşe, ...]
    empty_count, occupied_count
"""

from __future__ import annotations

import numpy as np
import cv2

from src.detection.parking_line_detector import ParkingLineDetector
from src.detection.street_parking_detector import StreetParkingDetector
from src.detection.temporal_voter import TemporalSlotVoter
from src.detection.grid_fusion import GridLineFusion


def _quad_to_aabb(quad):
    xs, ys = quad[:, 0], quad[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _bbox_to_quad(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


class AdaptiveSlotDetector:
    def __init__(self, line_detector: ParkingLineDetector | None = None,
                 street_detector: StreetParkingDetector | None = None,
                 min_vertical_lines: int = 5, overlap_thresh: float = 0.15,
                 voter: TemporalSlotVoter | None = None, use_voting: bool = True,
                 fusion: GridLineFusion | None = None, use_fusion: bool = True,
                 min_paint_ratio: float = 0.35):
        self.line = line_detector or ParkingLineDetector()
        self.street = street_detector or StreetParkingDetector()
        self.min_vertical_lines = min_vertical_lines
        self.overlap_thresh = overlap_thresh
        self.voter = voter or (TemporalSlotVoter() if use_voting else None)
        self.fusion = fusion or (GridLineFusion() if use_fusion else None)
        self.min_paint_ratio = min_paint_ratio
        self._mode_history = []
        self._current_mode = "geometry"
        self._history_len = 20
        self._last_line_slots_data = None

    def reset(self):
        """Kaynak değiştiğinde (yeni görüntü/video) zamansal geçmişi sıfırla."""
        if self.voter is not None:
            self.voter.reset()
        if self.fusion is not None:
            self.fusion.reset()
        self.street.reset_history()
        self._mode_history = []
        self._current_mode = "geometry"
        self._last_line_slots_data = None

    def _line_slots(self, img, xs, ys):
        """Izgara çizgilerini (füzyon varsa füzyonlayıp) slotlara çevir."""
        if self.fusion is not None:
            xs, ys = self.fusion.update(xs, ys)
        return self.line.build_slots_from_positions(xs, ys, img.shape)



    # ── Yardımcı: çizgi yöntemi sonucu paketle ───────────────────────────────
    def _pack_line_result(self, classified, to_source, size_fn=None,
                          vehicle_boxes=None):
        """classified slotları kaynağa haritalayıp sonuç sözlüğü üret.

        to_source: BEV bbox → kaynak (4,2) köşe fonksiyonu (IPM yoksa birim).
        size_fn: BEV bbox → (genişlik_m, derinlik_m) veya None. IPM kalibreyse
        verilir; boş slotların gerçek boyutu (sığma kontrolü için) hesaplanır.
        vehicle_boxes: kaynak uzayda tam araç kutuları. Bir slot araca denk
        geliyorsa (ör. van gövdesindeki şeritler) DOLU sayılır — boş gösterilmez.
        """
        empty_polys, occupied_polys, empty_spaces, empty_sizes_m = [], [], [], []
        confs = []
        occ = 0
        for s in classified:
            quad = to_source(s["bbox"])
            aabb = _quad_to_aabb(quad)
            occupied = bool(s["occupied"])
            # Çizgi-ızgara slotu bir araç kutusuna denk geliyorsa DOLU say
            # (BEV örtüşme sınıflaması kaçırmış olabilir).
            if (not occupied and vehicle_boxes
                    and StreetParkingDetector._slot_blocked_by_vehicle(
                        aabb, vehicle_boxes)):
                occupied = True
            if occupied:
                occupied_polys.append(quad)
                occ += 1
            else:
                empty_polys.append(quad)
                empty_spaces.append(aabb)
                empty_sizes_m.append(size_fn(s["bbox"]) if size_fn else None)
            # Karar güveni: dolu ise örtüşme skoru, boş ise (1 - skor)
            sc = s.get("score")
            if sc is not None:
                confs.append(sc if occupied else (1.0 - sc))
        mean_conf = float(np.mean(confs)) if confs else None
        return {
            "method": "line",
            "empty_spaces": empty_spaces,
            "empty_polys": empty_polys,
            "empty_sizes_m": empty_sizes_m,
            "occupied_polys": occupied_polys,
            "empty_count": len(empty_spaces),
            "occupied_count": occ,
            "mean_confidence": mean_conf,
        }

    # ── Geometri yöntemi sonucunu birleşik formata çevir ─────────────────────
    @staticmethod
    def _pack_geometry_result(geo: dict, all_vehicles=None):
        empty_spaces = list(geo.get("empty_spaces", []))
        parked = list(geo.get("parked", []))
        
        # Herhangi bir tespit edilen araç varsa (ve parked listesinde henüz yoksa),
        # onu da occupied_polys listesine ekliyoruz (dolu olarak kırmızı çizilsin diye).
        if all_vehicles:
            for v in all_vehicles:
                v_box = [float(x) for x in v]
                is_already_parked = False
                for p in parked:
                    p_box = [float(x) for x in p]
                    ax1, ay1, ax2, ay2 = v_box
                    bx1, by1, bx2, by2 = p_box
                    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
                    iou = inter / union if union > 0 else 0.0
                    
                    min_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                    iom = inter / min_area if min_area > 0 else 0.0
                    
                    if iou > 0.35 or iom > 0.85:
                        is_already_parked = True
                        break
                if not is_already_parked:
                    parked.append(v_box)
                    
        return {
            "method": "geometry",
            "empty_spaces": empty_spaces,
            "empty_polys": [_bbox_to_quad(b) for b in empty_spaces],
            "empty_sizes_m": geo.get("slot_sizes_m", []),
            "occupied_polys": [_bbox_to_quad(b) for b in parked],
            "empty_count": len(empty_spaces),
            "occupied_count": len(parked),
            "slot_sizes_m": geo.get("slot_sizes_m", []),
            "scale_m_per_px": geo.get("scale_m_per_px"),
            "mean_confidence": None,
        }

    def _color_confidence(self, img: np.ndarray, segments: list[tuple[int, int, int, int]]) -> float:
        if not segments or img is None or img.size == 0:
            return 0.0
            
        # 1) Get the reference mask for valid paint pixels
        if img.ndim == 3 and getattr(self.line, "use_color", True):
            ref_mask = self.line._color_line_mask(img)
        else:
            # Grayscale fallback: use brightness threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            _, ref_mask = cv2.threshold(gray, getattr(self.line, "white_thresh", 170), 255, cv2.THRESH_BINARY)
            
        # 2) Draw all segments on a blank canvas
        canvas = np.zeros(ref_mask.shape, dtype=np.uint8)
        for s in segments:
            cv2.line(canvas, (s[0], s[1]), (s[2], s[3]), 255, thickness=2)
            
        # 3) Calculate overlap ratio
        overlap = cv2.bitwise_and(ref_mask, canvas)
        overlap_pixels = np.count_nonzero(overlap)
        total_pixels = np.count_nonzero(canvas)
        
        if total_pixels == 0:
            return 0.0
        return overlap_pixels / total_pixels

    # ── Ana giriş ────────────────────────────────────────────────────────────
    def analyze(self, frame, detections, ipm=None, static_mask=None,
                external_road_mask=None, obstacles=None,
                ref_car_length_m: float = 4.5, ref_car_width_m: float = 2.0):
        vehicle_bboxes = [d["bbox"] for d in detections]
        # Kaynak uzayda tam araç kutuları — perspektif düzeltilmiş (dar) kutuları
        # kullan, aksi halde raw_bbox geniş olup komşu boş slotları engelleyebilir.
        _VEH_CLS = {2, 3, 5, 7}
        veh_src = [d["bbox"] for d in detections
                   if d.get("class_id") in _VEH_CLS]

        # 1) Çizgileri ara ve ham mod adayını belirle
        raw_mode = "geometry"
        bev = None
        xs, ys = [], []

        if ipm is not None:
            try:
                bev = ipm.warp_image(frame)
            except Exception:
                bev = None
            if bev is not None:
                xs, ys = self.line.grid_lines(bev)
                if len(xs) >= self.min_vertical_lines:
                    raw_mode = "line"
        else:
            xs, ys = self.line.grid_lines(frame)
            if len(xs) >= self.min_vertical_lines and len(ys) >= 2:
                raw_mode = "line"

        # 1.1) Dinamik Çizgi Güven Kontrolü: Çizgi modu adaysa boyalı şerit kalitesini doğrula
        if raw_mode == "line":
            img_to_check = bev if (ipm is not None and bev is not None) else frame
            h_tc, w_tc = img_to_check.shape[:2]
            max_w = 640
            if w_tc > max_w:
                scale = max_w / float(w_tc)
                img_small = cv2.resize(img_to_check, (max_w, int(h_tc * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                img_small = img_to_check
                scale = 1.0
                
            segments = self.line.detect_segments(img_small)
            verticals, _ = self.line._split_orientation(segments)
            paint_ratio = self._color_confidence(img_small, verticals)
            
            # Eğer dikey çizgilerde yeterince boyalı piksel yoksa, mod geçişini iptal et ve geometriye düş
            if paint_ratio < self.min_paint_ratio:
                raw_mode = "geometry"

        # 2) Zamansal oylama / histeresiz filtresi uygulayarak anlık geçişleri önle
        if not hasattr(self, "_mode_history") or self._mode_history is None:
            self._mode_history = []
            self._current_mode = raw_mode
            self._history_len = 20
            self._last_line_slots_data = None

        self._mode_history.append(raw_mode)
        if len(self._mode_history) > self._history_len:
            self._mode_history.pop(0)

        line_ratio = self._mode_history.count("line") / len(self._mode_history)
        if self._current_mode == "geometry" and line_ratio >= 0.9:
            self._current_mode = "line"
        elif self._current_mode == "line" and line_ratio <= 0.5:
            self._current_mode = "geometry"

        # 3) Belirlenen modu çalıştır
        res = None
        if self._current_mode == "line":
            has_enough_lines = len(xs) >= self.min_vertical_lines if ipm is not None else (len(xs) >= self.min_vertical_lines and len(ys) >= 2)
            if has_enough_lines:
                if ipm is not None and bev is not None:
                    veh_bev = [ipm.transform_box(b, ref_car_length_m) for b in vehicle_bboxes]
                    slots = self._line_slots(bev, xs, ys)
                    classified = self.line.classify_slots(
                        slots, veh_bev, self.overlap_thresh)
                    if self.voter is not None:
                        classified = self.voter.update(classified)
                    mpp = ipm.m_per_px
                    size_fn = (
                        (lambda b: ((b[2] - b[0]) * mpp, (b[3] - b[1]) * mpp))
                        if mpp else None)
                    self._last_line_slots_data = (classified, size_fn)
                    res = self._pack_line_result(
                        classified, lambda b: ipm.inverse_transform_quad(b),
                        size_fn=size_fn, vehicle_boxes=veh_src)
                else:
                    slots = self._line_slots(frame, xs, ys)
                    classified = self.line.classify_slots(
                        slots, vehicle_bboxes, self.overlap_thresh)
                    if self.voter is not None:
                        classified = self.voter.update(classified)
                    self._last_line_slots_data = (classified, None)
                    res = self._pack_line_result(classified, _bbox_to_quad,
                                                  vehicle_boxes=veh_src)
            elif self._last_line_slots_data is not None:
                # Geçici olarak çizgiler kaybolsa da mod çizgi modunda kaldığı için eski slotları koru
                classified, size_fn = self._last_line_slots_data
                to_src_fn = (lambda b: ipm.inverse_transform_quad(b)) if ipm is not None else _bbox_to_quad
                res = self._pack_line_result(classified, to_src_fn,
                                              size_fn=size_fn, vehicle_boxes=veh_src)

        if res is None:
            # Geometri yöntemine düş
            geo = self.street.analyze(
                frame, detections, static_mask=static_mask,
                external_road_mask=external_road_mask, obstacles=obstacles,
                ref_car_length_m=ref_car_length_m, ref_car_width_m=ref_car_width_m)
            res = self._pack_geometry_result(geo, all_vehicles=veh_src)

        return res
