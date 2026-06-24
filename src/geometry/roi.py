"""İlgi bölgesi (ROI) — analizi otopark alanına kısıtlar.

Kullanıcı görüntüde bir poligon (otopark sınırı) tanımlar. Bu poligon dışındaki
araç tespitleri ve slotlar elenir; arka plandaki yol/bina/gökyüzü kaynaklı
sahte pozitifler düşer ve yalnızca ilgili bölge işlenir.

Saf numpy/cv2; bağımsız test edilebilir.
"""

from __future__ import annotations

import cv2
import numpy as np


def _poly(polygon):
    return np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)


def point_in_polygon(point, polygon) -> bool:
    """Nokta poligonun içinde (veya kenarında) mı?"""
    p = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(p, (float(point[0]), float(point[1])), False) >= 0


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def filter_boxes(boxes, polygon):
    """Merkezi poligon içinde olan bbox'ları tut."""
    if polygon is None or len(polygon) < 3:
        return list(boxes)
    return [b for b in boxes if point_in_polygon(_center(b), polygon)]


def filter_detections(detections, polygon):
    """Merkezi poligon içinde olan tespitleri tut."""
    if polygon is None or len(polygon) < 3:
        return list(detections)
    return [d for d in detections
            if point_in_polygon(_center(d["bbox"]), polygon)]


def auto_roi_from_detections(detections, frame_shape, margin_frac: float = 0.06,
                             min_dets: int = 1):
    """Tespit edilen araçların dış bükey zarfından otomatik ROI poligonu üret.

    Tüm araç bbox köşeleri toplanır, convex hull alınır ve merkezden dışa doğru
    `margin_frac` kadar genişletilip kare sınırlarına kırpılır. Manuel poligona
    gerek kalmadan ilgi bölgesi oluşur.

    Döner: poligon nokta listesi [(x,y), ...] veya yetersiz tespitte None.
    """
    if not detections or len(detections) < min_dets:
        return None
    pts = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        pts += [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    pts = np.array(pts, dtype=np.float32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    h, w = frame_shape[:2]
    m = margin_frac * max(h, w)
    poly = []
    for p in hull:
        v = p - c
        n = float(np.linalg.norm(v))
        q = p if n < 1e-6 else p + v / n * m
        poly.append((float(min(max(q[0], 0), w - 1)),
                     float(min(max(q[1], 0), h - 1))))
    return poly


def draw_roi(frame, polygon, dim: float = 0.45):
    """Poligon dışını karart, sınırı çiz. Döner: yeni görüntü."""
    if polygon is None or len(polygon) < 3:
        return frame
    poly = _poly(polygon)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    out = frame.copy()
    dark = (frame.astype(np.float32) * dim).astype(np.uint8)
    out[mask == 0] = dark[mask == 0]
    cv2.polylines(out, [poly], True, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def line_segments_intersect(p1, p2, p3, p4) -> bool:
    """İki doğru parçasının kesişip kesişmediğini kontrol et."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def segment_intersects_bbox(p1, p2, bbox) -> bool:
    """Bir doğru parçasının bir bounding box ile kesişip kesişmediğini denetle."""
    rx1, ry1, rx2, ry2 = bbox
    # Uç noktalardan biri kutunun içinde mi?
    for px, py in (p1, p2):
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            return True
    # Kutunun 4 kenarı ile kesişim kontrolü
    edges = [
        ((rx1, ry1), (rx2, ry1)),
        ((rx1, ry2), (rx2, ry2)),
        ((rx1, ry1), (rx1, ry2)),
        ((rx2, ry1), (rx2, ry2))
    ]
    for e_p1, e_p2 in edges:
        if line_segments_intersect(p1, p2, e_p1, e_p2):
            return True
    return False


def filter_occluded_slots(empty_spaces, vehicle_boxes, frame_shape):
    """Ray-casting tabanlı görüş engeli filtresi.

    Kameranın görüntünün alt-orta noktasında (W/2, H) olduğunu varsayar.
    Kameradan slot merkezine çizilen ışının ön plandaki bir araç kutusuyla kesişip kesişmediğini denetler.
    Döner: Görüşü açık olan slotların indeksleri.
    """
    if not empty_spaces:
        return []
    h, w = frame_shape[:2]
    camera_pt = (w / 2.0, float(h))
    kept_indices = []

    for i, s_box in enumerate(empty_spaces):
        scx = (s_box[0] + s_box[2]) / 2.0
        scy = (s_box[1] + s_box[3]) / 2.0
        slot_pt = (scx, scy)

        slot_h = s_box[3] - s_box[1]
        is_blocked = False
        for v_box in vehicle_boxes:
            # Sadece slottan belirgin şekilde daha yakın (alt-y koordinatı daha büyük) olan araçları engelleme adayı al.
            # Aynı sıradaki yan araçların yanlışlıkla oklüzyon engeli olarak değerlendirilmemesi için
            # araç alt-y koordinatının slot alt-y'sinden en az 1.2 * slot_yüksekliği kadar aşağıda olması gerekir.
            if v_box[3] > s_box[3] + 1.2 * slot_h:
                if segment_intersects_bbox(camera_pt, slot_pt, v_box):
                    is_blocked = True
                    break

        if not is_blocked:
            kept_indices.append(i)

    return kept_indices
