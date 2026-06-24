"""Otomatik IPM kalibrasyonu — kaybolma noktası / yakınsayan çizgilerden.

Manuel 4-nokta yerine, zemindeki derinlik yönünde uzanan (kameradan uzaklaşıp
yakınsayan) çizgilerden bir trapez çıkarır ve onu dikdörtgene eşleyerek
homografi kurar. Kamera iç parametreleri gerekmez (pratik BEV yaklaşımı).

best-effort: yeterli/yakınsayan çizgi bulunamazsa None döner → çağıran manuel
kalibrasyona düşer. Hiçbir durumda hata fırlatmaz.
"""

from __future__ import annotations

import numpy as np

from src.detection.parking_line_detector import ParkingLineDetector
from src.geometry.ipm import PerspectiveTransformer


def _line_from_seg(seg):
    """Segment uçlarından normalize edilmiş çizgi (a, b, c): ax+by+c=0."""
    x1, y1, x2, y2 = seg
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    n = (a * a + b * b) ** 0.5
    if n < 1e-6:
        return None
    return a / n, b / n, c / n


def estimate_vanishing_point(lines):
    """Çizgilerin (a,b,c) en küçük kareler kesişimi (kaybolma noktası).

    Her çizgi a*x+b*y+c=0; [a b][x y]^T = -c. Aşırı belirtilmiş sistem çözülür.
    Döner: (x, y) veya çözülemezse None.
    """
    if len(lines) < 2:
        return None
    A = np.array([[l[0], l[1]] for l in lines], dtype=np.float64)
    rhs = np.array([-l[2] for l in lines], dtype=np.float64)
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None
    return float(sol[0]), float(sol[1])


def _x_at(line, y):
    a, b, c = line
    if abs(a) < 1e-6:
        return None
    return -(b * y + c) / a


def auto_calibrate(frame, out_w: int = 600, out_h: int = 800,
                   real_w_m: float | None = None, real_h_m: float | None = None,
                   line_detector: ParkingLineDetector | None = None,
                   angle_from_vertical_max: float = 55.0,
                   return_diagnostics: bool = False):
    """Yakınsayan çizgilerden otomatik IPM kur. Başarısızsa None döner."""
    try:
        ld = line_detector or ParkingLineDetector()
        h, w = frame.shape[:2]
        segments = ld.detect_segments(frame)
        if not segments:
            return (None, {}) if return_diagnostics else None

        # Derinlik yönü çizgileri: yataya yakın olmayanlar (dikeyden sapması
        # sınırlı). Yatay şeritler (ön/curb) elenir.
        lines, x_at_bottom = [], []
        vertical_segments = []
        y_near = 0.95 * h
        y_far = 0.55 * h
        for s in segments:
            x1, y1, x2, y2 = s
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # 0=yatay,90=dik
            dev = abs(90 - ang)  # dikeyden sapma
            if dev > angle_from_vertical_max:
                continue
            L = _line_from_seg(s)
            if L is None:
                continue
            xb = _x_at(L, y_near)
            if xb is None:
                continue
            lines.append(L)
            x_at_bottom.append(xb)
            vertical_segments.append(s)

        if len(lines) < 2:
            return (None, {}) if return_diagnostics else None

        # Yakınsama doğrulaması: kaybolma noktası örnekleme bandının üstünde olmalı
        vp = estimate_vanishing_point(lines)
        if vp is None or vp[1] > y_far:
            return (None, {}) if return_diagnostics else None

        # Kaybolma noktası (vanishing point) yardımıyla tam genişlikte trapez oluştur.
        # Bu, aşırı yakınlaşmış (zoomed-in) görüntüyü önler ve tüm sahneyi kuş bakışı yapar.
        vpx, vpy = vp
        # Enforce a symmetric, upright perspective projection (no diagonal roll/shear)
        vpx = w / 2.0

        # Altta tüm ekran genişliğini kapla: (0, y_near) ve (w, y_near)
        lb = 0.0
        rb = float(w)

        # Bu alt noktaları vanishing point'e bağlayan doğruların y_far seviyesindeki kesişimleri
        # x = x0 + (y - y0) * (vpx - x0) / (vpy - y0)
        dy_top = y_far - y_near
        dy_vp = vpy - y_near
        if abs(dy_vp) < 1e-3:
            return (None, {}) if return_diagnostics else None

        lt = lb + dy_top * (vpx - lb) / dy_vp
        rt = rb + dy_top * (vpx - rb) / dy_vp

        bottom_w = rb - lb
        top_w = rt - lt
        if bottom_w <= 10 or top_w <= 0 or top_w >= bottom_w:
            return (None, {}) if return_diagnostics else None

        # Kaynak trapez: sol-üst, sağ-üst, sağ-alt, sol-alt
        src = [(lt, y_far), (rt, y_far), (rb, y_near), (lb, y_near)]

        # Çıktı görüntüsünde (BEV) orantılı hedef noktaları (dst_pts) tanımla.
        # Yanlarda, üstte ve altta pay (margin) bırakarak sahnenin kırpılmasını ve 
        # yanlardaki araçların BEV dışında kalmasını engelliyoruz.
        margin_w = out_w * 0.35
        margin_h_top = out_h * 0.40
        margin_h_bottom = out_h * 0.10

        dst_pts = [
            [margin_w, margin_h_top],                # sol-üst
            [out_w - margin_w, margin_h_top],         # sağ-üst
            [out_w - margin_w, out_h - margin_h_bottom], # sağ-alt
            [margin_w, out_h - margin_h_bottom]       # sol-alt
        ]

        tf = PerspectiveTransformer.from_quad(
            src, out_w, out_h, real_w_m=real_w_m, real_h_m=real_h_m, dst_pts=dst_pts)

        if return_diagnostics:
            diag = {
                "segments": segments,
                "vertical_segments": vertical_segments,
                "vanishing_point": vp,
                "src_quad": src,
                "dst_pts": dst_pts,
                "method": "line"
            }
            return tf, diag
        return tf
    except Exception:
        return (None, {}) if return_diagnostics else None


def auto_calibrate_from_vehicles(detections, frame_shape, out_w: int = 600,
                                 out_h: int = 800, real_w_m: float | None = None,
                                 real_h_m: float | None = None,
                                 min_vehicles: int = 3,
                                 min_depth_frac: float = 0.18,
                                 return_diagnostics: bool = False):
    """Araçları referans alan otomatik IPM (çizgi yöntemine yedek).

    Park etmiş araçların sol/sağ sınırlarından vanishing point çıkararak
    tüm ekran genişliğine projekte edilmiş dengeli bir kuş bakışı oluşturur.
    """
    try:
        if not detections or len(detections) < min_vehicles:
            return (None, {}) if return_diagnostics else None
        h, w = frame_shape[:2]
        boxes = [d["bbox"] for d in detections]
        bottoms = [b[3] for b in boxes]
        y_near, y_far = max(bottoms), min(bottoms)
        depth = y_near - y_far
        if depth < min_depth_frac * h:
            return (None, {}) if return_diagnostics else None  # yeterli perspektif derinliği yok

        band = 0.25 * depth
        near_boxes = [b for b in boxes if b[3] >= y_near - band]
        far_boxes = [b for b in boxes if b[3] <= y_far + band]
        if not near_boxes or not far_boxes:
            return (None, {}) if return_diagnostics else None

        near_left = min(b[0] for b in near_boxes)
        near_right = max(b[2] for b in near_boxes)
        far_left = min(b[0] for b in far_boxes)
        far_right = max(b[2] for b in far_boxes)

        # Sol ve sağ çizgilerin kesişiminden vanishing point bul
        l_left = _line_from_seg((near_left, y_near, far_left, y_far))
        l_right = _line_from_seg((near_right, y_near, far_right, y_far))
        if l_left is None or l_right is None:
            return (None, {}) if return_diagnostics else None

        vp = estimate_vanishing_point([l_left, l_right])
        if vp is None or vp[1] > y_far:
            return (None, {}) if return_diagnostics else None

        vpx, vpy = vp
        # Enforce a symmetric, upright perspective projection (no diagonal roll/shear)
        vpx = w / 2.0

        # Tüm ekran genişliğini kapla
        lb = 0.0
        rb = float(w)

        dy_top = y_far - y_near
        dy_vp = vpy - y_near
        if abs(dy_vp) < 1e-3:
            return (None, {}) if return_diagnostics else None

        lt = lb + dy_top * (vpx - lb) / dy_vp
        rt = rb + dy_top * (vpx - rb) / dy_vp

        bottom_w = rb - lb
        top_w = rt - lt
        if bottom_w <= 10 or top_w <= 0 or top_w >= bottom_w:
            return (None, {}) if return_diagnostics else None

        src = [(lt, y_far), (rt, y_far), (rb, y_near), (lb, y_near)]

        # Çıktı görüntüsünde (BEV) orantılı hedef noktaları (dst_pts) tanımla.
        # Kırpılmayı önlemek için yan, üst ve alt paylar.
        margin_w = out_w * 0.35
        margin_h_top = out_h * 0.40
        margin_h_bottom = out_h * 0.10

        dst_pts = [
            [margin_w, margin_h_top],                # sol-üst
            [out_w - margin_w, margin_h_top],         # sağ-üst
            [out_w - margin_w, out_h - margin_h_bottom], # sağ-alt
            [margin_w, out_h - margin_h_bottom]       # sol-alt
        ]

        tf = PerspectiveTransformer.from_quad(
            src, out_w, out_h, real_w_m=real_w_m, real_h_m=real_h_m, dst_pts=dst_pts)

        if return_diagnostics:
            diag = {
                "segments": [(near_left, y_near, far_left, y_far), (near_right, y_near, far_right, y_far)],
                "vertical_segments": [(near_left, y_near, far_left, y_far), (near_right, y_near, far_right, y_far)],
                "vanishing_point": vp,
                "src_quad": src,
                "dst_pts": dst_pts,
                "method": "vehicle"
            }
            return tf, diag
        return tf
    except Exception:
        return (None, {}) if return_diagnostics else None
