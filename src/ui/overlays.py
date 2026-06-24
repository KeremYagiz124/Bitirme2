"""Demo görselleştirme yardımcıları — saf OpenCV (Qt bağımlılığı yok).

İçerik:
  - nearest_empty: bir referans noktaya en yakın boş slotu bul.
  - draw_guidance: referanstan hedefe yönlendirme oku + etiket çiz.
  - draw_pseudo_3d: slot poligonunu sahte-3B kutu olarak çiz (AR etkisi).

Tüm fonksiyonlar saf numpy/cv2; bağımsız test edilebilir.
"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def poly_centroid(poly) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).mean(axis=0)


def nearest_empty(empty_polys, origin):
    """origin'e en yakın boş slotu döndür.

    Döner: (index, centroid(2,), distance_px) veya boşsa None.
    """
    if not empty_polys:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    best = None
    for i, p in enumerate(empty_polys):
        c = poly_centroid(p)
        d = float(np.hypot(c[0] - ox, c[1] - oy))
        if best is None or d < best[2]:
            best = (i, c, d)
    return best


def draw_guidance(out, origin, target, label="EN YAKIN"):
    """Referanstan hedefe sarı yönlendirme oku + etiket."""
    o = (int(origin[0]), int(origin[1]))
    t = (int(target[0]), int(target[1]))
    cv2.arrowedLine(out, o, t, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.06)
    if label:
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
        tx, ty = t[0] - tw // 2, max(th + 4, t[1] - 12)
        cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def render_minimap(empty_polys, occupied_polys, fit_flags=None, width: int = 240,
                   height: int = 74, margin: int = 6, title: str = "PARK HARITASI"):
    """Otoparkın soyut 2B krokisi: yeşil boş / kırmızı dolu hücreler, numaralı.

    Slotlar merkez-x'e göre soldan sağa sıralanır. Ham görüntüden bağımsız,
    temiz bir şematik panel döndürür (BGR).
    """
    panel = np.full((height, width, 3), 28, np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv2.putText(panel, title, (margin, 15), _FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    items = []
    for i, p in enumerate(empty_polys):
        fit = fit_flags[i] if (fit_flags is not None and i < len(fit_flags)) else True
        items.append((float(np.asarray(p)[:, 0].mean()), False, fit))
    for p in occupied_polys:
        items.append((float(np.asarray(p)[:, 0].mean()), True, False))
    items.sort(key=lambda t: t[0])

    n = len(items)
    if n == 0:
        cv2.putText(panel, "slot yok", (margin, height // 2 + 8),
                    _FONT, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return panel

    top, bot = 24, height - margin
    cw = (width - 2 * margin) / n
    for i, (_, occ, fit) in enumerate(items):
        x1 = int(margin + i * cw)
        x2 = int(margin + (i + 1) * cw) - 2
        if occ:
            color = (0, 60, 200)       # Kırmızı (dolu)
        elif not fit:
            color = (0, 165, 255)      # Turuncu (sığmaz)
        else:
            color = (0, 200, 80)       # Yeşil (boş ve sığar)
        cv2.rectangle(panel, (x1, top), (x2, bot), color, -1)
        cv2.putText(panel, str(i + 1), (x1 + 3, bot - 4),
                    _FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def paste_minimap(out, minimap, pad: int = 8, top: int = 36):
    """Mini-haritayı çıktının sağ üst köşesine yerleştir (yerinde)."""
    mh, mw = minimap.shape[:2]
    H, W = out.shape[:2]
    x0 = W - mw - pad
    y0 = top
    if x0 < 0 or y0 + mh > H:
        return out
    out[y0:y0 + mh, x0:x0 + mw] = minimap
    return out


def draw_pseudo_3d(out, poly, color, lift: int = 18):
    """Slot poligonunu sahte-3B kutu gibi çiz: taban + yukarı kaldırılmış üst
    yüz + dikey kenarlar. Gerçek metrik 3B değil, görsel AR etkisi."""
    bottom = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
    if lift <= 0:
        cv2.polylines(out, [bottom.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        return out
    top = bottom.copy()
    top[:, 1] -= lift
    cv2.polylines(out, [bottom.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    cv2.polylines(out, [top.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    for b, t in zip(bottom, top):
        cv2.line(out, (int(b[0]), int(b[1])), (int(t[0]), int(t[1])),
                 color, 2, cv2.LINE_AA)
    return out


def _draw_rotated_car(panel, cx, cy, w, h, yaw_deg, color, scale):
    yaw_rad = np.deg2rad(yaw_deg)
    cos_a, sin_a = np.cos(yaw_rad), np.sin(yaw_rad)
    
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])
    
    rot_corners = []
    for x, y in corners:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        rot_corners.append([rx, ry])
    pts = np.array(rot_corners, dtype=np.int32)
    
    cv2.fillPoly(panel, [pts], (50, 42, 35))
    cv2.polylines(panel, [pts], True, color, 2, cv2.LINE_AA)
    
    # Windscreen (Front)
    fs_pts = np.array([
        [dx * 0.1, -dy * 0.85],
        [dx * 0.4, -dy * 0.85],
        [dx * 0.4, dy * 0.85],
        [dx * 0.1, dy * 0.85]
    ])
    rot_fs = []
    for x, y in fs_pts:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        rot_fs.append([rx, ry])
    cv2.fillPoly(panel, [np.array(rot_fs, dtype=np.int32)], (120, 100, 80))
    
    # Windscreen (Rear)
    rs_pts = np.array([
        [-dx * 0.4, -dy * 0.85],
        [-dx * 0.1, -dy * 0.85],
        [-dx * 0.1, dy * 0.85],
        [-dx * 0.4, dy * 0.85]
    ])
    rot_rs = []
    for x, y in rs_pts:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        rot_rs.append([rx, ry])
    cv2.fillPoly(panel, [np.array(rot_rs, dtype=np.int32)], (120, 100, 80))
    
    # Side mirrors
    m1_pts = np.array([
        [dx * 0.1, -dy - int(3 * scale)],
        [dx * 0.25, -dy - int(3 * scale)],
        [dx * 0.25, -dy],
        [dx * 0.1, -dy]
    ])
    rot_m1 = []
    for x, y in m1_pts:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        rot_m1.append([rx, ry])
    cv2.fillPoly(panel, [np.array(rot_m1, dtype=np.int32)], color)
    
    m2_pts = np.array([
        [dx * 0.1, dy],
        [dx * 0.25, dy],
        [dx * 0.25, dy + int(3 * scale)],
        [dx * 0.1, dy + int(3 * scale)]
    ])
    rot_m2 = []
    for x, y in m2_pts:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        rot_m2.append([rx, ry])
    cv2.fillPoly(panel, [np.array(rot_m2, dtype=np.int32)], color)

def _draw_headlights(panel, cx, cy, yaw_deg, w, h, scale, occupied_polys):
    import numpy as np
    import cv2
    
    height, width = panel.shape[:2]
    
    dx = w / 2.0
    dy = h / 2.0
    
    yaw_rad = np.deg2rad(yaw_deg)
    cos_a, sin_a = np.cos(yaw_rad), np.sin(yaw_rad)
    
    # Headlight positions at the front bumper
    hl_left_x = cx + dx * cos_a - (-dy * 0.75) * sin_a
    hl_left_y = cy + dx * sin_a + (-dy * 0.75) * cos_a
    
    hl_right_x = cx + dx * cos_a - (dy * 0.75) * sin_a
    hl_right_y = cy + dx * sin_a + (dy * 0.75) * cos_a
    
    headlights = [(hl_left_x, hl_left_y), (hl_right_x, hl_right_y)]
    
    light_mask = np.zeros((height, width), dtype=np.uint8)
    
    R = int(220 * scale)
    spread_deg = 20.0
    spread_rad = np.deg2rad(spread_deg)
    
    for h_x, h_y in headlights:
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 1. Draw light cone polygon
        cone_pts = [[h_x, h_y]]
        num_arc = 12
        for step in range(num_arc + 1):
            angle = (yaw_rad - spread_rad) + (2.0 * spread_rad * step / num_arc)
            ax = h_x + R * np.cos(angle)
            ay = h_y + R * np.sin(angle)
            cone_pts.append([ax, ay])
            
        cone_poly = np.array(cone_pts, dtype=np.int32)
        cv2.fillPoly(temp_mask, [cone_poly], 255)
        
        # 2. Project obstacle shadow volumes
        for poly in occupied_polys:
            shadow_pts = []
            for pt in poly:
                p_x, p_y = pt[0], pt[1]
                dir_x = p_x - h_x
                dir_y = p_y - h_y
                dist = np.hypot(dir_x, dir_y)
                if dist == 0:
                    dist = 0.01
                proj_x = p_x + (dir_x / dist) * 1000.0
                proj_y = p_y + (dir_y / dist) * 1000.0
                shadow_pts.append([proj_x, proj_y])
                
            shadow_poly_pts = []
            for pt in poly:
                shadow_poly_pts.append(pt)
            for pt in reversed(shadow_pts):
                shadow_poly_pts.append(pt)
                
            shadow_poly = np.array(shadow_poly_pts, dtype=np.int32)
            cv2.fillPoly(temp_mask, [shadow_poly], 0)
            
        cv2.bitwise_or(light_mask, temp_mask, dst=light_mask)
        
    if R > 4:
        ksize = int(25 * scale) | 1
        light_mask_blurred = cv2.GaussianBlur(light_mask, (ksize, ksize), 0)
    else:
        light_mask_blurred = light_mask

    # Premium warm golden glow
    glow_color = np.array([160, 235, 255], dtype=np.uint8)
    
    mask_normalized = light_mask_blurred.astype(np.float32) / 255.0
    mask_normalized *= 0.50 # max opacity for realistic light blending
    
    for c in range(3):
        panel[:, :, c] = np.clip(
            panel[:, :, c] * (1.0 - mask_normalized) + glow_color[c] * mask_normalized,
            0, 255
        ).astype(np.uint8)

def _draw_steering_wheel(panel, cx, cy, radius, angle_deg, color):
    cv2.circle(panel, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), int(radius * 0.2), color, -1)
    for offset in [0, 120, 240]:
        rad = np.deg2rad(angle_deg + offset - 90)
        sx = int(cx + radius * np.cos(rad))
        sy = int(cy + radius * np.sin(rad))
        cv2.line(panel, (cx, cy), (sx, sy), color, 2, cv2.LINE_AA)

def render_full_schematic_map(empty_polys, occupied_polys, sizes_m=None, fit_flags=None,
                              detections: list | None = None,
                              perp_mode: bool = False,
                              width: int = 900, height: int = 660,
                              difficulties: list | None = None,
                              sim_active: bool = False,
                              sim_car_x: float = 0.0,
                              sim_car_y: float = 0.0,
                              sim_car_yaw: float = 0.0,
                              sim_target_idx: int = -1,
                              sim_instruction: str = "",
                              sim_steering_angle: float = 0.0,
                              sim_step_name: str = "",
                              sim_path: list = [],
                              night_vision: bool = False) -> np.ndarray:
    """Otoparkın tümünü kuş bakışı modelleyen, son derece estetik ve temiz 2D şematik harita.

    Tüm perspektif bozulmalarından arındırılmış, Tesla stili premium dijital ikiz.
    """
    # Modern koyu tema arka planı (Slate-900: RGB 15, 23, 42 -> BGR 42, 23, 15)
    bg_color = (42, 23, 15)
    panel = np.full((height, width, 3), bg_color, np.uint8)

    # Harita çerçevesi
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (50, 41, 30), 2)

    # Üst Bilgi Kartı / Dashboard Alanı
    cv2.rectangle(panel, (20, 20), (width - 20, 100), (30, 25, 18), -1)
    cv2.rectangle(panel, (20, 20), (width - 20, 100), (80, 70, 50), 1)

    # Başlık
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "AKILLI OTOPARK DIJITAL IKIZI", (40, 55),
                _FONT, 0.7, (255, 180, 100), 2, cv2.LINE_AA)
    cv2.putText(panel, "2D KUS BAKISI SEMATIK HARITA", (40, 80),
                _FONT, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    # Slotları listele ve yatay konumlarına (x) göre sırala
    slots_list = []
    # (cx, is_occupied, size_m, fits, label/difficulty)
    
    # 1) Boş slotlar
    for i, p in enumerate(empty_polys):
        cx = float(np.asarray(p)[:, 0].mean())
        sz = sizes_m[i] if (sizes_m is not None and i < len(sizes_m)) else None
        fit = fit_flags[i] if (fit_flags is not None and i < len(fit_flags)) else True
        diff = difficulties[i] if (difficulties is not None and i < len(difficulties)) else None
        slots_list.append((cx, False, sz, fit, diff))

    # 2) Dolu slotlar
    occupied_xs = []
    for p in occupied_polys:
        cx = float(np.asarray(p)[:, 0].mean())
        slots_list.append((cx, True, None, False, "DOLU"))
        occupied_xs.append(cx)

    # 3) YOLO ile tespit edilen araçları şematik haritaya ekle (Deduplicate & Overlap)
    VEHICLE_CLASSES = {2, 3, 5, 7}
    if detections:
        for det in detections:
            cls_id = det.get("class_id")
            if cls_id not in VEHICLE_CLASSES:
                continue
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            
            # Dolu slotlar ile mükerrer kontrolü (40 piksel tolerans)
            is_duplicate = False
            for ox in occupied_xs:
                if abs(cx - ox) < 40:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Boş slot üzerine denk geliyorsa orayı dolu yap
            overlapping_empty_idx = -1
            for idx, item in enumerate(slots_list):
                if not item[1]:  # Boş slot
                    if abs(cx - item[0]) < 35:
                        overlapping_empty_idx = idx
                        break
            if overlapping_empty_idx != -1:
                slots_list[overlapping_empty_idx] = (cx, True, None, False, "DOLU")
            else:
                slots_list.append((cx, True, None, False, "ARAC"))

    # X koordinatına göre soldan sağa sırala
    slots_list.sort(key=lambda t: t[0])
    n = len(slots_list)

    # İstatistikler
    empty_cnt = sum(1 for s in slots_list if not s[1])
    occ_cnt = sum(1 for s in slots_list if s[1])
    total_cnt = n

    # İstatistik yazıları (Dashboard sağ kısım)
    stat_x = width - 360
    cv2.putText(panel, f"Toplam Slot: {total_cnt}", (stat_x, 48), _FONT, 0.5, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Bos Slot: {empty_cnt}", (stat_x, 78), _FONT, 0.5, (100, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Dolu Slot: {occ_cnt}", (stat_x + 160, 48), _FONT, 0.5, (100, 120, 255), 1, cv2.LINE_AA)

    pct = int(occ_cnt / total_cnt * 100) if total_cnt > 0 else 0
    cv2.putText(panel, f"Doluluk: %{pct}", (stat_x + 160, 78), _FONT, 0.5, (255, 220, 100), 1, cv2.LINE_AA)

    if n == 0:
        # Slot tespit edilmediğinde
        cv2.putText(panel, "Aktif Park Alani Tespit Edilemedi", (width // 2 - 200, height // 2),
                    _FONT, 0.7, (100, 116, 139), 2, cv2.LINE_AA)
        cv2.putText(panel, "Lutfen kamera/video beslemesini veya ROI secimini kontrol edin.", (width // 2 - 280, height // 2 + 40),
                    _FONT, 0.45, (71, 85, 105), 1, cv2.LINE_AA)
        return panel

    # Otopark Yol Çizimi (Ortadan geçen gri koridor)
    road_y = height // 2 - 10
    road_h = 160
    cv2.rectangle(panel, (20, road_y), (width - 20, road_y + road_h), (50, 45, 38), -1)
    # Kesik kesik yol şerit çizgileri
    for rx in range(40, width - 40, 40):
        cv2.line(panel, (rx, road_y + road_h // 2), (rx + 20, road_y + road_h // 2), (0, 220, 255), 2)

    # Slotların Çizimi (Yolun üst tarafına sıralı şekilde)
    if perp_mode:
        slot_w_base = 90
        slot_h_base = 140
        banner_h_base = 24
        gap_base = 12
    else:
        slot_w_base = 150
        slot_h_base = 80
        banner_h_base = 20
        gap_base = 12

    # Toplam yer kaplamasını bul ve sığmıyorsa ölçekle (yatayda sığması için)
    max_draw_w = width - 80
    total_w_base = n * slot_w_base + (n - 1) * gap_base
    if total_w_base > max_draw_w:
        scale = max_draw_w / total_w_base
    else:
        scale = 1.0

    # Ölçeklenmiş boyutlar
    slot_w = int(slot_w_base * scale)
    slot_h = int(slot_h_base * scale)
    banner_h = int(banner_h_base * scale)
    gap = int(gap_base * scale)

    # Toplam yer kaplamasını yeniden hesapla ve ortala
    total_w = n * slot_w + (n - 1) * gap
    start_x = (width - total_w) // 2

    slot_y = road_y - slot_h - 4 # Yolun hemen üstü

    # Yazı boyutları ölçeğe göre dinamik
    font_scale_header = max(0.24, 0.4 * scale)
    font_scale_content = max(0.24, 0.4 * scale)
    font_scale_dim = max(0.22, 0.35 * scale)

    occupied_polys_to_cast = []
    for i, (_, is_occ, sz, fit, extra) in enumerate(slots_list):
        x1 = start_x + i * (slot_w + gap)
        x2 = x1 + slot_w
        y1 = slot_y
        y2 = slot_y + slot_h

        # Slot No kutusu
        cv2.rectangle(panel, (x1, y1), (x2, y1 + banner_h), (30, 25, 18), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y1 + banner_h), (80, 70, 50), 1)
        
        # Slot no yazısı ortalama
        if perp_mode:
            cv2.putText(panel, f"SLOT {i+1}", (x1 + int(14 * scale), y1 + int(17 * scale)), _FONT, font_scale_header, (200, 180, 140), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, f"SLOT {i+1}", (x1 + int(48 * scale), y1 + int(14 * scale)), _FONT, font_scale_header, (200, 180, 140), 1, cv2.LINE_AA)

        if is_occ:
            occupied_polys_to_cast.append(np.array([[x1, y1 + banner_h], [x2, y1 + banner_h], [x2, y2], [x1, y2]], dtype=np.int32))
            label = extra
            # DOLU SLOT: Kırmızı/Mavi tonlar
            cv2.rectangle(panel, (x1, y1 + banner_h), (x2, y2), (25, 20, 45), -1)
            cv2.rectangle(panel, (x1, y1 + banner_h), (x2, y2), (100, 80, 220), 2)

            # 2D Top-down Car Silhouette
            cx = (x1 + x2) // 2
            cy = (y1 + banner_h + y2) // 2

            car_col = (220, 110, 40)
            if perp_mode:
                # DİK PARK: Araçlar dikey
                cw = int(48 * scale)
                ch = int(86 * scale)
                cv2.rectangle(panel, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), car_col, -1)
                cv2.rectangle(panel, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (255, 180, 100), 1, cv2.LINE_AA)

                # Windscreens
                cv2.rectangle(panel, (cx - cw // 2 + int(4 * scale), cy - ch // 2 + int(18 * scale)), (cx + cw // 2 - int(4 * scale), cy - ch // 2 + int(28 * scale)), (42, 23, 15), -1)
                cv2.rectangle(panel, (cx - cw // 2 + int(4 * scale), cy + ch // 2 - int(24 * scale)), (cx + cw // 2 - int(4 * scale), cy + ch // 2 - int(16 * scale)), (42, 23, 15), -1)

                # Side mirrors
                cv2.rectangle(panel, (cx - cw // 2 - int(3 * scale), cy - ch // 2 + int(22 * scale)), (cx - cw // 2, cy - ch // 2 + int(28 * scale)), car_col, -1)
                cv2.rectangle(panel, (cx + cw // 2, cy - ch // 2 + int(22 * scale)), (cx + cw // 2 + int(3 * scale), cy - ch // 2 + int(28 * scale)), car_col, -1)

                # Label
                cv2.putText(panel, label if label else "DOLU", (cx - int(18 * scale), cy + int(30 * scale)), _FONT, font_scale_content, (120, 120, 255), 1, cv2.LINE_AA)
            else:
                # PARALEL PARK: Araçlar yatay
                cw = int(96 * scale)
                ch = int(48 * scale)
                cv2.rectangle(panel, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), car_col, -1)
                cv2.rectangle(panel, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (255, 180, 100), 1, cv2.LINE_AA)

                # Windscreens
                cv2.rectangle(panel, (cx + cw // 2 - int(28 * scale), cy - ch // 2 + int(4 * scale)), (cx + cw // 2 - int(18 * scale), cy + ch // 2 - int(4 * scale)), (42, 23, 15), -1)
                cv2.rectangle(panel, (cx - cw // 2 + int(16 * scale), cy - ch // 2 + int(4 * scale)), (cx - cw // 2 + int(24 * scale), cy + ch // 2 - int(4 * scale)), (42, 23, 15), -1)

                # Side mirrors
                cv2.rectangle(panel, (cx + cw // 2 - int(28 * scale), cy - ch // 2 - int(3 * scale)), (cx + cw // 2 - int(22 * scale), cy - ch // 2), car_col, -1)
                cv2.rectangle(panel, (cx + cw // 2 - int(28 * scale), cy + ch // 2), (cx + cw // 2 - int(22 * scale), cy + ch // 2 + int(3 * scale)), car_col, -1)

                # Label
                cv2.putText(panel, label if label else "DOLU", (cx - int(18 * scale), cy + int(5 * scale)), _FONT, font_scale_content, (120, 120, 255), 1, cv2.LINE_AA)
        else:
            diff = extra
            # BOŞ SLOT: Zorluk derecesine göre Premium HSL-benzeri renkler
            if diff is not None:
                if diff >= 75:
                    bg_col = (20, 40, 20)      # Canlı yeşil dolgu
                    border_col = (80, 220, 0)   # Canlı yeşil kenar
                    lbl = f"KOLAY {diff}%"
                    lbl_color = (120, 255, 120)
                elif diff >= 45:
                    bg_col = (15, 30, 45)      # Canlı turuncu dolgu
                    border_col = (0, 165, 255) # Canlı turuncu kenar
                    lbl = f"ORTA {diff}%"
                    lbl_color = (0, 200, 255)
                else:
                    bg_col = (15, 15, 40)      # Canlı kırmızı dolgu
                    border_col = (0, 0, 240)   # Canlı kırmızı kenar
                    lbl = f"ZOR {diff}%"
                    lbl_color = (100, 100, 255)
            else:
                if fit:
                    bg_col = (20, 40, 20)
                    border_col = (50, 220, 100)
                    lbl = "SIGAR"
                    lbl_color = (100, 255, 120)
                else:
                    bg_col = (20, 30, 45)
                    border_col = (0, 140, 255)
                    lbl = "SIGMAZ"
                    lbl_color = (0, 180, 255)

            if sim_active and i == sim_target_idx:
                border_col = (255, 255, 0)
                bg_col = (50, 45, 10)
                lbl = "HEDEF"
                lbl_color = (255, 255, 0)

            cv2.rectangle(panel, (x1, y1 + banner_h), (x2, y2), bg_col, -1)
            cv2.rectangle(panel, (x1, y1 + banner_h), (x2, y2), border_col, 2)

            cx = (x1 + x2) // 2
            cy = (y1 + banner_h + y2) // 2

            if perp_mode:
                # P işareti (Büyük)
                cv2.putText(panel, "P", (cx - int(14 * scale), cy + int(5 * scale)), _FONT, max(0.6, 1.1 * scale), border_col, max(1, int(3 * scale)), cv2.LINE_AA)
                # Ebat / Sığma Durumu Etiketi
                cv2.putText(panel, lbl, (cx - int(24 * scale), cy + int(28 * scale)), _FONT, font_scale_content, lbl_color, 1, cv2.LINE_AA)
                if sz is not None:
                    cv2.putText(panel, f"{sz[0]:.1f}m", (cx - int(20 * scale), cy + int(42 * scale)), _FONT, font_scale_dim, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                # P işareti (Biraz daha küçük ve yukarda)
                cv2.putText(panel, "P", (cx - int(10 * scale), cy - int(5 * scale)), _FONT, max(0.5, 0.85 * scale), border_col, max(1, int(2 * scale)), cv2.LINE_AA)
                # Ebat / Sığma Durumu Etiketi
                cv2.putText(panel, lbl, (cx - int(22 * scale), cy + int(12 * scale)), _FONT, font_scale_content, lbl_color, 1, cv2.LINE_AA)
                if sz is not None:
                    cv2.putText(panel, f"{sz[0]:.1f}m", (cx - int(18 * scale), cy + int(22 * scale)), _FONT, font_scale_dim, (200, 200, 200), 1, cv2.LINE_AA)

    # 4) Kendi aracımızın konumunu hesapla (Kamera yatay ortasında kabul edilir)
    camera_w = 1280
    all_x = []
    for p in empty_polys:
        all_x.extend(np.asarray(p)[:, 0])
    for p in occupied_polys:
        all_x.extend(np.asarray(p)[:, 0])
    if detections:
        for det in detections:
            bbox = det["bbox"]
            all_x.extend([bbox[0], bbox[2]])
    if all_x:
        camera_w = max(camera_w, max(all_x))
    
    ego_x_camera = camera_w / 2.0
    
    if n <= 1:
        ego_x_schematic = width // 2
    else:
        # Interpolasyon ile kendi aracımızın şematik haritadaki x pozisyonunu bul
        camera_x_coords = [item[0] for item in slots_list]
        schematic_x_coords = [start_x + idx * (slot_w + gap) + slot_w // 2 for idx in range(n)]
        
        # Ekstrapolasyon sınırları
        if ego_x_camera <= camera_x_coords[0]:
            ego_x_schematic = schematic_x_coords[0] - (camera_x_coords[0] - ego_x_camera) * (slot_w + gap) / max(10.0, camera_x_coords[0])
        elif ego_x_camera >= camera_x_coords[-1]:
            ego_x_schematic = schematic_x_coords[-1] + (ego_x_camera - camera_x_coords[-1]) * (slot_w + gap) / max(10.0, camera_w - camera_x_coords[-1])
        else:
            ego_x_schematic = np.interp(ego_x_camera, camera_x_coords, schematic_x_coords)

    ego_x_schematic = int(ego_x_schematic)
    ego_y_schematic = int(road_y + road_h // 2 + 25)
    ego_car_w = int(90 * scale)
    ego_car_h = int(45 * scale)
    ego_color = (255, 200, 0) # Cyan

    # Dynamic Headlight & Shadow Simulation (Headlight Shader)
    if sim_active or night_vision:
        car_cx = sim_car_x if sim_active else ego_x_schematic
        car_cy = sim_car_y if sim_active else ego_y_schematic
        car_yaw = sim_car_yaw if sim_active else 0.0
        _draw_headlights(panel, car_cx, car_cy, car_yaw, ego_car_w, ego_car_h, scale, occupied_polys_to_cast)

    if sim_active:
        # 1) Yörünge yolunu çiz (kesik çizgiler/noktalar)
        if len(sim_path) > 1:
            for pt in sim_path:
                cv2.circle(panel, (int(pt[0]), int(pt[1])), max(2, int(4 * scale)), (0, 255, 255), -1, cv2.LINE_AA)
        
        # 2) Simüle edilen aracı çiz
        _draw_rotated_car(panel, int(sim_car_x), int(sim_car_y), ego_car_w, ego_car_h, sim_car_yaw, (0, 255, 255), scale)
        
        # 3) Autopilot Telemetry Dashboard (Koyu Slate glassmorphism kartı)
        dash_y = height - 135
        dash_h = 110
        cv2.rectangle(panel, (30, dash_y), (width - 30, dash_y + dash_h), (30, 24, 18), -1)
        cv2.rectangle(panel, (30, dash_y), (width - 30, dash_y + dash_h), (0, 220, 255), 1, cv2.LINE_AA)
        
        # Dashboard başlıkları ve verileri
        cv2.putText(panel, "ADAS OTONOM VALE PARKI (AVP) SIMULASYONU", (50, dash_y + 25), _FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"DURUM: {sim_step_name.upper()}", (50, dash_y + 50), _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"TALIMAT: {sim_instruction}", (50, dash_y + 80), _FONT, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
        
        # Telemetry değerleri (Direksiyon açısı ve hız)
        val_x = width - 380
        cv2.putText(panel, f"Hiz: 1.8 km/s", (val_x, dash_y + 40), _FONT, 0.4, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Direksiyon: {sim_steering_angle:+.1f} deg", (val_x, dash_y + 65), _FONT, 0.45, (255, 200, 100), 1, cv2.LINE_AA)
        
        # 4) Dönen direksiyon simgesi
        wheel_cx = width - 100
        wheel_cy = dash_y + dash_h // 2
        wheel_r = int(35 * scale)
        _draw_steering_wheel(panel, wheel_cx, wheel_cy, wheel_r, sim_steering_angle, (0, 255, 255))
    else:
        # Normal durumda kendi aracımızı çiz
        cv2.rectangle(panel, (ego_x_schematic - ego_car_w // 2, ego_y_schematic - ego_car_h // 2),
                      (ego_x_schematic + ego_car_w // 2, ego_y_schematic + ego_car_h // 2), (50, 42, 35), -1)
        cv2.rectangle(panel, (ego_x_schematic - ego_car_w // 2, ego_y_schematic - ego_car_h // 2),
                      (ego_x_schematic + ego_car_w // 2, ego_y_schematic + ego_car_h // 2), ego_color, 2, cv2.LINE_AA)
                      
        # Ön/Arka camlar
        cv2.rectangle(panel, (ego_x_schematic + ego_car_w // 2 - int(26 * scale), ego_y_schematic - ego_car_h // 2 + int(4 * scale)),
                      (ego_x_schematic + ego_car_w // 2 - int(16 * scale), ego_y_schematic + ego_car_h // 2 - int(4 * scale)), (120, 100, 80), -1)
        cv2.rectangle(panel, (ego_x_schematic - ego_car_w // 2 + int(14 * scale), ego_y_schematic - ego_car_h // 2 + int(4 * scale)),
                      (ego_x_schematic - ego_car_w // 2 + int(24 * scale), ego_y_schematic + ego_car_h // 2 - int(4 * scale)), (120, 100, 80), -1)
                      
        # Yan aynalar
        cv2.rectangle(panel, (ego_x_schematic + ego_car_w // 2 - int(26 * scale), ego_y_schematic - ego_car_h // 2 - int(3 * scale)),
                      (ego_x_schematic + ego_car_w // 2 - int(20 * scale), ego_y_schematic - ego_car_h // 2), ego_color, -1)
        cv2.rectangle(panel, (ego_x_schematic + ego_car_w // 2 - int(26 * scale), ego_y_schematic + ego_car_h // 2),
                      (ego_x_schematic + ego_car_w // 2 - int(20 * scale), ego_y_schematic + ego_car_h // 2 + int(3 * scale)), ego_color, -1)

        cv2.putText(panel, "ARACIMIZ", (ego_x_schematic - int(28 * scale), ego_y_schematic + int(4 * scale)),
                    _FONT, font_scale_content, (255, 255, 255), 1, cv2.LINE_AA)

        # Rehberlik / En yakın boş yönlendirmesi
        first_empty_idx = -1
        for i, (_, is_occ, _, fit, _) in enumerate(slots_list):
            if not is_occ and fit:
                first_empty_idx = i
                break

        if first_empty_idx != -1:
            tx = start_x + first_empty_idx * (slot_w + gap) + slot_w // 2
            ty_slot_bottom = slot_y + slot_h
            
            # Ok başlangıç ve bitiş noktası (Kendi aracımızdan boş otopark slotuna)
            start_pt = (ego_x_schematic, ego_y_schematic - ego_car_h // 2 - 5)
            end_pt = (tx, ty_slot_bottom + 5)
            
            cv2.arrowedLine(panel, start_pt, end_pt, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.2)
            
            # Metni ok üzerine hizala
            mx = (start_pt[0] + end_pt[0]) // 2
            my = (start_pt[1] + end_pt[1]) // 2
            cv2.putText(panel, "EN YAKIN BOS PARK ALANI", (mx - 80, my - 10), _FONT, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return panel


def draw_bev_overlays(bev_out, ipm, empty_polys, occupied_polys, detections,
                      sizes_m=None, fit_flags=None, difficulties=None):
    """Kuş bakışı bükülmüş görüntü üzerine temiz, dik ve okunabilir kutular, slotlar ve yazılar çizer."""
    COLOR_EASY = (80, 220, 0)
    COLOR_MEDIUM = (0, 165, 255)
    COLOR_HARD = (0, 0, 240)
    COLOR_OCC   = (0, 60, 200)

    # 1) Boş slotlar
    for i, poly in enumerate(empty_polys):
        try:
            poly_bev = ipm.transform_points(poly).astype(np.int32)
            score = difficulties[i] if (difficulties is not None and i < len(difficulties)) else None
            sz = sizes_m[i] if (sizes_m is not None and i < len(sizes_m)) else None
            w_m = sz[0] if (sz is not None and sz[0] > 0) else None

            if score is not None:
                if score >= 75:
                    col = COLOR_EASY
                    label = f"KOLAY {score}%"
                elif score >= 45:
                    col = COLOR_MEDIUM
                    label = f"ORTA {score}%"
                else:
                    col = COLOR_HARD
                    fit = fit_flags[i] if (fit_flags is not None and i < len(fit_flags)) else True
                    if not fit:
                        label = f"ZOR {score}% (DAR)"
                    else:
                        label = f"ZOR {score}%"
            else:
                fit = fit_flags[i] if (fit_flags is not None and i < len(fit_flags)) else True
                col = COLOR_EASY if fit else COLOR_HARD
                label = "BOS"

            if w_m is not None:
                label += f" ({w_m:.1f}m)"

            overlay = bev_out.copy()
            cv2.fillPoly(overlay, [poly_bev], col)
            cv2.polylines(overlay, [poly_bev], True, col, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.28, bev_out, 0.72, 0, bev_out)

            c = poly_bev.mean(axis=0).astype(int)
            cv2.putText(bev_out, label, (int(c[0]) - 32, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        except Exception:
            pass

    # 2) Dolu slotlar
    for i, poly in enumerate(occupied_polys):
        try:
            poly_bev = ipm.transform_points(poly).astype(np.int32)
            overlay = bev_out.copy()
            cv2.fillPoly(overlay, [poly_bev], COLOR_OCC)
            cv2.polylines(overlay, [poly_bev], True, COLOR_OCC, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.28, bev_out, 0.72, 0, bev_out)

            c = poly_bev.mean(axis=0).astype(int)
            cv2.putText(bev_out, "DOLU", (int(c[0]) - 15, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_OCC, 1, cv2.LINE_AA)
        except Exception:
            pass

    # 3) Tespit edilen araç kutuları
    VEHICLE_CLASSES = {2, 3, 5, 7}
    VEHICLE_COLORS_CV = {2: (0, 255, 0), 3: (0, 165, 255), 5: (0, 0, 255), 7: (255, 0, 255)}
    for det in detections:
        cls_id = det.get("class_id")
        if cls_id not in VEHICLE_CLASSES:
            continue
        try:
            bbox = det["bbox"]
            # En kararlı zemin temas noktası: taban çizgisi ortası
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3]

            pts_bev = ipm.transform_points([(cx, cy)])
            if not np.all(np.isfinite(pts_bev)):
                continue
            bx, by = pts_bev[0]

            # Kuş bakışı fiziksel araba boyutları (genişlik = 1.8m, boy = 4.5m)
            if ipm.m_per_px and ipm.m_per_px > 0:
                width_px = 1.8 / ipm.m_per_px
                length_px = 4.5 / ipm.m_per_px
            else:
                w_orig = bbox[2] - bbox[0]
                width_px = w_orig * 0.8
                length_px = width_px * 2.2

            vx1 = int(bx - width_px / 2.0)
            vx2 = int(bx + width_px / 2.0)
            vy2 = int(by)
            vy1 = int(by - length_px)

            color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
            cv2.rectangle(bev_out, (vx1, vy1), (vx2, vy2), color, 2)
            cv2.putText(bev_out, "ARAC", (vx1 + 5, vy2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        except Exception:
            pass
