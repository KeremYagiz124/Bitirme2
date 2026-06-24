"""Çok kriterli akıllı slot seçim motoru (A4).

Boş slotları yalnızca "boş" diye göstermek yerine, sürücü için en uygun slotu
çok kriterli bir skorla önerir:
  - Manevra zorluğu (slot genişliği / araç genişliği oranı)
  - Yakınlık (ego/kamera referansına piksel mesafe)
  - Genişlik payı (rahat sığma marjı)
  - Çıkışa yakınlık (haritada konum)

Saf Python/numpy (dış bağımlılık yok), bağımsız test edilebilir. Karar Destek
Sistemleri (Decision Support) açısından akademik katkı sağlar.
"""

from __future__ import annotations

DEFAULT_WEIGHTS = {
    "difficulty": 0.40,
    "proximity": 0.25,
    "width_margin": 0.20,
    "exit_proximity": 0.15,
}


def compute_difficulty(width_m, ref_width_m: float = 1.8) -> int:
    """Manevra zorluğu skoru (0-100; YÜKSEK = kolay/geniş, DÜŞÜK = zor/dar).

    Slot genişliği aracın enine göre ne kadar fazlaysa o kadar kolay.
    Boyut bilinmiyorsa nötr 50 döner.
    """
    if not width_m or width_m <= 0 or ref_width_m <= 0:
        return 50
    margin = (width_m - ref_width_m) / ref_width_m
    diff = 50 + margin * 120.0
    return int(max(0, min(100, round(diff))))


def compute_slot_score(difficulty: float, distance_px: float, width_m,
                       ref_width_m: float, slot_cx: float, map_width: float,
                       weights: dict | None = None) -> int:
    """Çok kriterli ağırlıklı slot skoru (0-100)."""
    w = weights or DEFAULT_WEIGHTS
    norm_diff = max(0.0, min(1.0, difficulty / 100.0))
    norm_dist = max(0.0, 1.0 - distance_px / 1000.0)
    if ref_width_m > 0 and width_m:
        width_margin = max(0.0, min(1.0, (width_m - ref_width_m) / ref_width_m))
    else:
        width_margin = 0.5
    if map_width > 0:
        exit_prox = max(0.0, min(1.0, 1.0 - abs(slot_cx - map_width) / map_width))
    else:
        exit_prox = 0.5
    score = (w["difficulty"] * norm_diff + w["proximity"] * norm_dist
             + w["width_margin"] * width_margin + w["exit_proximity"] * exit_prox)
    return int(round(max(0.0, min(1.0, score)) * 100))


def slot_reason_text(difficulty: float, width_margin: float,
                     distance_px: float, exit_prox: float) -> str:
    """Öneri gerekçesi metni (insan-okunur)."""
    parts = []
    if difficulty >= 75:
        parts.append("Kolay manevra")
    elif difficulty >= 45:
        parts.append("Orta zorluk")
    else:
        parts.append("Dar alan")
    if width_margin > 0.3:
        parts.append("Genis slot")
    if distance_px < 300:
        parts.append("Yakin mesafe")
    if exit_prox > 0.7:
        parts.append("Cikisa yakin")
    return " · ".join(parts)


def recommend_best_slot(slots, ref_width_m: float, map_width: float,
                        origin, weights: dict | None = None):
    """Aday boş slotlardan en yüksek skorluyu öner.

    slots: [{"cx": float, "cy": float, "width_m": float|None}, ...]
    origin: (x, y) ego/kamera referansı (mesafe için)
    Döner: {"index", "score", "difficulty", "reason"} veya boşsa None.
    """
    if not slots:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    best = None
    for i, s in enumerate(slots):
        width_m = s.get("width_m")
        cx, cy = float(s["cx"]), float(s["cy"])
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        diff = compute_difficulty(width_m, ref_width_m)
        score = compute_slot_score(diff, dist, width_m, ref_width_m,
                                   cx, map_width, weights)
        if ref_width_m > 0 and width_m:
            wmargin = max(0.0, min(1.0, (width_m - ref_width_m) / ref_width_m))
        else:
            wmargin = 0.5
        exit_prox = (max(0.0, min(1.0, 1.0 - abs(cx - map_width) / map_width))
                     if map_width > 0 else 0.5)
        reason = slot_reason_text(diff, wmargin, dist, exit_prox)
        if best is None or score > best["score"]:
            best = {"index": i, "score": score, "difficulty": diff,
                    "reason": reason}
    return best
