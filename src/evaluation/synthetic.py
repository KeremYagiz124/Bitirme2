"""Sentetik park sahnesi üreteci — ölçeklenebilir nicel değerlendirme için.

Gerçek etiketli veri seti (PKLot vb.) indirilemediğinde dahi, park tespit
algoritmasının çekirdek geometrik katkısı 100+ sahnede değerlendirilebilsin
diye prosedürel sahneler üretir.

Her sahne için:
  - frame: çizilmiş park sırası (araçlar + yol zemini)
  - detections: araç bbox listesi (YOLO çıktısı taklidi — algoritma doğrudan
    beslenir, böylece tespit kalitesinden bağımsız olarak SLOT bulma yeteneği
    ölçülür)
  - gt_empty: ground truth boş slot bbox listesi

Bu, "doğru araç tespiti verildiğinde algoritma boş yerleri ne kadar iyi
buluyor" sorusunu YOLO'dan ayrıştırarak ölçer.
"""

from __future__ import annotations

import numpy as np
import cv2


def make_scene(rng, img_w=1280, img_h=420, n_slots=8,
               occupancy_prob=0.6, car_class="car"):
    """Tek bir park sırası sahnesi üret.

    Sıra uçları (ilk ve son slot) daima dolu tutulur; böylece kenar-uzantı
    slotları ground truth'u kirletmez ve değerlendirme adil kalır.

    Döner: (frame, detections, gt_empty)
    """
    frame = np.full((img_h, img_w, 3), 90, dtype=np.uint8)  # asfalt grisi

    margin = 60
    usable = img_w - 2 * margin
    slot_w = usable / n_slots
    # Gerçek otoparkta araç slot adımının ~%90'ını doldurur; küçük servis
    # boşluğu kalır. Bu oran, gap/araç-genişliği bölmesinin slot sayısını
    # doğru vermesini sağlar (aşırı bölme olmaz).
    car_w = slot_w * 0.90
    # Paralel park (yan görünüm): araç yatay görünür (en > boy). Detektörün
    # paralel-mod aday filtresi bw/bh >= 0.8 ister; bu yüzden landscape çiziyoruz.
    car_h = car_w * 0.55
    base_bottom = img_h * 0.72
    y1 = int(base_bottom - car_h)
    y2 = int(base_bottom)

    occupied = []
    for i in range(n_slots):
        if i == 0 or i == n_slots - 1:
            occ = True  # uçlar daima dolu
        else:
            occ = rng.random() < occupancy_prob
        occupied.append(occ)

    detections = []
    gt_empty = []
    for i in range(n_slots):
        slot_x1 = margin + i * slot_w
        cx = slot_x1 + slot_w / 2
        cw = car_w
        bx1 = int(cx - cw / 2)
        bx2 = int(cx + cw / 2)
        if occupied[i]:
            # araç gövdesi (renkli dikdörtgen) — gerçekçi görünüm
            color = tuple(int(c) for c in rng.integers(40, 210, size=3))
            cv2.rectangle(frame, (bx1, y1), (bx2, y2), color, -1)
            cv2.rectangle(frame, (bx1, y1), (bx2, y2), (20, 20, 20), 2)
            detections.append({
                "bbox": [float(bx1), float(y1), float(bx2), float(y2)],
                "class_id": 2, "class_name": car_class, "confidence": 0.9,
            })
        else:
            # boş slot: ground truth bbox (araç boyutunda, slot merkezli)
            gt_empty.append((bx1, y1, bx2, y2))

    return frame, detections, gt_empty


def make_dataset(n_scenes=120, seed=42, **scene_kwargs):
    """N sentetik sahne üret. Döner: liste[(frame, detections, gt_empty)]."""
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n_scenes):
        # sahne başına slot sayısı ve doluluk çeşitliliği
        n_slots = int(rng.integers(6, 11))
        occ = float(rng.uniform(0.45, 0.8))
        frame, dets, gt = make_scene(rng, n_slots=n_slots,
                                     occupancy_prob=occ, **scene_kwargs)
        scenes.append((frame, dets, gt))
    return scenes
