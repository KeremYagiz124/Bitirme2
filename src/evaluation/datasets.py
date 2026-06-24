"""Gerçek otopark veri seti ingest — PKLot / CNRPark-EXT.

PKLot, her görüntü için XML açıklama dosyası içerir: her park yeri bir
<space occupied="0|1"> ve dört köşeli <contour> ile tanımlanır. Bu modül XML'i
projenin ground truth formatına (görüntü başına boş slot bbox listesi) çevirir.

Veri seti makinede yoksa fonksiyonlar boş/kısmi sonuç döndürür (çökme yok),
böylece pipeline indirme olmadan da içe aktarılabilir.

PKLot indirme (kullanıcı tarafından, ~1.5GB):
    https://web.inf.ufpr.br/vri/databases/parking-lot-database/
    İndir → çöz → --root ile bu modüle ver.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_pklot_xml(xml_path_or_string, is_string: bool = False):
    """Tek bir PKLot XML'ini park yeri listesine çevir.

    Döner: [{"bbox": (x1, y1, x2, y2), "occupied": bool}, ...]
    """
    if is_string:
        root = ET.fromstring(xml_path_or_string)
    else:
        root = ET.parse(str(xml_path_or_string)).getroot()

    spaces = []
    for space in root.findall(".//space"):
        occ_attr = space.get("occupied")
        occupied = (occ_attr == "1") if occ_attr is not None else False
        xs, ys = [], []
        for pt in space.findall(".//contour/point"):
            xs.append(float(pt.get("x")))
            ys.append(float(pt.get("y")))
        if not xs or not ys:
            # contour yoksa rotatedRect merkez+boyutundan kaba bbox üret
            rr = space.find(".//rotatedRect")
            if rr is None:
                continue
            c = rr.find("center"); s = rr.find("size")
            if c is None or s is None:
                continue
            cx, cy = float(c.get("x")), float(c.get("y"))
            w, h = float(s.get("w")), float(s.get("h"))
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        spaces.append({"bbox": (x1, y1, x2, y2), "occupied": occupied})
    return spaces


def ingest_pklot(root_dir: str, out_json: str = "data/pklot_ground_truth.json",
                 limit: int | None = None) -> dict:
    """PKLot kök klasörünü tarayıp ground truth JSON üret.

    root_dir altında her .xml ile aynı isimli .jpg eşleştirilir. Çıktı:
        { "goreli/yol/uuid.jpg": {"empty": n, "occupied": m,
                                   "slots": [{"bbox": [...], "occupied": bool}]} }
    Veri seti yoksa boş sözlük döner (çökme yok).
    """
    root = Path(root_dir)
    gt = {}
    if not root.exists():
        return gt

    xml_files = sorted(root.rglob("*.xml"))
    if limit:
        xml_files = xml_files[:limit]

    for xml_path in xml_files:
        jpg_path = xml_path.with_suffix(".jpg")
        if not jpg_path.exists():
            continue
        try:
            spaces = parse_pklot_xml(xml_path)
        except ET.ParseError:
            continue
        empty = sum(1 for s in spaces if not s["occupied"])
        occupied = sum(1 for s in spaces if s["occupied"])
        rel = str(jpg_path.relative_to(root)).replace("\\", "/")
        gt[rel] = {
            "empty": empty,
            "occupied": occupied,
            "slots": [{"bbox": list(s["bbox"]), "occupied": s["occupied"]}
                      for s in spaces],
        }

    if gt:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2)
    return gt
