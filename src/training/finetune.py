"""Park-özel YOLO fine-tuning pipeline'ı (PKLot/CNRPark).

"Hazır YOLO" yerine park tespitine ince ayarlanmış model akademik ağırlık katar.
Pipeline üç aşama:

  1. pklot_to_yolo(): PKLot ground truth → YOLO formatı (images/ + labels/*.txt)
  2. make_data_yaml(): Ultralytics data.yaml üret
  3. train_finetune(): yolov8n.pt'den başlayıp park sınıflarına ince ayar
  4. compare_before_after(): öncesi (hazır) vs sonrası (fine-tuned) metrik

NOT: Aşama 3 (eğitim) GPU + indirilmiş veri seti gerektirir. Veri dönüşümü
ve config üretimi (aşama 1-2) bağımsız çalışır ve test edilebilir.

Sınıflar: 0 = space-empty (boş), 1 = space-occupied (dolu)
"""

from __future__ import annotations

from pathlib import Path

import cv2

CLASS_EMPTY = 0
CLASS_OCCUPIED = 1
CLASS_NAMES = {CLASS_EMPTY: "space-empty", CLASS_OCCUPIED: "space-occupied"}


def _bbox_to_yolo(bbox, img_w: int, img_h: int):
    """(x1,y1,x2,y2) piksel bbox → normalize (cx, cy, w, h) YOLO formatı."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def write_yolo_labels(slots, img_w, img_h, out_txt):
    """Bir görüntünün slot listesini YOLO .txt etiket dosyasına yaz.

    slots: [{"bbox": [x1,y1,x2,y2], "occupied": bool}, ...]
    """
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in slots:
        cls = CLASS_OCCUPIED if s["occupied"] else CLASS_EMPTY
        cx, cy, w, h = _bbox_to_yolo(s["bbox"], img_w, img_h)
        # YOLO normalize değerleri [0,1] aralığında olmalı
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        w, h = min(max(w, 0), 1), min(max(h, 0), 1)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def pklot_to_yolo(ground_truth: dict, images_root: str, out_dir: str,
                  val_split: float = 0.2, seed: int = 42):
    """PKLot ground truth sözlüğünü YOLO eğitim klasör yapısına çevir.

    Çıktı:
        out_dir/images/train, out_dir/images/val
        out_dir/labels/train, out_dir/labels/val

    Döner: {"train": n_train, "val": n_val}
    """
    import random
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    items = list(ground_truth.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(len(items) * val_split)
    splits = {"val": items[:n_val], "train": items[n_val:]}

    counts = {}
    for split, entries in splits.items():
        n = 0
        for rel_path, spec in entries:
            img_path = images_root / rel_path
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            slots = spec.get("slots", [])
            if not slots:
                continue
            stem = Path(rel_path).stem
            img_out = out_dir / "images" / split / f"{stem}.jpg"
            lbl_out = out_dir / "labels" / split / f"{stem}.txt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_out), frame)
            write_yolo_labels(slots, w, h, lbl_out)
            n += 1
        counts[split] = n
    return counts


def make_data_yaml(dataset_dir: str, out_path: str = None) -> str:
    """Ultralytics data.yaml içeriği üret (ve istenirse dosyaya yaz)."""
    dataset_dir = Path(dataset_dir).resolve()
    content = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
        f"  {CLASS_EMPTY}: {CLASS_NAMES[CLASS_EMPTY]}\n"
        f"  {CLASS_OCCUPIED}: {CLASS_NAMES[CLASS_OCCUPIED]}\n"
    )
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    return content


def train_finetune(data_yaml: str, base_model: str = "yolov8n.pt",
                   epochs: int = 50, imgsz: int = 640, device: str = "cpu",
                   project: str = "outputs/training", name: str = "pklot_finetune"):
    """yolov8n.pt'den başlayıp park sınıflarına ince ayar (Ultralytics).

    GPU önerilir (device='0'). CPU'da çok yavaştır; demo için epochs düşük tut.
    Döner: eğitilen modelin .pt yolu.
    """
    from ultralytics import YOLO
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, device=device,
        project=project, name=name, exist_ok=True, verbose=False,
    )
    best = Path(project) / name / "weights" / "best.pt"
    return str(best) if best.exists() else str(results.save_dir)


def compare_before_after(data_yaml: str, finetuned_model: str,
                         base_model: str = "yolov8n.pt", device: str = "cpu"):
    """Hazır model vs fine-tuned model val metriklerini karşılaştır.

    Döner: {"before": {...}, "after": {...}} (mAP50, mAP50-95, precision, recall)
    """
    from ultralytics import YOLO

    def _val(model_path):
        m = YOLO(model_path)
        r = m.val(data=data_yaml, device=device, verbose=False)
        box = r.box
        return {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "Precision": float(box.mp),
            "Recall": float(box.mr),
        }

    return {"before": _val(base_model), "after": _val(finetuned_model)}
