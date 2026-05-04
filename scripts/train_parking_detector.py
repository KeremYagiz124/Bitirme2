"""
Park alanı tespit modeli eğitimi (PKLot tabanlı).

Sınıflar: 0 = boş slot, 1 = dolu slot

Kullanım:
    python scripts/train_parking_detector.py --data-dir data/pklot
    python scripts/train_parking_detector.py --data-dir data/pklot --epochs 50 --validate
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   required=True,                    help="PKLot dataset dizini")
    ap.add_argument("--output-dir", default="models/parking_detector", help="Model çıktı dizini")
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--batch",      type=int, default=16)
    ap.add_argument("--imgsz",      type=int, default=640)
    ap.add_argument("--validate",   action="store_true")
    ap.add_argument("--device",     default="",  help="cpu veya 0 (GPU). Boş bırakılırsa otomatik seçilir.")
    args = ap.parse_args()

    data_dir  = Path(args.data_dir)
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"HATA: {yaml_path} bulunamadı.")
        print("Önce dataset indirin: python scripts/download_pklot.py --api-key YOUR_KEY")
        sys.exit(1)

    import torch
    from ultralytics import YOLO

    if args.device:
        device = args.device
    else:
        device = "0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Eğitim başlıyor — device: {device}, epochs: {args.epochs}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(out_dir),
        name="pklot_detector",
        device=device,
        workers=0,
        patience=15,
        save=True,
        verbose=True,
        # Sınıf dengesizliğini düzelt (BOS:24, DOLU:129 → ağırlık ~5x)
        cls=0.5,
        # Augmentation - az veriyi telafi et
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
    )

    # En iyi model yolunu bul
    candidates = [out_dir / "pklot_detector" / "weights" / "best.pt"]
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        for p in sorted(runs_dir.rglob("best.pt")):
            candidates.append(p)

    best_path = next((c for c in candidates if c.exists()), None)
    if best_path is None:
        print("HATA: Eğitilmiş model bulunamadı.")
        sys.exit(1)

    print(f"\nModel kaydedildi: {best_path}")

    if args.validate:
        results = model.val(data=str(yaml_path), verbose=True)
        print(f"mAP@0.5:     {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision:   {results.box.mp:.4f}")
        print(f"Recall:      {results.box.mr:.4f}")

    print(f"\nGUI'de kullanmak için 'Oto Model Yükle' butonuna tıklayın → {best_path}")


if __name__ == "__main__":
    main()
