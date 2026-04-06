"""
Model evaluation script.

Runs the fine-tuned (or COCO) model against the validation set and prints
Precision, Recall, mAP@0.5 and mAP@0.5:0.95 metrics.

Usage:
    # Fine-tuned model
    python scripts/evaluate_model.py --model models/fine_tuned/yolov8_fine_tuned/weights/best.pt

    # COCO baseline
    python scripts/evaluate_model.py --model yolov8n.pt --data data/synthetic/data.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def evaluate(model_path: str, data_yaml: str, imgsz: int = 640, conf: float = 0.5, iou: float = 0.45):
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"\nModel : {model_path}")
    print(f"Data  : {data_yaml}")
    print(f"conf={conf}  iou={iou}  imgsz={imgsz}\n")

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=True,
    )

    # Per-class names
    names = results.names  # {0: 'car', 1: 'motorcycle', ...}

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    mp = results.box.mp        # mean Precision
    mr = results.box.mr        # mean Recall
    map50 = results.box.map50  # mAP@0.5
    map   = results.box.map    # mAP@0.5:0.95

    print(f"  Precision (mean) : {mp:.4f}  ({mp*100:.2f}%)")
    print(f"  Recall    (mean) : {mr:.4f}  ({mr*100:.2f}%)")
    print(f"  mAP@0.5         : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95    : {map:.4f}  ({map*100:.2f}%)")

    # Per-class breakdown
    print("\nPer-class mAP@0.5:")
    ap50_per_class = results.box.ap50  # numpy array
    for i, ap in enumerate(ap50_per_class):
        cls_name = names.get(i, str(i))
        print(f"  {cls_name:<12}: {ap:.4f}  ({ap*100:.2f}%)")

    print("=" * 50)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model on the synthetic val set.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to model weights (default: yolov8n.pt COCO baseline)",
    )
    parser.add_argument(
        "--data",
        default="data/synthetic/data.yaml",
        help="Path to data.yaml (default: data/synthetic/data.yaml)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    model_path = args.model
    data_yaml = args.data

    if not Path(model_path).exists() and model_path != "yolov8n.pt":
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    if not Path(data_yaml).exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        sys.exit(1)

    evaluate(model_path, data_yaml, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()
