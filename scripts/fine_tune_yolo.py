"""
YOLOv8 fine-tuning script for parking scenario using synthetic data.

This script fine-tunes a pre-trained YOLOv8 model on synthetic vehicle data
to adapt it for parking lot detection scenarios.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import argparse

def create_data_config(synthetic_data_dir: str, output_path: str = "data.yaml") -> str:
    """Create data configuration file for YOLO training."""
    data_dir = Path(synthetic_data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Synthetic data directory not found: {synthetic_data_dir}")

    # Check if data.yaml already exists from synthetic generator
    existing_yaml = data_dir / "data.yaml"
    if existing_yaml.exists():
        print(f"Using existing data.yaml from {existing_yaml}")
        return str(existing_yaml)

    # Create data.yaml
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images or labels directory not found in {synthetic_data_dir}")

    # Use relative paths so the yaml works on any machine
    data_config = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 4,
        'names': ['car', 'motorcycle', 'bus', 'truck']
    }

    # Write next to the data directory so relative paths resolve correctly
    output_path = str(data_dir / "data.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Created data.yaml at {output_path}")
    return output_path

def fine_tune_yolo(data_config: str, output_dir: str = "models/fine_tuned",
                   epochs: int = 50, batch_size: int = 8, img_size: int = 640) -> str:
    """Fine-tune YOLOv8 model on synthetic data."""
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO('yolov8n.pt')

    print("Starting YOLOv8 fine-tuning...")
    print(f"Device: {device}")
    print(f"Data config: {data_config}")
    print(f"Output directory: {output_path}")
    print(f"Training parameters: epochs={epochs}, batch={batch_size}, imgsz={img_size}")

    # Fine-tune the model
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(output_path),
        name="yolov8_fine_tuned",
        save=True,
        save_period=10,
        cache=False,
        device=device,
        workers=0,
        patience=10,
        verbose=True
    )

    # Get the best model path — Ultralytics saves under runs/detect/<project>/<name>/
    # Try expected path first, then search runs/ directory
    candidates = [
        output_path / "yolov8_fine_tuned" / "weights" / "best.pt",
        output_path / "yolov8_fine_tuned2" / "weights" / "best.pt",
    ]

    # Also search inside runs/detect/
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        for p in sorted(runs_dir.rglob("best.pt")):
            candidates.append(p)

    for candidate in candidates:
        if candidate.exists():
            print(f"Fine-tuning complete! Best model saved at: {candidate}")
            return str(candidate)

    raise FileNotFoundError("No trained model found! Check runs/detect/ directory.")

def validate_model(model_path: str, data_config: str) -> None:
    """Validate the fine-tuned model."""
    print(f"Validating model: {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_config, verbose=True)

    print("Validation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on synthetic vehicle data")
    parser.add_argument("--data-dir", required=True, help="Directory containing synthetic data")
    parser.add_argument("--output-dir", default="models/fine_tuned", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")

    args = parser.parse_args()

    # Create data configuration
    data_config = create_data_config(args.data_dir)

    # Fine-tune model
    model_path = fine_tune_yolo(
        data_config=data_config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    # Validate if requested
    if args.validate:
        validate_model(model_path, data_config)

    print("\nFine-tuning process completed successfully!")
    print(f"Trained model: {model_path}")
    print("You can now use this model in your vehicle detector by updating the model_path parameter.")

if __name__ == "__main__":
    main()