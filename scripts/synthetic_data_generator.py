"""
Synthetic data generator for vehicle detection training.

Generates synthetic images with different vehicle types and automatic YOLO-format labels.
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse

# Vehicle types and their approximate sizes (width, height in pixels)
VEHICLE_TYPES = {
    "car": (80, 40),
    "motorcycle": (30, 60),
    "bus": (120, 50),
    "truck": (100, 45)
}

# Colors for different vehicle types
VEHICLE_COLORS = {
    "car": (0, 0, 255),        # Red
    "motorcycle": (255, 0, 0), # Blue
    "bus": (0, 255, 0),        # Green
    "truck": (255, 255, 0)     # Yellow
}

class SyntheticDataGenerator:
    def __init__(self, output_dir: str = "data/synthetic", image_size: tuple = (640, 480)):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def generate_background(self) -> np.ndarray:
        """Generate a simple background (road-like)."""
        # Create a gray background
        background = np.full((self.image_size[1], self.image_size[0], 3), 100, dtype=np.uint8)

        # Add some road texture
        for _ in range(50):
            x = random.randint(0, self.image_size[0])
            y = random.randint(0, self.image_size[1])
            cv2.circle(background, (x, y), random.randint(1, 3), (120, 120, 120), -1)

        return background

    def draw_vehicle(self, image: np.ndarray, vehicle_type: str, position: tuple, size: tuple) -> tuple:
        """Draw a simple vehicle shape and return bounding box."""
        x, y = position
        w, h = size

        # Draw a simple rectangle for the vehicle
        cv2.rectangle(image, (x, y), (x + w, y + h), VEHICLE_COLORS[vehicle_type], -1)

        # Add some details
        if vehicle_type == "car":
            # Windows
            cv2.rectangle(image, (x + 5, y + 5), (x + w - 5, y + h//2), (200, 200, 200), -1)
        elif vehicle_type == "motorcycle":
            # Handlebars
            cv2.line(image, (x + 5, y + 10), (x + 15, y + 5), (0, 0, 0), 2)
        elif vehicle_type == "bus":
            # Windows along the side
            for i in range(3):
                cv2.rectangle(image, (x + 5 + i*30, y + 5), (x + 25 + i*30, y + h - 5), (200, 200, 200), -1)
        elif vehicle_type == "truck":
            # Cargo area
            cv2.rectangle(image, (x + w//2, y + 5), (x + w - 5, y + h - 5), (150, 150, 150), -1)

        # Return bounding box in YOLO format (normalized)
        x_center = (x + w/2) / self.image_size[0]
        y_center = (y + h/2) / self.image_size[1]
        w_norm = w / self.image_size[0]
        h_norm = h / self.image_size[1]

        return x_center, y_center, w_norm, h_norm

    def generate_sample(self, sample_id: int, num_vehicles: int = 3) -> None:
        """Generate one synthetic image with vehicles."""
        # Generate background
        image = self.generate_background()

        labels = []

        # Class mapping for fine-tuned model (0-indexed, sequential)
        # NOT COCO IDs — fine_tune_yolo.py ve vehicle_detector.py bunu bilmeli
        class_mapping = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}
        # 0=car, 1=motorcycle, 2=bus, 3=truck

        for _ in range(num_vehicles):
            # Random vehicle type
            vehicle_type = random.choice(list(VEHICLE_TYPES.keys()))

            # Random position (ensure it fits in image)
            base_w, base_h = VEHICLE_TYPES[vehicle_type]
            scale = random.uniform(0.8, 1.5)
            w = int(base_w * scale)
            h = int(base_h * scale)

            x = random.randint(0, self.image_size[0] - w)
            y = random.randint(0, self.image_size[1] - h)

            # Draw vehicle and get YOLO bbox
            bbox = self.draw_vehicle(image, vehicle_type, (x, y), (w, h))
            class_id = class_mapping[vehicle_type]

            labels.append(f"{class_id} {' '.join(f'{coord:.6f}' for coord in bbox)}")

        # Save image
        image_path = self.images_dir / f"synthetic_{sample_id:06d}.jpg"
        cv2.imwrite(str(image_path), image)

        # Save labels
        label_path = self.labels_dir / f"synthetic_{sample_id:06d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

    def generate_dataset(self, num_samples: int = 1000) -> None:
        """Generate a dataset with specified number of samples and train/val split."""
        print(f"Generating {num_samples} synthetic samples...")

        # Train/val split: %80 train, %20 val
        train_count = int(num_samples * 0.8)
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_indices = set(indices[:train_count])

        train_images = self.images_dir / "train"
        val_images   = self.images_dir / "val"
        train_labels = self.labels_dir / "train"
        val_labels   = self.labels_dir / "val"
        for d in (train_images, val_images, train_labels, val_labels):
            d.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            split = "train" if i in train_indices else "val"
            num_vehicles = random.randint(1, 5)
            self._generate_sample_split(i, num_vehicles, split)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

        print("Dataset generation complete!")

        # Relative paths — YOLO resolves these relative to the yaml file location
        # This way the project works on any machine without modification
        data_yaml = (
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"nc: 4\n"
            f"names: ['car', 'motorcycle', 'bus', 'truck']\n"
            f"# class mapping: 0=car, 1=motorcycle, 2=bus, 3=truck\n"
        )

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(data_yaml)

        print(f"Created data.yaml at {yaml_path}")

    def _generate_sample_split(self, sample_id: int, num_vehicles: int, split: str) -> None:
        """Generate one synthetic image into the correct train/val subfolder."""
        image = self.generate_background()
        labels = []
        class_mapping = {"car": 0, "motorcycle": 1, "bus": 2, "truck": 3}

        for _ in range(num_vehicles):
            vehicle_type = random.choice(list(VEHICLE_TYPES.keys()))
            base_w, base_h = VEHICLE_TYPES[vehicle_type]
            scale = random.uniform(0.8, 1.5)
            w = int(base_w * scale)
            h = int(base_h * scale)
            x = random.randint(0, self.image_size[0] - w)
            y = random.randint(0, self.image_size[1] - h)
            bbox = self.draw_vehicle(image, vehicle_type, (x, y), (w, h))
            labels.append(f"{class_mapping[vehicle_type]} {' '.join(f'{c:.6f}' for c in bbox)}")

        fname = f"synthetic_{sample_id:06d}"
        cv2.imwrite(str(self.images_dir / split / f"{fname}.jpg"), image)
        with open(self.labels_dir / split / f"{fname}.txt", 'w') as f:
            f.write('\n'.join(labels))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle detection data")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")

    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        output_dir=args.output,
        image_size=(args.width, args.height)
    )

    generator.generate_dataset(args.samples)

if __name__ == "__main__":
    main()