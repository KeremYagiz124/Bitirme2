"""YOLOv8-based vehicle detection module."""

import cv2
import numpy as np
from pathlib import Path

# COCO pretrained model class IDs
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Fine-tuned model class IDs (0-indexed, synthetic data)
FINETUNED_VEHICLE_CLASSES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.5, iou: float = 0.45):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.conf = conf
        self.iou = iou

        # Fine-tuned modelde class ID'ler 0-indexed, COCO'da değil
        self._is_finetuned = "fine_tuned" in self.model_path or "finetuned" in self.model_path
        if self._is_finetuned:
            self.class_map = FINETUNED_VEHICLE_CLASSES
            self.filter_classes = list(FINETUNED_VEHICLE_CLASSES.keys())  # [0,1,2,3]
        else:
            self.class_map = COCO_VEHICLE_CLASSES
            self.filter_classes = list(COCO_VEHICLE_CLASSES.keys())  # [2,3,5,7]

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect vehicles in a frame.

        Returns list of dicts with keys:
            bbox: [x1, y1, x2, y2]
            confidence: float
            class_id: int
            class_name: str
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.filter_classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": self.class_map.get(class_id, "vehicle"),
            })

        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes on frame."""
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output
