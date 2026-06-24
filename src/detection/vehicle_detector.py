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
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
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
            half=(self.device == "cuda"),
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            bbox = box.xyxy[0].tolist()
            # class 7 (truck): küçük bbox'lar aslında SUV/araba - yeniden sınıflandır
            if class_id == 7 and not self._is_finetuned:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                # Görüntünün %3'ünden küçükse büyük olasılıkla SUV/araba
                if area < frame_area * 0.03:
                    class_id = 2  # car
            detections.append({
                "bbox": bbox,
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": self.class_map.get(class_id, "vehicle"),
            })

        # Sınıf-bağımsız NMS dedup: YOLO'nun dahili NMS'i sınıf-bazlıdır; aynı
        # araç birden çok sınıfta (ör. car + truck) çıkıp çift sayılabilir.
        # Yüksek güvenli kutuyu tut, %60+ örtüşen düşük güvenliyi ele.
        return self._dedupe(detections, iou_thresh=0.6)

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / union if union > 0 else 0.0

    @classmethod
    def _dedupe(cls, detections: list[dict], iou_thresh: float = 0.6) -> list[dict]:
        """Greedy sınıf-bağımsız NMS: yüksek güvenliyi tut, çok örtüşeni ele."""
        order = sorted(detections, key=lambda d: -d["confidence"])
        kept: list[dict] = []
        for d in order:
            if all(cls._iou(d["bbox"], k["bbox"]) <= iou_thresh for k in kept):
                kept.append(d)
        return kept

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
