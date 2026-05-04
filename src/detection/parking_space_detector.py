"""Park alanı otomatik tespit modülü (PKLot tabanlı model).

Model.names içindeki sınıf isimleri otomatik olarak BOS/DOLU olarak eşleştirilir;
sabit ID varsayımı yapılmaz.
"""

import cv2
import numpy as np

_EMPTY_KEYWORDS    = {"empty", "bos", "free", "available", "space-empty"}
_OCCUPIED_KEYWORDS = {"occupied", "dolu", "taken", "full", "space-occupied"}

COLOR_EMPTY    = (0, 220, 80)   # yeşil
COLOR_OCCUPIED = (60, 60, 220)  # mavi


class ParkingSpaceDetector:
    def __init__(self, model_path: str, conf: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.model.to("cpu")
        self.conf = conf
        self._build_class_map()

    def _build_class_map(self):
        """model.names içinden boş/dolu sınıflarını otomatik eşleştir."""
        names = self.model.names  # {0: "space-empty", 1: "space-occupied"} gibi
        self._status_map = {}     # cls_id → "BOS" | "DOLU" | "?"
        self._color_map  = {}     # cls_id → BGR tuple
        for cls_id, name in names.items():
            lower = name.lower().replace("-", "").replace("_", "")
            if any(k in lower for k in _EMPTY_KEYWORDS):
                self._status_map[cls_id] = "BOS"
                self._color_map[cls_id]  = COLOR_EMPTY
            elif any(k in lower for k in _OCCUPIED_KEYWORDS):
                self._status_map[cls_id] = "DOLU"
                self._color_map[cls_id]  = COLOR_OCCUPIED
            else:
                self._status_map[cls_id] = name.upper()
                self._color_map[cls_id]  = (128, 128, 128)

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "bbox":       box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id":   cls_id,
                "status":     self._status_map.get(cls_id, "?"),
            })
        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame.copy()
        overlay = out.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          self._color_map.get(det["class_id"], (128, 128, 128)), -1)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = self._color_map.get(det["class_id"], (128, 128, 128))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det['status']}  {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out

    @property
    def class_names(self) -> dict:
        """Yüklenen modelin sınıf eşleşmelerini döndür."""
        return {cls_id: status for cls_id, status in self._status_map.items()}
