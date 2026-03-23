"""Load and query annotated parking zones from JSON."""

import json
import numpy as np
import cv2
from pathlib import Path


class ParkingZone:
    def __init__(self, zone_id: int, zone_type: str, points: list[list[int]]):
        self.id = zone_id
        self.type = zone_type  # "parking" | "forbidden"
        self.polygon = np.array(points, dtype=np.int32)

    def contains_center(self, bbox: list[float]) -> bool:
        """Check if bbox center falls inside this polygon."""
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        return cv2.pointPolygonTest(self.polygon, (cx, cy), False) >= 0

    def iou_with_bbox(self, bbox: list[float]) -> float:
        """Approximate IoU between polygon and bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        h = max(y2, self.polygon[:, 1].max()) + 1
        w = max(x2, self.polygon[:, 0].max()) + 1

        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 1, -1)

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [self.polygon], 1)

        intersection = np.logical_and(bbox_mask, poly_mask).sum()
        union = np.logical_or(bbox_mask, poly_mask).sum()
        return float(intersection / union) if union > 0 else 0.0


class ZoneLoader:
    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.image_path = data.get("image", "")
        self.zones: list[ParkingZone] = [
            ParkingZone(z["id"], z["type"], z["points"])
            for z in data.get("zones", [])
        ]

    @property
    def parking_zones(self) -> list[ParkingZone]:
        return [z for z in self.zones if z.type == "parking"]

    @property
    def forbidden_zones(self) -> list[ParkingZone]:
        return [z for z in self.zones if z.type == "forbidden"]

    def find_zone(self, bbox: list[float], iou_threshold: float = 0.3) -> ParkingZone | None:
        """Return the zone with highest IoU above threshold, or None."""
        best, best_iou = None, iou_threshold
        for zone in self.zones:
            iou = zone.iou_with_bbox(bbox)
            if iou > best_iou:
                best, best_iou = zone, iou
        return best
