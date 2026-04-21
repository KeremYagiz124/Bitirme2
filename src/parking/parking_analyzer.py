"""IoU-based parking analysis pipeline."""

from dataclasses import dataclass, field
import cv2
import numpy as np

from .zone_loader import ZoneLoader, ParkingZone

STATUS_AVAILABLE = "Park Edilebilir"
STATUS_OCCUPIED  = "Park Alanı Dolu"
STATUS_FORBIDDEN = "Park Edilemez"

_STATUS_COLORS = {
    STATUS_AVAILABLE: (0, 220, 80),
    STATUS_OCCUPIED:  (0, 60, 220),
    STATUS_FORBIDDEN: (0, 0, 200),
}

_VEHICLE_COLOR = (255, 200, 0)

_CLASS_NAME_TR = {
    "car": "Araba", "motorcycle": "Motosiklet",
    "bus": "Otobus", "truck": "Kamyon", "vehicle": "Arac",
}


@dataclass
class ZoneStatus:
    zone: ParkingZone
    status: str
    vehicle_bbox: list[float] | None = None


@dataclass
class AnalysisResult:
    zone_statuses: list[ZoneStatus] = field(default_factory=list)
    vehicle_labels: dict = field(default_factory=dict)  # detection index → status str

    @property
    def available(self) -> int:
        return sum(1 for z in self.zone_statuses if z.status == STATUS_AVAILABLE)

    @property
    def occupied(self) -> int:
        return sum(1 for z in self.zone_statuses if z.status == STATUS_OCCUPIED)

    @property
    def forbidden_vehicles(self) -> int:
        return sum(1 for s in self.vehicle_labels.values() if s == STATUS_FORBIDDEN)


class ParkingAnalyzer:
    def __init__(self, zone_loader: ZoneLoader, iou_threshold: float = 0.25):
        self.loader = zone_loader
        self.iou_threshold = iou_threshold

    def analyze(self, detections: list[dict]) -> AnalysisResult:
        result = AnalysisResult()

        # Forbidden zone check for each vehicle
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            for fz in self.loader.forbidden_zones:
                if fz.iou_with_bbox(bbox) >= self.iou_threshold:
                    result.vehicle_labels[i] = STATUS_FORBIDDEN
                    break

        # Parking zone occupancy
        for zone in self.loader.parking_zones:
            occupied_by = None
            for i, det in enumerate(detections):
                if zone.iou_with_bbox(det["bbox"]) >= self.iou_threshold:
                    occupied_by = det["bbox"]
                    break
            status = STATUS_OCCUPIED if occupied_by else STATUS_AVAILABLE
            result.zone_statuses.append(ZoneStatus(zone, status, occupied_by))

        # Forbidden zone statuses (for drawing)
        for zone in self.loader.forbidden_zones:
            result.zone_statuses.append(ZoneStatus(zone, STATUS_FORBIDDEN))

        return result

    def draw(self, frame: np.ndarray, result: AnalysisResult,
             detections: list[dict]) -> np.ndarray:
        out = frame.copy()

        # Draw zones
        for zs in result.zone_statuses:
            color = _STATUS_COLORS[zs.status]
            pts = zs.zone.polygon

            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            short = {STATUS_AVAILABLE: "BOS", STATUS_OCCUPIED: "DOLU",
                     STATUS_FORBIDDEN: "YASAK"}[zs.status]
            cv2.putText(out, short, (cx - 20, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Draw vehicles with status color
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            label_status = result.vehicle_labels.get(i)
            color = _STATUS_COLORS[label_status] if label_status else _VEHICLE_COLOR
            label = _CLASS_NAME_TR.get(det["class_name"], det["class_name"])
            if label_status == STATUS_FORBIDDEN:
                label += " YASAK"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return out
