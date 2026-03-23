"""
Parking zone annotation tool.

Click to add polygon points. Keyboard controls:
  ENTER / S  — save current polygon as a slot
  F          — save current polygon as forbidden zone
  Z          — undo last point
  C          — clear current polygon
  D          — delete last saved zone
  Q / ESC    — quit and save all zones to JSON

Usage:
    python scripts/annotate_zones.py --image <path> [--output <json_path>] [--load <json_path>]
"""

import cv2
import json
import numpy as np
from pathlib import Path
from copy import deepcopy

COLORS = {
    "parking": (0, 255, 0),      # green
    "forbidden": (0, 0, 255),    # red
    "current": (255, 165, 0),    # orange (in-progress)
}

ZONE_TYPES = {
    ord("s"): "parking",
    13: "parking",        # ENTER
    ord("f"): "forbidden",
}


class ZoneAnnotator:
    def __init__(self, image_path: str, output_path: str = None, load_path: str = None):
        self.image_path = image_path
        self.output_path = output_path or str(Path(image_path).with_suffix(".json"))
        self.base_image = cv2.imread(image_path)
        if self.base_image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.zones: list[dict] = []
        self.current_points: list[tuple] = []
        self.next_id = 1

        if load_path and Path(load_path).exists():
            self._load(load_path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.zones = data.get("zones", [])
        self.next_id = max((z["id"] for z in self.zones), default=0) + 1
        print(f"Loaded {len(self.zones)} zone(s) from {path}")

    def save(self) -> str:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image": str(self.image_path),
            "zones": self.zones,
        }
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.zones)} zone(s) to {self.output_path}")
        return self.output_path

    def _render(self) -> np.ndarray:
        frame = self.base_image.copy()

        # Draw saved zones
        for zone in self.zones:
            pts = np.array(zone["points"], dtype=np.int32)
            color = COLORS.get(zone["type"], (200, 200, 200))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(frame, [pts], (*color[::-1], 40))  # semi-transparent fill approx
            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = f"#{zone['id']} {zone['type'][0].upper()}"
            cv2.putText(frame, label, (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw current in-progress polygon
        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False,
                          color=COLORS["current"], thickness=2)
            for p in self.current_points:
                cv2.circle(frame, p, 4, COLORS["current"], -1)

        # HUD
        h = frame.shape[0]
        lines = [
            "ENTER/S: parking  F: forbidden",
            "Z: undo point  C: clear  D: del last zone",
            "Q/ESC: save & quit",
            f"Zones: {len(self.zones)}  Points: {len(self.current_points)}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - 80 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def _save_current(self, zone_type: str) -> None:
        if len(self.current_points) < 3:
            print("Need at least 3 points to save a zone.")
            return
        self.zones.append({
            "id": self.next_id,
            "type": zone_type,
            "points": list(self.current_points),
        })
        print(f"Saved zone #{self.next_id} ({zone_type}) with {len(self.current_points)} points.")
        self.next_id += 1
        self.current_points = []

    def run(self) -> str:
        win = "Zone Annotator"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_callback)

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in ZONE_TYPES:
                self._save_current(ZONE_TYPES[key])
            elif key == ord("z"):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord("c"):
                self.current_points = []
            elif key == ord("d"):
                if self.zones:
                    removed = self.zones.pop()
                    print(f"Deleted zone #{removed['id']}")
                    self.next_id = max((z["id"] for z in self.zones), default=0) + 1
            elif key in (ord("q"), 27):  # Q or ESC
                break

        cv2.destroyAllWindows()
        return self.save()
