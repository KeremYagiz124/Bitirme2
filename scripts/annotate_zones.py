"""
Parking zone annotation tool — CLI entry point.

Usage:
    python scripts/annotate_zones.py --image <path_to_image>
    python scripts/annotate_zones.py --image frame.jpg --output zones/lot1.json
    python scripts/annotate_zones.py --image frame.jpg --load zones/lot1.json  # resume editing

Controls:
    Left click        Add point
    ENTER or S        Save current polygon as PARKING zone
    F                 Save current polygon as FORBIDDEN zone
    Z                 Undo last point
    C                 Clear current polygon
    D                 Delete last saved zone
    Q / ESC           Save all zones and quit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parking import ZoneAnnotator


def main():
    parser = argparse.ArgumentParser(description="Annotate parking zones on an image.")
    parser.add_argument("--image", required=True, help="Path to the image/frame")
    parser.add_argument("--output", default=None, help="Output JSON path (default: same dir as image)")
    parser.add_argument("--load", default=None, help="Load existing zones JSON to continue editing")
    args = parser.parse_args()

    annotator = ZoneAnnotator(
        image_path=args.image,
        output_path=args.output,
        load_path=args.load,
    )
    out = annotator.run()
    print(f"Done. Zones saved to: {out}")


if __name__ == "__main__":
    main()
