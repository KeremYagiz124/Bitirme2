#!/usr/bin/env python3
"""Debug script to visualize zones and detections for issue analysis."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def draw_zones(frame, loader):
    """Draw parking zones on frame for visualization."""
    out = frame.copy()
    
    # Draw parking zones in green
    for zone in loader.parking_zones:
        pts = zone.polygon
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Label the zone
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(out, f"Zone {zone.id}", (cx - 20, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw forbidden zones in red
    for zone in loader.forbidden_zones:
        pts = zone.polygon
        cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(out, f"Forbidden {zone.id}", (cx - 20, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return out

def draw_detections(frame, detections):
    """Draw detections on frame for visualization."""
    out = frame.copy()
    
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(out, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return out

def test_zone_conditions(zone, bbox, threshold=0.25):
    """Test the specific conditions used in parking analysis."""
    coverage = zone.coverage_by_bbox(bbox)
    contains_center = zone.contains_center(bbox)
    iou = zone.iou_with_bbox(bbox)
    
    return {
        'coverage': coverage,
        'contains_center': contains_center,
        'iou': iou,
        'coverage_passes': coverage >= threshold,
        'overall_passes': coverage >= threshold or contains_center
    }

def main():
    # Load image
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Load zones
    loader = ZoneLoader(zones_path)
    
    # Initialize detector and analyzer
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Run detection
    detections = detector.detect(frame)
    
    print(f"\n=== DETECTION RESULTS ===")
    print(f"Detected {len(detections)} vehicles:")
    for i, det in enumerate(detections):
        print(f"  {i}: {det['class_name']} (conf={det['confidence']:.3f}) bbox={det['bbox']}")
    
    print(f"\n=== ZONE INFORMATION ===")
    print(f"Parking zones: {len(loader.parking_zones)}")
    print(f"Forbidden zones: {len(loader.forbidden_zones)}")
    
    # Analyze each detection against each zone
    print(f"\n=== ZONE-DETECTION ANALYSIS ===")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        print(f"\nDetection {i}: {det['class_name']} bbox={bbox}")
        
        for zone in loader.parking_zones:
            results = test_zone_conditions(zone, bbox, analyzer.iou_threshold)
            print(f"  Zone {zone.id}:")
            print(f"    Coverage: {results['coverage']:.3f} (>{analyzer.iou_threshold}? {results['coverage_passes']})")
            print(f"    Contains center: {results['contains_center']}")
            print(f"    IoU: {results['iou']:.3f}")
            print(f"    Overall pass: {results['overall_passes']}")
    
    # Run full analysis
    print(f"\n=== FULL ANALYSIS ===")
    result = analyzer.analyze(detections)
    
    print(f"Results:")
    print(f"  Available: {result.available}")
    print(f"  Occupied: {result.occupied}")
    print(f"  Forbidden vehicles: {result.forbidden_vehicles}")
    
    print("\nDetailed zone status:")
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            print(f"  Zone {zs.zone.id}: {zs.status}")
            if zs.vehicle_bbox:
                print(f"    Occupied by bbox: {zs.vehicle_bbox}")
    
    # Create visualization
    print(f"\n=== CREATING VISUALIZATION ===")
    viz_frame = frame.copy()
    
    # Draw zones
    viz_frame = draw_zones(viz_frame, loader)
    
    # Draw detections
    viz_frame = draw_detections(viz_frame, detections)
    
    # Draw analysis results (similar to analyzer.draw but simplified)
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            color = (0, 220, 80) if zs.status == "Park Edilebilir" else (0, 60, 220)
            pts = zs.zone.polygon
            overlay = viz_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, viz_frame, 0.75, 0, viz_frame)
            cv2.polylines(viz_frame, [pts], isClosed=True, color=color, thickness=2)
            
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            short = "BOS" if zs.status == "Park Edilebilir" else "DOLU"
            cv2.putText(viz_frame, short, (cx - 20, cy + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    
    # Save visualization
    output_dir = Path("outputs/debug_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "zone_detection_analysis.jpg"
    cv2.imwrite(str(output_path), viz_frame)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    main()