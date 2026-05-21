#!/usr/bin/env python3
"""Detailed debug to understand false positive detection."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def debug_overlap_calculations():
    """Debug the exact overlap calculations that might be causing false positives."""
    
    # Load the test image and zones
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Get detections
    detections = detector.detect(frame)
    
    print("=== DETECTIONS ===")
    for i, det in enumerate(detections):
        print(f"{i}: {det['class_name']} conf={det['confidence']:.3f}")
        print(f"    bbox: {det['bbox']}")
    
    print("\n=== ZONE DEFINITIONS ===")
    for zone in loader.parking_zones:
        print(f"Zone {zone.id} ({zone.type}):")
        print(f"    Points: {zone.polygon.tolist()}")
        print(f"    Bounding box: [{zone.polygon[:, 0].min()}, {zone.polygon[:, 1].min()}, "
              f"{zone.polygon[:, 0].max()}, {zone.polygon[:, 1].max()}]")
        print(f"    Area: {cv2.contourArea(zone.polygon)} pixels")
    
    print("\n=== OVERLAP CALCULATIONS ===")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        print(f"\nDetection {i}: {det['class_name']} bbox={bbox}")
        
        for zone in loader.parking_zones:
            # Test the exact functions used in analyzer
            coverage = zone.coverage_by_bbox(bbox)
            contains_center = zone.contains_center(bbox)
            iou = zone.iou_with_bbox(bbox)
            
            print(f"  Zone {zone.id}:")
            print(f"    Coverage (bbox covers zone): {coverage:.4f}")
            print(f"    Contains center: {contains_center}")
            print(f"    IoU: {iou:.4f}")
            print(f"    Coverage >= threshold (0.25): {coverage >= 0.25}")
            print(f"    Overall pass (coverage OR center): {coverage >= 0.25 or contains_center}")
    
    # Now let's manually trace through the analyze method
    print("\n=== MANUAL ANALYZE TRACE ===")
    result = type('AnalysisResult', (), {
        'zone_statuses': [],
        'vehicle_labels': {}
    })()
    
    now = 0  # Simplified
    
    def _is_overlapping(zone, bbox):
        """Copy of the function from parking_analyzer.py"""
        if zone.coverage_by_bbox(bbox) >= analyzer.iou_threshold:
            return True
        if zone.contains_center(bbox):
            return True
        return False
    
    # Forbidden zone check (none in this case)
    print("Forbidden zone check: Skipping (no forbidden zones)")
    
    # Parking zone occupancy
    print("\nParking zone processing:")
    for zone in loader.parking_zones:
        occupied_by = None
        for i, det in enumerate(detections):
            if _is_overlapping(zone, det["bbox"]):
                occupied_by = det["bbox"]
                print(f"  Zone {zone.id}: Detection {i} overlaps (bbox={det['bbox']})")
                break
            else:
                print(f"  Zone {zone.id}: Detection {i} does NOT overlap")
        
        if occupied_by:
            print(f"  Zone {zone.id}: MARKED AS OCCUPIED")
            result.zone_statuses.append(type('ZoneStatus', (), {
                'zone': zone,
                'status': 'Park Alanı Dolu',
                'vehicle_bbox': occupied_by,
                'occupied_since': now
            })())
        else:
            print(f"  Zone {zone.id}: MARKED AS AVAILABLE")
            result.zone_statuses.append(type('ZoneStatus', (), {
                'zone': zone,
                'status': 'Park Edilebilir',
                'vehicle_bbox': None,
                'occupied_since': None
            })())
    
    print(f"\n=== RESULTS ===")
    available = sum(1 for zs in result.zone_statuses if zs.status == 'Park Edilebilir')
    occupied = sum(1 for zs in result.zone_statuses if zs.status == 'Park Alanı Dolu')
    print(f"Available: {available}")
    print(f"Occupied: {occupied}")

if __name__ == "__main__":
    debug_overlap_calculations()