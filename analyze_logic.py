#!/usr/bin/env python3
"""Investigate potential issues in the parking analysis logic."""

import cv2
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detection.vehicle_detector import VehicleDetector
from src.parking import ZoneLoader, ParkingAnalyzer

def analyze_parking_logic():
    """Deep dive into the parking analysis logic to find potential issues."""
    
    out_lines = []
    def log(msg=""):
        out_lines.append(msg)
        print(msg)
    
    log("=== DEEP DIVE INTO PARKING ANALYSIS LOGIC ===")
    
    # Load test data
    image_path = "data/raw/araba1.jpeg"
    zones_path = "data/raw/araba1.json"
    
    frame = cv2.imread(image_path)
    loader = ZoneLoader(zones_path)
    detector = VehicleDetector(conf=0.5)
    analyzer = ParkingAnalyzer(loader, iou_threshold=0.25)
    
    # Get detections
    detections = detector.detect(frame)
    
    log(f"Frame shape: {frame.shape}")
    log(f"Number of detections: {len(detections)}")
    log(f"IoU threshold: {analyzer.iou_threshold}")
    
    log("\n=== DETAILED ZONE ANALYSIS ===")
    
    # For each zone, let's manually compute what the analyzer does
    for zone in loader.parking_zones:
        log(f"\nZone {zone.id}:")
        log(f"  Type: {zone.type}")
        log(f"  Points shape: {zone.polygon.shape}")
        log(f"  Bounding box: [{zone.polygon[:, 0].min()}, {zone.polygon[:, 1].min()}, {zone.polygon[:, 0].max()}, {zone.polygon[:, 1].max()}]")
        
        # Calculate zone properties
        zone_area = cv2.contourArea(zone.polygon)
        log(f"  Polygon area: {zone_area}")
        
        # Check each detection
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            log(f"    Detection {i}: {det['class_name']} conf={det['confidence']:.3f}")
            log(f"      BBox: {bbox}")
            
            # Compute the metrics used in the analyzer
            coverage = zone.coverage_by_bbox(bbox)
            contains_center = zone.contains_center(bbox)
            iou = zone.iou_with_bbox(bbox)
            
            log(f"      Coverage (bbox covers zone): {coverage:.4f}")
            log(f"      Contains center: {contains_center}")
            log(f"      IoU: {iou:.4f}")
            
            # The actual condition used in _is_overlapping
            would_trigger = coverage >= analyzer.iou_threshold or contains_center
            log(f"      Would trigger occupancy (coverage >= {analyzer.iou_threshold} OR center): {would_trigger}")
            
            if would_trigger:
                log(f"      --> This detection would MARK ZONE {zone.id} AS OCCUPIED")
    
    log("\n=== CHECKING FOR EDGE CASES IN LOGIC ===")
    
    # Let's look at the actual _is_overlapping function implementation
    log("Examining the _is_overlapping function from parking_analyzer.py:")
    log("""
    def _is_overlapping(zone, bbox: list[float]) -> bool:
        \"\"\"Zone-araç örtüşmesi: coverage VEYA merkez noktası kontrolü.

        Araç kamerası perspektifinde araç bbox'ları zone'dan çok büyük
        olabilir; IoU düşük kalır ama coverage yüksek olur. İkisini
        birden kontrol ederek her iki durumu da yakalarız.
        \"\"\"
        if zone.coverage_by_bbox(bbox) >= self.iou_threshold:
            return True
        if zone.contains_center(bbox):
            return True
        return False
    """)
    
    log("\nThe logic is: IF (coverage >= threshold) OR (contains_center) THEN overlap")
    log("This means a zone will be considered occupied if EITHER:")
    log("  1. The detection covers at least threshold% of the zone area, OR")
    log("  2. The center of the detection is inside the zone")
    
    log("\n=== POTENTIAL ISSUES TO INVESTIGATE ===")
    log("1. Coverage calculation might be incorrect for perspective-distorted zones")
    log("2. The threshold of 0.25 might be too low for some zone shapes")
    log("3. There might be an issue with how we're interpreting what constitutes 'coverage'")
    log("4. The contains_center check might be problematic with skewed polygons")
    
    # Let's test some extreme cases
    log("\n=== TESTING EXTREME CASES ===")
    
    # Case: Very small detection that just touches the edge of a zone
    # This could potentially cause false positives if coverage calculation is wrong
    
    # Actually, let's check if there are any issues with the actual data
    # by looking at what the expected values are in the JSON
    
    log("\n=== CHECKING EXPECTED VALUES FROM JSON ===")
    with open(zones_path) as f:
        zone_data = json.load(f)
    
    for zone in zone_data['zones']:
        if zone['type'] == 'parking':
            log(f"Zone {zone['id']}: expected = '{zone.get('expected', 'NOT SET')}'")
    
    log("\n=== RUNNING FULL ANALYSIS ===")
    result = analyzer.analyze(detections)
    
    log(f"Results:")
    log(f"  Available: {result.available}")
    log(f"  Occupied: {result.occupied}")
    log(f"  Forbidden vehicles: {result.forbidden_vehicles}")
    
    log("\nDetailed zone status:")
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            log(f"  Zone {zs.zone.id}: {zs.status}")
            if zs.vehicle_bbox:
                log(f"    Occupied by vehicle detection")
            if zs.occupied_since:
                import time
                dur = time.time() - zs.occupied_since
                log(f"    Occupied for: {dur:.1f} seconds")
    
    # Compare with expected
    log("\n=== COMPARISON WITH EXPECTED VALUES ===")
    expected_map = {zone['id']: zone.get('expected', 'unknown') for zone in zone_data['zones'] if zone['type'] == 'parking'}
    
    correct = 0
    total = 0
    
    for zs in result.zone_statuses:
        if zs.zone.type == "parking":
            total += 1
            zone_id = zs.zone.id
            expected = expected_map.get(zone_id, "unknown")
            actual = zs.status
            
            # Map our statuses to expected values
            actual_expected = "available" if actual == "Park Edilebilir" else "occupied" if actual == "Park Alanı Dolu" else "unknown"
            
            is_correct = (expected == actual_expected) or (expected == "unknown")
            if is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                status = "✗ INCORRECT"
                
            log(f"  Zone {zone_id}: Expected={expected}, Actual={actual_expected} {status}")
    
    log(f"\nOverall accuracy: {correct}/{total} zones correct")
    
    if correct == total:
        log("All zone classifications match expected values!")
    else:
        log("MISMATCH DETECTED - this indicates the problem!")
    
    return correct == total

if __name__ == "__main__":
    analyze_parking_logic()