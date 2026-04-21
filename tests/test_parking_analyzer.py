import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parking.zone_loader import ZoneLoader, ParkingZone
from src.parking.parking_analyzer import (
    ParkingAnalyzer,
    STATUS_AVAILABLE, STATUS_OCCUPIED, STATUS_FORBIDDEN,
)


@pytest.fixture
def loader_with_zones(tmp_path):
    data = {
        "image": "test.jpg",
        "zones": [
            {"id": 1, "type": "parking",  "points": [[0,0],[200,0],[200,200],[0,200]]},
            {"id": 2, "type": "parking",  "points": [[300,0],[500,0],[500,200],[300,200]]},
            {"id": 3, "type": "forbidden","points": [[0,300],[500,300],[500,400],[0,400]]},
        ]
    }
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(data))
    return ZoneLoader(str(p))


@pytest.fixture
def analyzer(loader_with_zones):
    return ParkingAnalyzer(loader_with_zones, iou_threshold=0.25)


# ── Zone statüs testleri ──────────────────────────────────────────

def test_empty_detections_all_available(analyzer):
    result = analyzer.analyze([])
    parking_statuses = [zs.status for zs in result.zone_statuses
                        if zs.zone.type == "parking"]
    assert all(s == STATUS_AVAILABLE for s in parking_statuses)


def test_vehicle_in_parking_zone_marks_occupied(analyzer):
    detections = [{"bbox": [10, 10, 190, 190], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    zone1 = next(zs for zs in result.zone_statuses if zs.zone.id == 1)
    assert zone1.status == STATUS_OCCUPIED


def test_vehicle_outside_all_zones_leaves_available(analyzer):
    detections = [{"bbox": [600, 600, 800, 800], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    parking_statuses = [zs.status for zs in result.zone_statuses
                        if zs.zone.type == "parking"]
    assert all(s == STATUS_AVAILABLE for s in parking_statuses)


def test_vehicle_in_forbidden_zone_sets_label(analyzer):
    detections = [{"bbox": [10, 310, 490, 390], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    assert result.vehicle_labels.get(0) == STATUS_FORBIDDEN


def test_vehicle_not_in_forbidden_zone_no_label(analyzer):
    detections = [{"bbox": [10, 10, 190, 190], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    assert 0 not in result.vehicle_labels


def test_second_parking_zone_stays_available(analyzer):
    detections = [{"bbox": [10, 10, 190, 190], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    zone2 = next(zs for zs in result.zone_statuses if zs.zone.id == 2)
    assert zone2.status == STATUS_AVAILABLE


# ── AnalysisResult sayaç testleri ────────────────────────────────

def test_available_count_no_vehicles(analyzer):
    result = analyzer.analyze([])
    assert result.available == 2


def test_occupied_count(analyzer):
    detections = [{"bbox": [10, 10, 190, 190], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    assert result.occupied == 1
    assert result.available == 1


def test_forbidden_vehicles_count(analyzer):
    detections = [
        {"bbox": [10, 310, 490, 390], "class_name": "car", "confidence": 0.9},
        {"bbox": [10, 10, 190, 190],  "class_name": "car", "confidence": 0.9},
    ]
    result = analyzer.analyze(detections)
    assert result.forbidden_vehicles == 1


def test_multiple_vehicles_multiple_occupied(analyzer):
    detections = [
        {"bbox": [10,  10, 190, 190], "class_name": "car", "confidence": 0.9},
        {"bbox": [310, 10, 490, 190], "class_name": "car", "confidence": 0.9},
    ]
    result = analyzer.analyze(detections)
    assert result.occupied == 2
    assert result.available == 0


# ── IoU eşiği testleri ────────────────────────────────────────────

def test_low_iou_below_threshold_stays_available(loader_with_zones):
    analyzer = ParkingAnalyzer(loader_with_zones, iou_threshold=0.8)
    # Partial overlap — will be below 0.8
    detections = [{"bbox": [150, 150, 350, 350], "class_name": "car", "confidence": 0.9}]
    result = analyzer.analyze(detections)
    zone1 = next(zs for zs in result.zone_statuses if zs.zone.id == 1)
    assert zone1.status == STATUS_AVAILABLE


# ── Forbidden zone zone_status listede mevcut ────────────────────

def test_forbidden_zone_in_zone_statuses(analyzer):
    result = analyzer.analyze([])
    forbidden_statuses = [zs for zs in result.zone_statuses
                          if zs.zone.type == "forbidden"]
    assert len(forbidden_statuses) == 1
    assert forbidden_statuses[0].status == STATUS_FORBIDDEN
