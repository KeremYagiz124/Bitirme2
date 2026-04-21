import json
import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parking.zone_loader import ZoneLoader, ParkingZone


@pytest.fixture
def sample_json(tmp_path):
    data = {
        "image": "test.jpg",
        "zones": [
            {"id": 1, "type": "parking",  "points": [[0,0],[100,0],[100,100],[0,100]]},
            {"id": 2, "type": "parking",  "points": [[200,0],[300,0],[300,100],[200,100]]},
            {"id": 3, "type": "forbidden","points": [[0,200],[400,200],[400,250],[0,250]]},
        ]
    }
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_load_zones_count(sample_json):
    loader = ZoneLoader(sample_json)
    assert len(loader.zones) == 3


def test_parking_zones_filter(sample_json):
    loader = ZoneLoader(sample_json)
    assert len(loader.parking_zones) == 2
    assert all(z.type == "parking" for z in loader.parking_zones)


def test_forbidden_zones_filter(sample_json):
    loader = ZoneLoader(sample_json)
    assert len(loader.forbidden_zones) == 1
    assert loader.forbidden_zones[0].type == "forbidden"


def test_zone_ids(sample_json):
    loader = ZoneLoader(sample_json)
    ids = [z.id for z in loader.zones]
    assert ids == [1, 2, 3]


def test_empty_zones(tmp_path):
    data = {"image": "test.jpg", "zones": []}
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(data))
    loader = ZoneLoader(str(p))
    assert loader.zones == []
    assert loader.parking_zones == []
    assert loader.forbidden_zones == []


def test_contains_center_inside():
    zone = ParkingZone(1, "parking", [[0,0],[100,0],[100,100],[0,100]])
    assert zone.contains_center([20, 20, 80, 80]) is True


def test_contains_center_outside():
    zone = ParkingZone(1, "parking", [[0,0],[100,0],[100,100],[0,100]])
    assert zone.contains_center([200, 200, 300, 300]) is False


def test_iou_full_overlap():
    zone = ParkingZone(1, "parking", [[0,0],[100,0],[100,100],[0,100]])
    iou = zone.iou_with_bbox([0, 0, 100, 100])
    assert iou == pytest.approx(1.0, abs=0.01)


def test_iou_no_overlap():
    zone = ParkingZone(1, "parking", [[0,0],[100,0],[100,100],[0,100]])
    iou = zone.iou_with_bbox([200, 200, 300, 300])
    assert iou == pytest.approx(0.0, abs=0.01)


def test_iou_partial_overlap():
    zone = ParkingZone(1, "parking", [[0,0],[100,0],[100,100],[0,100]])
    iou = zone.iou_with_bbox([50, 0, 150, 100])
    assert 0.3 < iou < 0.4


def test_find_zone_returns_best(sample_json):
    loader = ZoneLoader(sample_json)
    # bbox that overlaps zone #1 heavily
    zone = loader.find_zone([10, 10, 90, 90], iou_threshold=0.3)
    assert zone is not None
    assert zone.id == 1


def test_find_zone_returns_none_below_threshold(sample_json):
    loader = ZoneLoader(sample_json)
    # bbox far from all zones
    zone = loader.find_zone([500, 500, 600, 600], iou_threshold=0.3)
    assert zone is None
