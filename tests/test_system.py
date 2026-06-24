"""Sistem bütünleşik test paketi.

Kapsam:
  - AlertSystem: fire/throttle/check_occupancy/check_forbidden/check_no_fit
  - VehicleTracker: statik tespit, park süresi, ego-motion reset
  - StreetParkingDetector: boş alan tespiti, max_edge_extension_ratio=0.20
  - DrivableAreaSegmenter: fallback (model yok → graceful degradation)
  - Overhead/üst görüş perspektif testi (yeni otopark senaryosu)
  - Kamera canlı akış simülasyonu (tekrarlı kare testi)
  - Sokak modu ground truth: 3 görüntü, Mikro F1 = %100
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui.alert_system import Alert, AlertLevel, AlertSystem
from src.detection.vehicle_tracker import VehicleTracker
from src.detection.street_parking_detector import StreetParkingDetector
from src.detection.drivable_area import DrivableAreaSegmenter
from src.evaluation import metrics as EM


# ─── Yardımcı ────────────────────────────────────────────────────────────────

def _det(x1, y1, x2, y2, cls=2):
    return {"bbox": [x1, y1, x2, y2], "class_id": cls, "confidence": 0.9}


def _blank(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# AlertSystem testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertSystem:

    def test_fire_returns_true_first_time(self):
        a = AlertSystem(throttle_sec=30)
        assert a.fire("test", AlertLevel.INFO, "msg") is True

    def test_fire_throttled_returns_false(self):
        a = AlertSystem(throttle_sec=30)
        a.fire("test", AlertLevel.INFO, "msg")
        assert a.fire("test", AlertLevel.INFO, "msg") is False

    def test_fire_after_throttle_window(self):
        a = AlertSystem(throttle_sec=0.05)
        a.fire("t", AlertLevel.INFO, "x")
        time.sleep(0.06)
        assert a.fire("t", AlertLevel.INFO, "x") is True

    def test_listener_called(self):
        received = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(received.append)
        a.fire("c", AlertLevel.WARNING, "uyarı")
        assert len(received) == 1
        assert received[0].code == "c"
        assert received[0].level == AlertLevel.WARNING

    def test_multiple_listeners(self):
        calls = [0, 0]
        a = AlertSystem(throttle_sec=0)
        a.add_listener(lambda _: calls.__setitem__(0, calls[0] + 1))
        a.add_listener(lambda _: calls.__setitem__(1, calls[1] + 1))
        a.fire("x", AlertLevel.INFO, "")
        assert calls == [1, 1]

    def test_history_grows(self):
        a = AlertSystem(throttle_sec=0)
        a.fire("a", AlertLevel.INFO, "")
        a.fire("b", AlertLevel.INFO, "")
        assert len(a._history) == 2

    def test_history_max_capped(self):
        a = AlertSystem(throttle_sec=0, max_history=3)
        for i in range(5):
            a.fire(f"k{i}", AlertLevel.INFO, "")
        assert len(a._history) == 3

    def test_clear_code_resets_throttle(self):
        a = AlertSystem(throttle_sec=60)
        a.fire("z", AlertLevel.CRITICAL, "")
        assert a.fire("z", AlertLevel.CRITICAL, "") is False
        a.clear_code("z")
        assert a.fire("z", AlertLevel.CRITICAL, "") is True

    # ── check_occupancy ──

    def test_check_occupancy_park_full(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=0, occupied=5)
        assert any(al.code == "park_full" and al.level == AlertLevel.CRITICAL
                   for al in fired)

    def test_check_occupancy_park_high(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=1, occupied=9)   # %90 dolu
        assert any(al.code == "park_high" and al.level == AlertLevel.WARNING
                   for al in fired)

    def test_check_occupancy_low_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_occupancy(available=8, occupied=2)   # %20 dolu
        assert fired == []

    def test_check_occupancy_total_zero_no_crash(self):
        a = AlertSystem(throttle_sec=0)
        a.check_occupancy(available=0, occupied=0)   # toplam 0 — crash olmamalı

    def test_check_occupancy_clears_on_low(self):
        a = AlertSystem(throttle_sec=0)
        a.check_occupancy(available=0, occupied=5)   # park_full
        a.check_occupancy(available=5, occupied=1)   # %17 → throttle temizlenmeli
        assert "park_full" not in a._last_fired
        assert "park_high" not in a._last_fired

    # ── check_forbidden ──

    def test_check_forbidden_fires_when_positive(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_forbidden(2)
        assert any(al.code == "forbidden_park" for al in fired)

    def test_check_forbidden_zero_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_forbidden(0)
        assert fired == []

    # ── check_no_fit ──

    def test_check_no_fit_fires_when_no_space_fits(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=3)
        assert any(al.code == "no_fit" and al.level == AlertLevel.INFO
                   for al in fired)

    def test_check_no_fit_clears_when_fit_exists(self):
        a = AlertSystem(throttle_sec=0)
        a.fire("no_fit", AlertLevel.INFO, "")
        a._last_fired["no_fit"] = time.time()
        a.check_no_fit(fit_count=1, total_empty=3)
        assert "no_fit" not in a._last_fired

    def test_check_no_fit_empty_zero_no_alert(self):
        fired = []
        a = AlertSystem(throttle_sec=0)
        a.add_listener(fired.append)
        a.check_no_fit(fit_count=0, total_empty=0)
        assert fired == []


# ══════════════════════════════════════════════════════════════════════════════
# VehicleTracker testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestVehicleTracker:

    def test_new_detection_assigned(self):
        tr = VehicleTracker()
        flags = tr.update([_det(0, 0, 100, 60)])
        assert len(flags) == 1

    def test_not_static_immediately(self):
        tr = VehicleTracker(min_history=5)
        flags = tr.update([_det(0, 0, 100, 60)])
        assert flags[0] is False

    def test_static_after_enough_frames(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(20):
            flags = tr.update(det)
        assert flags[0] is True

    def test_moving_vehicle_not_static(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.05, max_misses=50)
        for i in range(20):
            flags = tr.update([_det(i * 5, 10, i * 5 + 100, 70)])
        assert flags[0] is False

    def test_track_removed_after_max_misses(self):
        tr = VehicleTracker(max_misses=3)
        tr.update([_det(0, 0, 100, 60)])
        for _ in range(5):
            tr.update([])
        assert len(tr._tracks) == 0

    def test_get_static_tracks_empty_initially(self):
        tr = VehicleTracker()
        assert tr.get_static_tracks() == []

    def test_get_static_tracks_with_duration_returns_tuples(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(25):
            tr.update(det)
        pairs = tr.get_static_tracks_with_duration(min_frames=20)
        assert len(pairs) == 1
        bbox, dur = pairs[0]
        assert len(bbox) == 4
        assert dur >= 0.0

    def test_duration_increases_over_time(self):
        tr = VehicleTracker(min_history=3, max_disp_ratio=0.5, max_misses=50)
        det = [_det(10, 10, 110, 70)]
        for _ in range(25):
            tr.update(det)
        d1 = tr.get_static_tracks_with_duration(min_frames=20)[0][1]
        tr.update(det, dt=0.05)
        d2 = tr.get_static_tracks_with_duration(min_frames=20)[0][1]
        assert d2 > d1

    def test_first_seen_set_on_creation(self):
        tr = VehicleTracker()
        before = time.time()
        tr.update([_det(0, 0, 100, 60)])
        after = time.time()
        track = list(tr._tracks.values())[0]
        assert before <= track.first_seen <= after

    def test_reset_clears_tracks(self):
        tr = VehicleTracker()
        tr.update([_det(0, 0, 100, 60)])
        tr.reset()
        assert len(tr._tracks) == 0
        assert tr._prev_gray is None

    def test_multiple_tracks_independent(self):
        tr = VehicleTracker(iou_threshold=0.1, max_misses=50)
        tr.update([_det(0, 0, 80, 50), _det(300, 0, 380, 50)])
        assert len(tr._tracks) == 2


# ══════════════════════════════════════════════════════════════════════════════
# StreetParkingDetector testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestStreetParkingDetector:

    def _run(self, frame, dets, **kw):
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            **kw
        )
        return det.analyze(frame, dets)

    def test_no_vehicles_no_spaces(self):
        frame = _blank(640, 480)
        result = self._run(frame, [])
        assert result["empty_count"] == 0
        assert result["occupied_count"] == 0

    def test_single_vehicle_no_gap(self):
        frame = _blank(640, 480)
        dets = [_det(100, 300, 300, 400)]
        result = self._run(frame, dets, detect_row_edges=False)
        assert result["occupied_count"] == 1

    def test_gap_between_two_vehicles_detected(self):
        frame = _blank(1200, 500)
        # İki araç arası ~300px boşluk — araç genişliği ~200px → gap_ratio ~1.5 (>0.20)
        dets = [_det(50, 350, 250, 450), _det(550, 350, 750, 450)]
        result = self._run(frame, dets, detect_row_edges=False, max_gap_ratio=10.0)
        assert result["empty_count"] >= 1

    def test_result_keys_present(self):
        frame = _blank(640, 480)
        result = self._run(frame, [_det(100, 300, 300, 400)])
        for key in ("empty_count", "occupied_count", "empty_spaces",
                    "parked", "slot_sizes_m", "scale_m_per_px"):
            assert key in result

    def test_max_edge_extension_ratio_limits_slots(self):
        """0.20 ile geniş sağ kenar tek slota bölünmeli (0.40 üçe böler)."""
        frame = _blank(1920, 1080)
        # Araçlar x=50..1173 bölgesinde — sağda 747px kenar boşluğu
        dets = [
            _det(50,  700, 250,  800),
            _det(260, 700, 460,  800),
            _det(470, 700, 670,  800),
            _det(680, 700, 880,  800),
            _det(890, 700, 1090, 800),
            _det(1100,700, 1173, 800),
        ]
        r_loose = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.40,
            detect_row_edges=True,
        ).analyze(frame, dets)

        r_tight = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.20,
            detect_row_edges=True,
        ).analyze(frame, dets)

        # Sıkı parametre ile kenar slotu daha az üretmeli
        assert r_tight["empty_count"] <= r_loose["empty_count"]

    def test_slot_sizes_length_matches_empty_count(self):
        frame = _blank(1200, 500)
        dets = [_det(50, 350, 250, 450), _det(550, 350, 750, 450)]
        result = self._run(frame, dets, detect_row_edges=False, max_gap_ratio=10.0)
        assert len(result["slot_sizes_m"]) == result["empty_count"]

    def test_reset_history_clears_state(self):
        det = StreetParkingDetector(smoothing_frames=5, road_color_check=False)
        frame = _blank(640, 480)
        det.analyze(frame, [_det(100, 300, 300, 400)])
        det.reset_history()
        assert len(det._history) == 0
        assert det._analyze_count == 0

    def test_draw_returns_same_shape(self):
        frame = _blank(640, 480)
        dets = [_det(100, 300, 300, 400)]
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1, road_color_check=False
        )
        result = det.analyze(frame, dets)
        out = det.draw(frame.copy(), result)
        assert out.shape == frame.shape


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth değerlendirme testi (3 görüntü, Mikro F1 = %100)
# ══════════════════════════════════════════════════════════════════════════════

GT_IMAGES = [
    ("1.png", 0),
    ("2.png", 2),
    ("3.png", 3),
]


@pytest.mark.skipif(
    not all((ROOT / p).exists() for p, _ in GT_IMAGES),
    reason="Ground truth görüntüleri bulunamadı",
)
class TestGroundTruth:

    @pytest.fixture(scope="class")
    def detector(self):
        from src.detection.vehicle_detector import VehicleDetector
        return VehicleDetector(conf=0.35)

    def _eval(self, img_name, expected, detector):
        frame = cv2.imread(str(ROOT / img_name))
        assert frame is not None, f"{img_name} açılamadı"
        det = StreetParkingDetector(max_edge_extension_ratio=0.20)
        dets = detector.detect(frame)
        result = det.analyze(frame, dets)
        return result["empty_count"], expected

    def test_1png_no_empty(self, detector):
        detected, expected = self._eval("1.png", 0, detector)
        assert detected == expected, f"1.png: expected={expected} detected={detected}"

    def test_2png_two_empty(self, detector):
        detected, expected = self._eval("2.png", 2, detector)
        assert detected == expected, f"2.png: expected={expected} detected={detected}"

    def test_3png_three_empty(self, detector):
        detected, expected = self._eval("3.png", 3, detector)
        assert detected == expected, f"3.png: expected={expected} detected={detected}"

    def test_micro_f1_100(self, detector):
        total_tp = total_fp = total_fn = 0
        for img, expected in GT_IMAGES:
            detected, _ = self._eval(img, expected, detector)
            total_tp += min(detected, expected)
            total_fp += max(0, detected - expected)
            total_fn += max(0, expected - detected)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        assert micro_f1 == pytest.approx(1.0), f"Mikro F1 = {micro_f1:.3f}, beklenen 1.0"


# ══════════════════════════════════════════════════════════════════════════════
# DrivableAreaSegmenter fallback testi
# ══════════════════════════════════════════════════════════════════════════════

class TestDrivableAreaFallback:

    def test_missing_model_available_false(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        assert seg.available is False

    def test_infer_returns_none_none_when_unavailable(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        frame = _blank(640, 480)
        mask, overlay = seg.infer(frame)
        assert mask is None
        assert overlay is None

    def test_infer_does_not_raise_on_none_frame(self):
        seg = DrivableAreaSegmenter(model_path="/nonexistent/model.pt")
        mask, overlay = seg.infer(None)
        assert mask is None and overlay is None


# ══════════════════════════════════════════════════════════════════════════════
# Üst görüş (overhead) perspektif testi
# ══════════════════════════════════════════════════════════════════════════════

class TestOverheadPerspective:
    """Kuş bakışı otopark senaryosu: araçlar ızgara düzeninde, aralarında boşluklar."""

    def _make_overhead_frame(self, n_cars=4, car_w=120, car_h=60,
                              gap=80, margin=40, img_w=900, img_h=300):
        frame = np.full((img_h, img_w, 3), 128, dtype=np.uint8)
        dets = []
        x = margin
        y = (img_h - car_h) // 2
        for _ in range(n_cars):
            cv2.rectangle(frame, (x, y), (x + car_w, y + car_h), (50, 50, 200), -1)
            dets.append(_det(x, y, x + car_w, y + car_h))
            x += car_w + gap
        return frame, dets

    def test_detects_gaps_between_overhead_vehicles(self):
        frame, dets = self._make_overhead_frame(n_cars=3, gap=100)
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.10,
        )
        result = det.analyze(frame, dets)
        assert result["empty_count"] >= 1

    def test_no_gaps_when_cars_packed_tightly(self):
        # 2 piksel boşlukla yanyana park — gap_ratio eşiği altında kalmalı
        frame, dets = self._make_overhead_frame(n_cars=5, gap=2, img_w=1000)
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.05,
        )
        result = det.analyze(frame, dets)
        # 2 piksel boşluk ortalama araç genişliğine (120px) kıyasla çok küçük
        assert result["empty_count"] < len(dets)

    def test_result_keys_present_overhead(self):
        frame, dets = self._make_overhead_frame()
        det = StreetParkingDetector(smoothing_frames=1, smoothing_min_hits=1,
                                    road_color_check=False)
        result = det.analyze(frame, dets)
        for key in ("empty_count", "occupied_count", "slot_sizes_m", "empty_spaces"):
            assert key in result


# ══════════════════════════════════════════════════════════════════════════════
# Kamera canlı akış simülasyonu testi
# ══════════════════════════════════════════════════════════════════════════════

class TestCameraSimulation:
    """Sabit kameradan gelen tekrarlı karelerin tracker'ı kararlı hâle getirdiğini doğrular."""

    def _static_detections(self):
        return [
            _det(100, 200, 300, 350),
            _det(320, 200, 520, 350),
            _det(540, 200, 740, 350),
        ]

    def test_tracker_stabilizes_after_enough_frames(self):
        tracker = VehicleTracker(min_history=5, max_disp_ratio=0.20)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        static = tracker.get_static_tracks()
        assert len(static) == len(dets)

    def test_duration_increases_across_simulated_frames(self):
        tracker = VehicleTracker(min_history=5)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        tracks1 = tracker.get_static_tracks_with_duration()
        tracker.update(dets, frame, dt=0.05)
        tracks2 = tracker.get_static_tracks_with_duration()
        durations1 = [d for _, d in tracks1]
        durations2 = [d for _, d in tracks2]
        assert all(d2 > d1 for d1, d2 in zip(durations1, durations2))

    def test_new_vehicle_mid_stream_tracked(self):
        tracker = VehicleTracker(min_history=5)
        frame = _blank(800, 500)
        base = self._static_detections()
        for _ in range(25):
            tracker.update(base, frame)
        extra = base + [_det(760, 200, 960, 350)]
        for _ in range(25):
            tracker.update(extra, frame)
        static = tracker.get_static_tracks()
        assert len(static) == len(extra)

    def test_disappeared_vehicle_removed(self):
        tracker = VehicleTracker(min_history=5, max_misses=5)
        frame = _blank(800, 500)
        dets = self._static_detections()
        for _ in range(25):
            tracker.update(dets, frame)
        for _ in range(10):
            tracker.update([], frame)
        assert len(tracker.get_static_tracks()) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Dik Park Modu testleri
# ══════════════════════════════════════════════════════════════════════════════

def _perp_det_front(x1, y1, x2, y2):
    """Ön görünüm dik park: araç kareye yakın (bw/bh ≈ 1.2)."""
    return {"bbox": [x1, y1, x2, y2], "class_id": 2, "confidence": 0.9}


def _perp_det_side(x1, y1, x2, y2):
    """Yan görünüm dik park: araç geniş (bw/bh ≈ 3.0)."""
    return {"bbox": [x1, y1, x2, y2], "class_id": 2, "confidence": 0.9}


class TestPerpendiculaMode:
    """Dik park (perpendicular) modu birim testleri."""

    # ── Görünüm açısı tespiti ─────────────────────────────────────

    def test_front_view_detected_square_cars(self):
        """Kareye yakın bboxlar (bw/bh ≈ 1.2) → ön görünüm."""
        frame = _blank(800, 400)
        # 3 araç, her biri 100×80 px (bw/bh=1.25)
        dets = [
            _perp_det_front(50, 150, 150, 230),
            _perp_det_front(200, 150, 300, 230),
            _perp_det_front(350, 150, 450, 230),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result["perp_side_view"] is False

    def test_side_view_detected_wide_cars(self):
        """Geniş bboxlar (bw/bh ≈ 3.0) → yan görünüm."""
        frame = _blank(1200, 400)
        # 3 araç, her biri 300×100 px (bw/bh=3.0)
        dets = [
            _perp_det_side(50,  150, 350, 250),
            _perp_det_side(400, 150, 700, 250),
            _perp_det_side(750, 150, 1050, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result["perp_side_view"] is True

    # ── Bölme yok (n=1) ──────────────────────────────────────────

    def test_no_subdivision_large_gap(self):
        """Dik modda büyük boşluk tek slot olarak raporlanmalı."""
        frame = _blank(1000, 400)
        # Tek araç ortada → her iki yanda büyük boşluk
        dets = [_perp_det_front(400, 150, 550, 250)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_edge_extension_ratio=0.40,
            min_gap_ratio=0.20,
        )
        result = det.analyze(frame, dets)
        # Her boşluk tek slot olmalı — araç genişliğine bölünmemeli
        for slot in result["empty_spaces"]:
            x1, _, x2, _ = slot
            # Slot genişliği = toplam boşluk, çok daha büyük olmalı
            assert (x2 - x1) > 130  # araç genişliği (150px) den büyük

    def test_parallel_mode_subdivides(self):
        """Paralel modda aynı boşluk birden fazla slota bölünebilir."""
        frame = _blank(1000, 400)
        # 2 araç arası büyük boşluk (5× araç genişliği)
        dets = [
            _det(50, 150, 150, 250),   # araç genişliği 100px
            _det(650, 150, 750, 250),  # 500px boşluk → ~5 slot
        ]
        det = StreetParkingDetector(
            orientation="parallel",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
            max_spaces_per_gap=4,
            min_gap_ratio=0.20,
        )
        result = det.analyze(frame, dets)
        # Paralel modda büyük boşluk birden çok slota bölünmeli
        assert result["empty_count"] >= 2

    # ── Ölçek referansı ──────────────────────────────────────────

    def test_scale_uses_width_ref_in_front_view(self):
        """Ön görünümde ölçek ref_car_width_m'e göre hesaplanmalı."""
        frame = _blank(800, 400)
        dets = [
            _perp_det_front(100, 150, 200, 230),  # 100px genişlik
            _perp_det_front(350, 150, 450, 230),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_width_m=2.0)
        scale = result["scale_m_per_px"]
        assert scale is not None
        # scale = 2.0 / 100px = 0.02 m/px
        assert abs(scale - 0.02) < 0.005

    def test_scale_uses_length_ref_in_side_view(self):
        """Yan görünümde ölçek ref_car_length_m'e göre hesaplanmalı."""
        frame = _blank(1200, 400)
        dets = [
            _perp_det_side(50,  150, 350, 250),  # 300px genişlik
            _perp_det_side(400, 150, 700, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        scale = result["scale_m_per_px"]
        assert scale is not None
        # scale = 4.5 / 300px = 0.015 m/px
        assert abs(scale - 0.015) < 0.003

    # ── Sığma kontrolü ───────────────────────────────────────────

    def test_slot_size_reported_in_meters(self):
        """Slot boyutu metre cinsinden raporlanmalı."""
        frame = _blank(1200, 400)
        dets = [
            _perp_det_side(50,  150, 350, 250),
            _perp_det_side(750, 150, 1050, 250),
        ]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        if result["empty_count"] > 0:
            w_m, h_m = result["slot_sizes_m"][0]
            assert w_m > 0
            assert h_m >= 0

    # ── result dict anahtarları ───────────────────────────────────

    def test_result_has_perp_side_view_key(self):
        frame = _blank(800, 400)
        dets = [_perp_det_front(200, 150, 350, 270)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert "perp_side_view" in result
        assert isinstance(result["perp_side_view"], bool)

    def test_parallel_result_no_perp_flag(self):
        """Paralel modda perp_side_view False olmalı (default)."""
        frame = _blank(800, 400)
        dets = [_det(100, 200, 300, 320), _det(400, 200, 600, 320)]
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets)
        assert result.get("perp_side_view") is False

    # ── reset_history ─────────────────────────────────────────────

    def test_reset_clears_perp_state(self):
        """reset_history sonrası perp_side_view False'a dönmeli."""
        frame = _blank(1200, 400)
        dets = [_perp_det_side(50, 150, 350, 250)]
        det = StreetParkingDetector(
            orientation="perpendicular",
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        det.analyze(frame, dets)
        det.reset_history()
        assert getattr(det, "_perp_side_view", False) is False


# ══════════════════════════════════════════════════════════════════════════════
# Sınıf-duyarlı ölçek tahmini testleri (büyük araç desteği)
# ══════════════════════════════════════════════════════════════════════════════

def _det_named(x1, y1, x2, y2, class_name, cls=2):
    return {"bbox": [x1, y1, x2, y2], "class_id": cls,
            "class_name": class_name, "confidence": 0.9}


class TestClassAwareScale:
    """Park etmiş araç sınıfına göre m/px ölçeğinin doğru hesaplanması."""

    def test_real_dim_lookup_car_length(self):
        d = StreetParkingDetector._real_dim_for("car", False, 4.5, 2.0)
        assert d == 4.5

    def test_real_dim_lookup_bus_length(self):
        d = StreetParkingDetector._real_dim_for("bus", False, 4.5, 2.0)
        assert d == 12.0

    def test_real_dim_lookup_truck_length(self):
        d = StreetParkingDetector._real_dim_for("truck", False, 4.5, 2.0)
        assert d == 8.0

    def test_real_dim_lookup_width(self):
        d = StreetParkingDetector._real_dim_for("bus", True, 4.5, 2.0)
        assert d == 2.5

    def test_real_dim_unknown_falls_back_to_default(self):
        d = StreetParkingDetector._real_dim_for(None, False, 4.5, 2.0)
        assert d == 4.5

    def test_class_for_box_matches_by_iou(self):
        dets = [_det_named(100, 200, 300, 320, "truck")]
        cname = StreetParkingDetector._class_for_box([100, 200, 300, 320], dets)
        assert cname == "truck"

    def test_class_for_box_no_match_returns_none(self):
        dets = [_det_named(100, 200, 300, 320, "car")]
        cname = StreetParkingDetector._class_for_box([900, 900, 950, 950], dets)
        assert cname is None

    def test_scale_consistent_with_mixed_vehicle_sizes(self):
        """Sıraya otobüs karışsa bile ölçek otomobil-tutarlı kalmalı.

        Otomobil: 4.5m / 100px = 0.045 m/px
        Otobüs:   12.0m / 267px ≈ 0.045 m/px (aynı ölçek, daha büyük piksel)
        Sınıf bilgisi olmadan otobüsün 267px'i ölçeği ~0.017'ye düşürürdü.
        """
        frame = _blank(1400, 400)
        dets = [
            _det_named(50,  150, 150, 270, "car"),    # 100px
            _det_named(250, 150, 350, 270, "car"),    # 100px
            _det_named(450, 150, 717, 270, "bus"),    # 267px
        ]
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        scale = result["scale_m_per_px"]
        assert scale is not None
        # Sınıf-duyarlı: medyan(0.045, 0.045, 0.0449) ≈ 0.045
        assert abs(scale - 0.045) < 0.005

    def test_scale_without_class_uses_default(self):
        """class_name yoksa kullanıcı varsayılan boyutuna düşülmeli."""
        frame = _blank(800, 400)
        dets = [_det(100, 200, 200, 320), _det(300, 200, 400, 320)]  # 100px
        det = StreetParkingDetector(
            smoothing_frames=1, smoothing_min_hits=1,
            road_color_check=False,
        )
        result = det.analyze(frame, dets, ref_car_length_m=4.5)
        scale = result["scale_m_per_px"]
        assert scale is not None
        assert abs(scale - 0.045) < 0.005


# ══════════════════════════════════════════════════════════════════════════════
# Değerlendirme metrikleri testleri
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics:
    """src/evaluation/metrics.py saf fonksiyon testleri."""

    def test_bbox_iou_identical(self):
        assert EM.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_bbox_iou_disjoint(self):
        assert EM.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_bbox_iou_half_overlap(self):
        # 10x10 ve 10x10, yatayda yarı örtüşme → inter=50, union=150 → 1/3
        assert EM.bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1/3)

    def test_match_all_correct(self):
        preds = [(0, 0, 10, 10), (20, 0, 30, 10)]
        gts = [(0, 0, 10, 10), (20, 0, 30, 10)]
        tp, fp, fn, matches = EM.match_detections(preds, gts, 0.5)
        assert (tp, fp, fn) == (2, 0, 0)
        assert len(matches) == 2

    def test_match_false_positive(self):
        preds = [(0, 0, 10, 10), (100, 100, 110, 110)]
        gts = [(0, 0, 10, 10)]
        tp, fp, fn, _ = EM.match_detections(preds, gts, 0.5)
        assert (tp, fp, fn) == (1, 1, 0)

    def test_match_false_negative(self):
        preds = [(0, 0, 10, 10)]
        gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
        tp, fp, fn, _ = EM.match_detections(preds, gts, 0.5)
        assert (tp, fp, fn) == (1, 0, 1)

    def test_match_empty_predictions(self):
        tp, fp, fn, _ = EM.match_detections([], [(0, 0, 10, 10)], 0.5)
        assert (tp, fp, fn) == (0, 0, 1)

    def test_precision_recall_f1_perfect(self):
        p, r, f = EM.precision_recall_f1(10, 0, 0)
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_precision_recall_f1_zero_division(self):
        p, r, f = EM.precision_recall_f1(0, 0, 0)
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_average_precision_perfect(self):
        preds = [[(0, 0, 10, 10)], [(0, 0, 10, 10)]]
        gts = [[(0, 0, 10, 10)], [(0, 0, 10, 10)]]
        ap = EM.average_precision(preds, gts, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_average_precision_no_gt(self):
        ap = EM.average_precision([[(0, 0, 10, 10)]], [[]], iou_threshold=0.5)
        assert ap == 0.0

    def test_evaluate_detections_full(self):
        preds = [[(0, 0, 10, 10), (20, 0, 30, 10)], [(0, 0, 10, 10)]]
        gts = [[(0, 0, 10, 10), (20, 0, 30, 10)], [(0, 0, 10, 10)]]
        m = EM.evaluate_detections(preds, gts, iou_threshold=0.5)
        assert m.tp == 3 and m.fp == 0 and m.fn == 0
        assert m.f1 == pytest.approx(1.0)

    def test_evaluate_counts_perfect(self):
        m = EM.evaluate_counts([0, 2, 3], [0, 2, 3])
        assert m.micro_f1 == pytest.approx(1.0)
        assert m.mae == 0.0

    def test_evaluate_counts_with_errors(self):
        # tahmin: 1,3,3  beklenen: 0,2,3
        # TP=min: 0+2+3=5, FP=1+1+0=2, FN=0
        m = EM.evaluate_counts([1, 3, 3], [0, 2, 3])
        assert m.tp == 5 and m.fp == 2 and m.fn == 0
        assert m.mae == pytest.approx(2/3)

    def test_evaluate_counts_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            EM.evaluate_counts([1, 2], [1])


class TestEvaluationPlots:
    """Grafik üretiminin dosya oluşturduğunu doğrular (Agg backend)."""

    def test_confusion_matrix_png_created(self, tmp_path):
        from src.evaluation import plots as P
        out = tmp_path / "cm.png"
        P.plot_confusion_matrix(5, 1, 2, 0, out)
        assert out.exists() and out.stat().st_size > 0

    def test_metric_bars_png_created(self, tmp_path):
        from src.evaluation import plots as P
        out = tmp_path / "bars.png"
        P.plot_metric_bars({"Precision": 0.9, "Recall": 0.8, "F1": 0.85}, out)
        assert out.exists() and out.stat().st_size > 0

    def test_per_image_counts_png_created(self, tmp_path):
        from src.evaluation import plots as P
        out = tmp_path / "counts.png"
        P.plot_per_image_counts([1, 2, 3], [0, 2, 3], out)
        assert out.exists() and out.stat().st_size > 0

    def test_method_comparison_png_created(self, tmp_path):
        from src.evaluation import plots as P
        out = tmp_path / "cmp.png"
        P.plot_method_comparison(
            {"Sezgisel": {"Precision": 0.8, "F1": 0.75},
             "Derin": {"Precision": 0.9, "F1": 0.88}},
            ["Precision", "F1"], out)
        assert out.exists() and out.stat().st_size > 0


class TestSyntheticDataset:
    """Sentetik park sahnesi üreteci ve değerlendirmesi."""

    def test_scene_shapes_and_keys(self):
        from src.evaluation import synthetic
        rng = np.random.default_rng(0)
        frame, dets, gt = synthetic.make_scene(rng, n_slots=8, occupancy_prob=0.5)
        assert frame.ndim == 3
        for d in dets:
            assert set(("bbox", "class_id", "class_name", "confidence")) <= set(d)

    def test_scene_ends_always_occupied(self):
        """İlk ve son slot daima dolu → kenar slotları GT'yi kirletmez."""
        from src.evaluation import synthetic
        rng = np.random.default_rng(1)
        frame, dets, gt = synthetic.make_scene(rng, n_slots=8, occupancy_prob=0.0)
        # occupancy_prob=0 olsa bile en az 2 araç (iki uç) olmalı
        assert len(dets) >= 2

    def test_scene_cars_are_landscape(self):
        """Üretilen araçlar yatay (en > boy) olmalı — paralel mod filtresi için."""
        from src.evaluation import synthetic
        rng = np.random.default_rng(2)
        _, dets, _ = synthetic.make_scene(rng, n_slots=8, occupancy_prob=1.0)
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            assert (x2 - x1) > (y2 - y1)

    def test_dataset_size(self):
        from src.evaluation import synthetic
        scenes = synthetic.make_dataset(n_scenes=30, seed=5)
        assert len(scenes) == 30

    def test_dataset_deterministic(self):
        from src.evaluation import synthetic
        a = synthetic.make_dataset(n_scenes=10, seed=7)
        b = synthetic.make_dataset(n_scenes=10, seed=7)
        assert len(a[0][1]) == len(b[0][1])  # aynı seed → aynı tespit sayısı

    def test_synthetic_evaluation_runs_and_scores(self, tmp_path):
        """100+ sahnede değerlendirme çalışır ve makul F1 üretir."""
        from src.evaluation.runner import run_synthetic_evaluation
        dm, cm = run_synthetic_evaluation(
            n_scenes=40, seed=42, out_dir=str(tmp_path), iou_threshold=0.4)
        assert (tmp_path / "detection_metrics.csv").exists()
        assert (tmp_path / "confusion_matrix.png").exists()
        # Algoritma sentetik sahnelerde anlamlı performans göstermeli
        assert dm.f1 > 0.7
        assert dm.precision > 0.7


class TestVideoTools:
    """Videodan kare çıkarma (offline değerlendirme için)."""

    def _make_video(self, path, n=30, w=120, h=80):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
        for i in range(n):
            frame = np.full((h, w, 3), i * 5 % 255, np.uint8)
            vw.write(frame)
        vw.release()

    def test_extract_frames_count(self, tmp_path):
        from src.evaluation.video_tools import extract_frames
        vid = tmp_path / "v.mp4"
        self._make_video(vid, n=30)
        if not vid.exists() or vid.stat().st_size == 0:
            pytest.skip("VideoWriter codec yok")
        out = extract_frames(str(vid), str(tmp_path / "frames"), count=5)
        # codec mevcutsa ~5 kare çıkmalı
        assert len(out) >= 1
        for p in out:
            assert Path(p).exists()

    def test_extract_frames_missing_video(self, tmp_path):
        from src.evaluation.video_tools import extract_frames
        out = extract_frames("/nonexistent/video.mp4", str(tmp_path), count=5)
        assert out == []


class TestPKLotIngest:
    """PKLot XML ayrıştırma ve graceful degrade."""

    _XML = """<parking id="lot">
      <space id="1" occupied="0">
        <contour>
          <point x="10" y="20"/><point x="60" y="20"/>
          <point x="60" y="50"/><point x="10" y="50"/>
        </contour>
      </space>
      <space id="2" occupied="1">
        <contour>
          <point x="100" y="20"/><point x="150" y="20"/>
          <point x="150" y="50"/><point x="100" y="50"/>
        </contour>
      </space>
    </parking>"""

    def test_parse_counts_and_bbox(self):
        from src.evaluation.datasets import parse_pklot_xml
        spaces = parse_pklot_xml(self._XML, is_string=True)
        assert len(spaces) == 2
        assert spaces[0]["occupied"] is False
        assert spaces[0]["bbox"] == (10, 20, 60, 50)
        assert spaces[1]["occupied"] is True

    def test_parse_rotatedrect_fallback(self):
        from src.evaluation.datasets import parse_pklot_xml
        xml = ('<parking><space id="1" occupied="0"><rotatedRect>'
               '<center x="50" y="50"/><size w="40" h="20"/><angle d="0"/>'
               '</rotatedRect></space></parking>')
        spaces = parse_pklot_xml(xml, is_string=True)
        assert spaces[0]["bbox"] == (30, 40, 70, 60)

    def test_ingest_missing_dataset_returns_empty(self):
        from src.evaluation.datasets import ingest_pklot
        gt = ingest_pklot("/nonexistent/pklot/root")
        assert gt == {}


class TestAblation:
    """Yöntem karşılaştırması, ablation ve duyarlılık analizi."""

    def _scenes(self, n=25):
        from src.evaluation import synthetic
        return synthetic.make_dataset(n_scenes=n, seed=11)

    def test_method_comparison_produces_all_configs(self, tmp_path):
        from src.evaluation.ablation import run_method_comparison
        res = run_method_comparison(self._scenes(), tmp_path)
        assert set(res.keys()) == {"Baseline", "Agresif", "Muhafazakar"}
        assert (tmp_path / "method_comparison.csv").exists()
        assert (tmp_path / "method_comparison.png").exists()

    def test_baseline_beats_aggressive_precision(self, tmp_path):
        """Baseline precision'ı, agresif konfigürasyondan yüksek olmalı."""
        from src.evaluation.ablation import run_method_comparison
        res = run_method_comparison(self._scenes(), tmp_path)
        assert res["Baseline"]["Precision"] >= res["Agresif"]["Precision"]

    def test_ablation_runs(self, tmp_path):
        from src.evaluation.ablation import run_ablation
        res = run_ablation(self._scenes(), tmp_path)
        assert len(res) == 3
        assert (tmp_path / "ablation.png").exists()

    def test_sensitivity_sweep(self, tmp_path):
        from src.evaluation.ablation import run_sensitivity
        rows = run_sensitivity(self._scenes(), "min_gap_ratio",
                               [0.2, 0.4, 0.6], tmp_path,
                               base_config={"max_edge_extension_ratio": 0.0})
        assert len(rows) == 3
        assert (tmp_path / "sensitivity_min_gap_ratio.png").exists()
        # her satır (deger, precision, recall, f1)
        for v, p, r, f in rows:
            assert 0.0 <= f <= 1.0


class TestFineTuningPipeline:
    """Fine-tuning veri dönüşümü ve config üretimi (eğitim hariç)."""

    def test_bbox_to_yolo_normalization(self):
        from src.training.finetune import _bbox_to_yolo
        cx, cy, w, h = _bbox_to_yolo((0, 0, 100, 100), 200, 200)
        assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)

    def test_write_yolo_labels(self, tmp_path):
        from src.training.finetune import write_yolo_labels
        slots = [{"bbox": [10, 10, 50, 50], "occupied": False},
                 {"bbox": [60, 60, 100, 100], "occupied": True}]
        out = tmp_path / "lbl.txt"
        n = write_yolo_labels(slots, 200, 200, out)
        assert n == 2
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("0 ")  # empty → class 0
        assert lines[1].startswith("1 ")  # occupied → class 1

    def test_make_data_yaml_contents(self):
        from src.training.finetune import make_data_yaml
        content = make_data_yaml("data/pklot_yolo")
        assert "space-empty" in content
        assert "space-occupied" in content
        assert "nc: 2" in content

    def test_pklot_to_yolo_split(self, tmp_path):
        from src.training.finetune import pklot_to_yolo
        # küçük sahte görüntü + ground truth
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        gt = {}
        for i in range(10):
            p = img_dir / f"img{i}.jpg"
            cv2.imwrite(str(p), np.full((100, 100, 3), 100, np.uint8))
            gt[f"img{i}.jpg"] = {"slots": [
                {"bbox": [10, 10, 40, 40], "occupied": i % 2 == 0}]}
        counts = pklot_to_yolo(gt, str(img_dir), str(tmp_path / "yolo"),
                               val_split=0.2, seed=1)
        assert counts["train"] + counts["val"] == 10
        assert counts["val"] == 2
        assert (tmp_path / "yolo" / "labels" / "train").exists()


class TestIPM:
    """Inverse Perspective Mapping (kuş bakışı) dönüşümü."""

    def _square_tf(self, real_w=10.0, real_h=10.0):
        from src.geometry import PerspectiveTransformer
        # kaynakta hafif trapez → kuş bakışında 200x200 kare
        src = [(50, 50), (150, 50), (170, 150), (30, 150)]
        return PerspectiveTransformer.from_quad(
            src, 200, 200, real_w_m=real_w, real_h_m=real_h)

    def test_from_quad_builds_homography(self):
        tf = self._square_tf()
        assert tf.H is not None and tf.H.shape == (3, 3)
        assert tf.out_size == (200, 200)

    def test_from_quad_rejects_wrong_point_count(self):
        from src.geometry import PerspectiveTransformer
        with pytest.raises(ValueError):
            PerspectiveTransformer.from_quad([(0, 0), (1, 1)], 100, 100)

    def test_corner_maps_to_origin(self):
        tf = self._square_tf()
        # kaynak sol-üst köşe → kuş bakışı (0,0)
        out = tf.transform_points([(50, 50)])
        assert abs(out[0, 0]) < 1e-3 and abs(out[0, 1]) < 1e-3

    def test_warp_image_output_size(self):
        tf = self._square_tf()
        frame = np.full((200, 200, 3), 120, np.uint8)
        warped = tf.warp_image(frame)
        assert warped.shape[:2] == (200, 200)

    def test_m_per_px_calibration(self):
        tf = self._square_tf(real_w=10.0, real_h=10.0)
        # 10m / 200px = 0.05 m/px
        assert abs(tf.m_per_px - 0.05) < 1e-6

    def test_measure_distance_metric(self):
        tf = self._square_tf(real_w=10.0, real_h=10.0)
        # kaynak sol-üst (50,50) ve sağ-üst (150,50) → kuş bakışında 200px = 10m
        d = tf.measure_distance_m((50, 50), (150, 50))
        assert abs(d - 10.0) < 0.1

    def test_box_size_metric(self):
        tf = self._square_tf(real_w=10.0, real_h=10.0)
        size = tf.box_size_m((50, 50, 150, 150))
        assert size is not None
        w_m, h_m = size
        assert w_m > 0 and h_m > 0

    def test_measure_without_calibration_returns_none(self):
        from src.geometry import PerspectiveTransformer
        tf = PerspectiveTransformer.from_quad(
            [(0, 0), (10, 0), (10, 10), (0, 10)], 100, 100)
        assert tf.measure_distance_m((0, 0), (5, 5)) is None


class TestParkingLineDetector:
    """Park çizgisi tespiti ve ızgara çıkarımı (kuş bakışı)."""

    def _grid_image(self, w=400, h=200, xs=(40, 120, 200, 280, 360),
                    ys=(30, 170)):
        """Koyu zemin üzerine beyaz dikey/yatay şeritler çiz (BEV taklidi)."""
        img = np.full((h, w, 3), 40, np.uint8)
        for x in xs:
            cv2.line(img, (x, 10), (x, h - 10), (255, 255, 255), 3)
        for y in ys:
            cv2.line(img, (20, y), (w - 20, y), (255, 255, 255), 3)
        return img

    def test_detects_vertical_grid_lines(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        xs, ys = det.grid_lines(self._grid_image())
        # 5 dikey şerit bekleniyor (tolerans dahilinde)
        assert len(xs) >= 4

    def test_has_lines_true_on_grid(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        assert det.has_lines(self._grid_image()) is True

    def test_has_lines_false_on_blank(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        blank = np.full((200, 400, 3), 70, np.uint8)
        assert det.has_lines(blank) is False

    def test_build_slots_count(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        slots = det.build_slots(self._grid_image())
        # 5 dikey çizgi → ~4 hücre
        assert 3 <= len(slots) <= 4

    def test_classify_slots_occupied_empty(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        slots = [(40, 30, 120, 170), (120, 30, 200, 170)]
        # ilk slotu kaplayan bir araç
        vehicles = [(45, 40, 115, 160)]
        res = ParkingLineDetector.classify_slots(slots, vehicles)
        assert res[0]["occupied"] is True
        assert res[1]["occupied"] is False

    def test_cluster_positions_merges_close(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        merged = ParkingLineDetector._cluster_positions([10, 11, 12, 100, 101], 5)
        assert len(merged) == 2

    def test_detects_yellow_lines_on_gray_asphalt(self):
        """Gri asfalt üzerine SARI şeritler — renk maskesi yakalamalı."""
        from src.detection.parking_line_detector import ParkingLineDetector
        # gri zemin (beyaz eşik bunu yakalamaz), sarı dikey şeritler
        img = np.full((200, 400, 3), 120, np.uint8)
        for x in (40, 120, 200, 280, 360):
            cv2.line(img, (x, 10), (x, 190), (0, 200, 230), 3)  # BGR sarı
        det = ParkingLineDetector()
        xs, ys = det.grid_lines(img)
        assert len(xs) >= 4

    def test_color_mask_can_be_disabled(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector(use_color=False)
        img = np.full((200, 400, 3), 120, np.uint8)
        for x in (40, 120, 200, 280, 360):
            cv2.line(img, (x, 10), (x, 190), (0, 200, 230), 3)
        # renk kapalı: gri sarı düşük kontrast → daha az/zayıf tespit
        mask = det._edge_mask(img)
        assert mask.shape == (200, 400)

    def test_refine_position_snaps_to_peak(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        profile = np.zeros(100, np.float64)
        profile[50] = 10.0  # tepe 50'de
        # kaba tahmin 47 → tepeye yaklaşmalı
        refined = ParkingLineDetector._refine_position(47, profile, window=8)
        assert abs(refined - 50) < 1.0

    def test_refine_position_empty_window_keeps_pos(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        profile = np.zeros(100, np.float64)
        refined = ParkingLineDetector._refine_position(30, profile, window=5)
        assert refined == 30

    def test_grid_lines_refine_flag_runs(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        img = self._grid_image()
        xs_r, _ = det.grid_lines(img, refine=True)
        xs_n, _ = det.grid_lines(img, refine=False)
        assert len(xs_r) >= 4 and len(xs_n) >= 4


class TestAdaptiveSlotDetector:
    """Adaptif seçici: çizgi varsa ızgara, yoksa geometri."""

    def _grid_image(self, w=400, h=200):
        img = np.full((h, w, 3), 40, np.uint8)
        for x in (40, 120, 200, 280, 360):
            cv2.line(img, (x, 10), (x, h - 10), (255, 255, 255), 3)
        for y in (30, 170):
            cv2.line(img, (20, y), (w - 20, y), (255, 255, 255), 3)
        return img

    def test_uses_line_method_when_lines_present(self):
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        det = AdaptiveSlotDetector()
        img = self._grid_image()
        # bir slotu dolduran araç
        dets = [{"bbox": [45, 40, 115, 160], "class_id": 2,
                 "class_name": "car", "confidence": 0.9}]
        res = det.analyze(img, dets)
        assert res["method"] == "line"
        assert res["occupied_count"] >= 1
        assert res["empty_count"] >= 1

    def test_falls_back_to_geometry_without_lines(self):
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        from src.detection.street_parking_detector import StreetParkingDetector
        street = StreetParkingDetector(smoothing_frames=1, smoothing_min_hits=1,
                                       road_color_check=False)
        det = AdaptiveSlotDetector(street_detector=street)
        blank = np.full((300, 800, 3), 70, np.uint8)
        dets = [_det(50, 150, 150, 250), _det(650, 150, 750, 250)]
        res = det.analyze(blank, dets)
        assert res["method"] == "geometry"
        assert "empty_spaces" in res

    def test_result_has_polys(self):
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        det = AdaptiveSlotDetector()
        res = det.analyze(self._grid_image(), [])
        assert "empty_polys" in res and "occupied_polys" in res
        # tüm slotlar boş (araç yok)
        assert res["occupied_count"] == 0

    def test_line_slot_over_vehicle_marked_occupied(self):
        """Izgara slotu bir aracın üstüne denk gelirse boş değil dolu sayılmalı."""
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        det = AdaptiveSlotDetector(use_voting=False, use_fusion=False)
        img = self._grid_image()  # 400x200, dikey şeritler 40..360
        # İlk slot bandını (x≈40-120) kaplayan bir araç (raw_bbox tam kutu).
        # classify YOLO bbox'ı görmese de _pack_line_result dolu işaretlemeli.
        dets = [{"bbox": [45, 40, 115, 165], "raw_bbox": [45, 40, 115, 165],
                 "class_id": 2, "class_name": "car", "confidence": 0.9}]
        res = det.analyze(img, dets)
        assert res["method"] == "line"
        assert res["occupied_count"] >= 1

    def test_line_method_with_ipm_roundtrip(self):
        """IPM verildiğinde kuş bakışında çizgi bulup kaynağa geri haritalar."""
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        from src.geometry import PerspectiveTransformer
        det = AdaptiveSlotDetector()
        grid = self._grid_image(400, 200)
        # birim-benzeri IPM: kaynak köşeleri → aynı boyut BEV (kimlik yakın)
        ipm = PerspectiveTransformer.from_quad(
            [(0, 0), (399, 0), (399, 199), (0, 199)], 400, 200,
            real_w_m=20.0, real_h_m=10.0)
        res = det.analyze(grid, [], ipm=ipm)
        assert res["method"] == "line"
        # geri haritalanan poligonlar kaynak sınırları içinde
        for poly in res["empty_polys"]:
            assert poly.shape == (4, 2)

    def test_ipm_result_has_metric_slot_sizes(self):
        """IPM kalibreyse boş slotların metrik boyutu hesaplanmalı (sığma için)."""
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        from src.geometry import PerspectiveTransformer
        det = AdaptiveSlotDetector()
        grid = self._grid_image(400, 200)
        ipm = PerspectiveTransformer.from_quad(
            [(0, 0), (399, 0), (399, 199), (0, 199)], 400, 200,
            real_w_m=20.0, real_h_m=10.0)
        res = det.analyze(grid, [], ipm=ipm)
        assert "empty_sizes_m" in res
        assert len(res["empty_sizes_m"]) == len(res["empty_polys"])
        if res["empty_sizes_m"]:
            w_m, h_m = res["empty_sizes_m"][0]
            assert w_m > 0 and h_m > 0

    def test_geometry_result_has_empty_sizes_key(self):
        from src.detection.adaptive_slot_detector import AdaptiveSlotDetector
        from src.detection.street_parking_detector import StreetParkingDetector
        street = StreetParkingDetector(smoothing_frames=1, smoothing_min_hits=1,
                                       road_color_check=False)
        det = AdaptiveSlotDetector(street_detector=street)
        blank = np.full((300, 800, 3), 70, np.uint8)
        res = det.analyze(blank, [_det(50, 150, 150, 250), _det(650, 150, 750, 250)])
        assert "empty_sizes_m" in res


class TestIPMInverse:
    """IPM ters dönüşüm (kuş bakışı → kaynak)."""

    def _tf(self):
        from src.geometry import PerspectiveTransformer
        return PerspectiveTransformer.from_quad(
            [(50, 50), (150, 50), (170, 150), (30, 150)], 200, 200, 10, 10)

    def test_inverse_roundtrip(self):
        tf = self._tf()
        src_pt = [(80, 70)]
        bev = tf.transform_points(src_pt)
        back = tf.inverse_transform_points(bev)
        assert abs(back[0, 0] - 80) < 1e-2 and abs(back[0, 1] - 70) < 1e-2

    def test_inverse_quad_shape(self):
        tf = self._tf()
        quad = tf.inverse_transform_quad((0, 0, 100, 100))
        assert quad.shape == (4, 2)


class TestAutoIPM:
    """Otomatik IPM — yakınsayan çizgilerden homografi."""

    def test_vanishing_point_estimate(self):
        from src.geometry.auto_ipm import estimate_vanishing_point, _line_from_seg
        # iki çizgi (10,0)-(50,100) ve (90,0)-(50,100) → (50,100)'de kesişir
        l1 = _line_from_seg((10, 0, 50, 100))
        l2 = _line_from_seg((90, 0, 50, 100))
        vp = estimate_vanishing_point([l1, l2])
        assert abs(vp[0] - 50) < 1.0 and abs(vp[1] - 100) < 1.0

    def test_auto_calibrate_on_converging_lines(self):
        from src.geometry.auto_ipm import auto_calibrate
        img = np.full((240, 600, 3), 40, np.uint8)
        # yukarı yakınsayan iki kenar (trapez) + iç çizgiler
        cv2.line(img, (100, 235), (250, 90), (255, 255, 255), 3)
        cv2.line(img, (500, 235), (350, 90), (255, 255, 255), 3)
        cv2.line(img, (250, 235), (290, 90), (255, 255, 255), 3)
        cv2.line(img, (350, 235), (320, 90), (255, 255, 255), 3)
        tf = auto_calibrate(img, out_w=400, out_h=600,
                            real_w_m=10.0, real_h_m=15.0)
        assert tf is not None
        assert tf.H is not None and tf.m_per_px is not None

    def test_auto_calibrate_fails_gracefully_on_blank(self):
        from src.geometry.auto_ipm import auto_calibrate
        blank = np.full((240, 600, 3), 50, np.uint8)
        assert auto_calibrate(blank) is None

    def test_vehicle_ipm_needs_depth_spread(self):
        from src.geometry.auto_ipm import auto_calibrate_from_vehicles
        # tüm araçlar aynı derinlikte (bottom-y aynı) → derinlik yok → None
        dets = [{"bbox": [50, 100, 110, 200], "class_id": 2},
                {"bbox": [200, 100, 260, 200], "class_id": 2},
                {"bbox": [350, 100, 410, 200], "class_id": 2}]
        assert auto_calibrate_from_vehicles(dets, (300, 600, 3)) is None

    def test_vehicle_ipm_builds_with_receding_row(self):
        from src.geometry.auto_ipm import auto_calibrate_from_vehicles
        # uzaktaki araçlar daha küçük ve yukarıda (perspektif sıra)
        dets = [
            {"bbox": [60, 230, 180, 290], "class_id": 2},   # yakın, geniş
            {"bbox": [420, 230, 540, 290], "class_id": 2},  # yakın sağ
            {"bbox": [250, 80, 300, 120], "class_id": 2},   # uzak, dar
            {"bbox": [310, 80, 360, 120], "class_id": 2},   # uzak sağ
        ]
        tf = auto_calibrate_from_vehicles(dets, (300, 600, 3),
                                          out_w=400, out_h=600,
                                          real_w_m=10.0, real_h_m=15.0,
                                          min_vehicles=3)
        assert tf is not None and tf.H is not None


class TestSlotScoring:
    """A4 — çok kriterli slot seçim motoru."""

    def test_difficulty_wider_is_easier(self):
        from src.parking.slot_scoring import compute_difficulty
        dar = compute_difficulty(1.9, 1.8)
        genis = compute_difficulty(3.0, 1.8)
        assert genis > dar
        assert 0 <= dar <= 100 and 0 <= genis <= 100

    def test_difficulty_unknown_is_neutral(self):
        from src.parking.slot_scoring import compute_difficulty
        assert compute_difficulty(None, 1.8) == 50

    def test_score_range(self):
        from src.parking.slot_scoring import compute_slot_score
        s = compute_slot_score(80, 200, 2.6, 1.8, 500, 1000)
        assert 0 <= s <= 100

    def test_score_prefers_closer_and_wider(self):
        from src.parking.slot_scoring import compute_slot_score
        near_wide = compute_slot_score(90, 100, 3.0, 1.8, 900, 1000)
        far_narrow = compute_slot_score(40, 900, 1.85, 1.8, 100, 1000)
        assert near_wide > far_narrow

    def test_reason_text_contains_parts(self):
        from src.parking.slot_scoring import slot_reason_text
        txt = slot_reason_text(80, 0.4, 200, 0.8)
        assert "Kolay manevra" in txt and "Genis slot" in txt

    def test_recommend_best_picks_highest(self):
        from src.parking.slot_scoring import recommend_best_slot
        slots = [
            {"cx": 100, "cy": 500, "width_m": 1.85},   # dar, uzak
            {"cx": 900, "cy": 520, "width_m": 3.0},    # geniş, yakın, sağda
        ]
        best = recommend_best_slot(slots, ref_width_m=1.8, map_width=1000,
                                   origin=(950, 540))
        assert best is not None and best["index"] == 1
        assert 0 <= best["score"] <= 100

    def test_recommend_empty_returns_none(self):
        from src.parking.slot_scoring import recommend_best_slot
        assert recommend_best_slot([], 1.8, 1000, (0, 0)) is None


class TestVehicleDedupe:
    """YOLO çift-tespit (sınıf-bağımsız NMS) dedup."""

    def test_dedupe_removes_overlapping_lower_conf(self):
        from src.detection.vehicle_detector import VehicleDetector
        dets = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 2, "class_name": "car"},
            {"bbox": [105, 102, 205, 198], "confidence": 0.6, "class_id": 7, "class_name": "truck"},
        ]
        out = VehicleDetector._dedupe(dets, iou_thresh=0.6)
        assert len(out) == 1
        assert out[0]["confidence"] == 0.9  # yüksek güvenli kaldı

    def test_dedupe_keeps_distinct_vehicles(self):
        from src.detection.vehicle_detector import VehicleDetector
        dets = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 2, "class_name": "car"},
            {"bbox": [400, 100, 500, 200], "confidence": 0.8, "class_id": 2, "class_name": "car"},
        ]
        out = VehicleDetector._dedupe(dets, iou_thresh=0.6)
        assert len(out) == 2  # ayrı araçlar korunur

    def test_dedupe_empty(self):
        from src.detection.vehicle_detector import VehicleDetector
        assert VehicleDetector._dedupe([]) == []


class TestVehicleOverlapReject:
    """Park aracı boş alan olarak sayılmasın (araç-örtüşme reddi)."""

    def test_slot_center_inside_vehicle_blocked(self):
        from src.detection.street_parking_detector import StreetParkingDetector
        slot = (100, 100, 160, 220)
        veh = [(80, 90, 180, 240)]  # slot merkezi araç içinde
        assert StreetParkingDetector._slot_blocked_by_vehicle(slot, veh) is True

    def test_slot_far_from_vehicle_not_blocked(self):
        from src.detection.street_parking_detector import StreetParkingDetector
        slot = (10, 10, 60, 120)
        veh = [(300, 300, 400, 420)]
        assert StreetParkingDetector._slot_blocked_by_vehicle(slot, veh) is False

    def test_partial_overlap_blocked(self):
        from src.detection.street_parking_detector import StreetParkingDetector
        # slot ile araç %20+ örtüşüyor
        slot = (100, 100, 200, 300)
        veh = [(150, 100, 260, 300)]
        assert StreetParkingDetector._slot_blocked_by_vehicle(slot, veh) is True

    def test_no_vehicles_not_blocked(self):
        from src.detection.street_parking_detector import StreetParkingDetector
        assert StreetParkingDetector._slot_blocked_by_vehicle((0, 0, 10, 10), []) is False

    def test_degenerate_slot_blocked(self):
        from src.detection.street_parking_detector import StreetParkingDetector
        assert StreetParkingDetector._slot_blocked_by_vehicle((5, 5, 5, 5), []) is True


class TestVoiceAssistant:
    """C1 — sesli asistan komut ayrıştırıcı + graceful degrade."""

    def test_match_find_empty(self):
        from src.voice import match_command
        assert match_command("en yakin bos yeri bul") == "find_empty"

    def test_match_depth(self):
        from src.voice import match_command
        assert match_command("derinlik haritasi ac") == "toggle_depth"

    def test_match_stop(self):
        from src.voice import match_command
        assert match_command("simdi dur") == "stop"

    def test_match_unknown_returns_none(self):
        from src.voice import match_command
        assert match_command("hava bugun guzel") is None

    def test_match_empty_text(self):
        from src.voice import match_command
        assert match_command("") is None
        assert match_command(None) is None

    def test_assistant_unavailable_without_model(self):
        from src.voice import VoiceAssistant
        va = VoiceAssistant("/nonexistent/model/path", callback=lambda c, t: None)
        assert va.available is False
        assert va.start() is False  # yüklenemediğinde başlamaz


class TestAutoROI:
    """Tespitlerden otomatik ROI üretimi."""

    def test_auto_roi_covers_detections(self):
        from src.geometry import roi
        dets = [{"bbox": [100, 100, 160, 180], "class_id": 2},
                {"bbox": [300, 120, 360, 200], "class_id": 2}]
        poly = roi.auto_roi_from_detections(dets, (400, 500, 3))
        assert poly is not None and len(poly) >= 3
        # araç merkezleri ROI içinde olmalı
        assert roi.point_in_polygon((130, 140), poly)
        assert roi.point_in_polygon((330, 160), poly)

    def test_auto_roi_none_without_detections(self):
        from src.geometry import roi
        assert roi.auto_roi_from_detections([], (400, 500, 3)) is None

    def test_auto_roi_points_within_frame(self):
        from src.geometry import roi
        dets = [{"bbox": [5, 5, 60, 60], "class_id": 2},
                {"bbox": [450, 350, 495, 395], "class_id": 2}]
        poly = roi.auto_roi_from_detections(dets, (400, 500, 3), margin_frac=0.2)
        for x, y in poly:
            assert 0 <= x <= 499 and 0 <= y <= 399


class TestROI:
    """İlgi bölgesi (ROI) maskeleme yardımcıları."""

    _POLY = [(50, 50), (250, 50), (250, 200), (50, 200)]  # dikdörtgen

    def test_point_inside(self):
        from src.geometry import roi
        assert roi.point_in_polygon((150, 120), self._POLY) is True

    def test_point_outside(self):
        from src.geometry import roi
        assert roi.point_in_polygon((10, 10), self._POLY) is False

    def test_filter_detections(self):
        from src.geometry import roi
        dets = [
            {"bbox": [100, 80, 180, 160], "class_id": 2},   # merkez içeride
            {"bbox": [300, 300, 360, 360], "class_id": 2},  # dışarıda
        ]
        kept = roi.filter_detections(dets, self._POLY)
        assert len(kept) == 1
        assert kept[0]["bbox"][0] == 100

    def test_filter_none_polygon_keeps_all(self):
        from src.geometry import roi
        dets = [{"bbox": [0, 0, 10, 10], "class_id": 2}]
        assert len(roi.filter_detections(dets, None)) == 1

    def test_draw_roi_shape_and_dims(self):
        from src.geometry import roi
        img = np.full((240, 320, 3), 200, np.uint8)
        out = roi.draw_roi(img, self._POLY, dim=0.4)
        assert out.shape == img.shape
        # dışarısı kararmış olmalı (köşe), içerisi parlak kalmalı (merkez)
        assert out[5, 5].mean() < img[5, 5].mean()
        assert out[120, 150].mean() >= img[120, 150].mean() - 1


class TestVideoStabilizer:
    """Video sabitleme — elde-çekim kayması kompanzasyonu."""

    def _textured(self, w=320, h=240):
        rng = np.random.default_rng(3)
        img = np.full((h, w, 3), 60, np.uint8)
        for _ in range(40):
            x, y = int(rng.integers(10, w - 30)), int(rng.integers(10, h - 30))
            c = tuple(int(v) for v in rng.integers(80, 255, 3))
            cv2.rectangle(img, (x, y), (x + 20, y + 16), c, -1)
        return img

    def test_first_frame_sets_reference(self):
        from src.detection.video_stabilizer import VideoStabilizer
        s = VideoStabilizer()
        ref = self._textured()
        out, ok = s.stabilize(ref)
        assert ok is False  # ilk kare referans olur
        assert s.ref_des is not None

    def test_compensates_translation(self):
        from src.detection.video_stabilizer import VideoStabilizer
        s = VideoStabilizer()
        ref = self._textured()
        s.stabilize(ref)  # referans
        M = np.float32([[1, 0, 12], [0, 1, 8]])
        shifted = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
        out, ok = s.stabilize(shifted)
        assert ok is True
        # hizalanmış kare, kaymış girdiye göre referansa daha yakın olmalı
        d_before = float(np.mean(np.abs(shifted.astype(int) - ref.astype(int))))
        d_after = float(np.mean(np.abs(out.astype(int) - ref.astype(int))))
        assert d_after < d_before

    def test_graceful_on_blank(self):
        from src.detection.video_stabilizer import VideoStabilizer
        s = VideoStabilizer()
        blank = np.zeros((240, 320, 3), np.uint8)
        out, ok = s.stabilize(blank)
        # özellik yok → referans kurulamaz, kare olduğu gibi döner
        assert ok is False
        assert out.shape == blank.shape

    def test_reset(self):
        from src.detection.video_stabilizer import VideoStabilizer
        s = VideoStabilizer()
        s.stabilize(self._textured())
        s.reset()
        assert s.ref_des is None


class TestGridFusion:
    """Kalıcı ızgara füzyonu — çizgi geometrisi kararlılığı."""

    def test_persists_missing_line(self):
        from src.detection.grid_fusion import GridLineFusion
        f = GridLineFusion(match_tol=14, min_hits=2, max_miss=12)
        for _ in range(3):
            xs, ys = f.update([40, 120, 200], [])
        # 120 bir karede kaybolsa bile harita onu korumalı
        xs, ys = f.update([40, 200], [])
        assert any(abs(x - 120) < 14 for x in xs)

    def test_ema_smoothing(self):
        from src.detection.grid_fusion import GridLineFusion
        f = GridLineFusion(match_tol=14, alpha=0.4, min_hits=1)
        f.update([40], [])
        xs, _ = f.update([44], [])  # 0.4*44 + 0.6*40 = 41.6
        assert any(abs(x - 41.6) < 0.5 for x in xs)

    def test_new_line_needs_min_hits(self):
        from src.detection.grid_fusion import GridLineFusion
        f = GridLineFusion(min_hits=2)
        xs, _ = f.update([100], [])
        assert xs == []  # tek görülme → henüz güvenilir değil
        xs, _ = f.update([100], [])
        assert any(abs(x - 100) < 5 for x in xs)  # ikinci → güvenilir

    def test_reset(self):
        from src.detection.grid_fusion import GridLineFusion
        f = GridLineFusion(min_hits=1)
        f.update([40, 80], [])
        f.reset()
        xs, ys = f.update([], [])
        assert xs == [] and ys == []

    def test_build_slots_from_positions(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        det = ParkingLineDetector()
        slots = det.build_slots_from_positions(
            [40, 120, 200, 280], [30, 170], (200, 400, 3))
        assert len(slots) == 3


class TestTemporalVoter:
    """Zamansal slot oylama — titreme yumuşatma."""

    def test_single_flip_is_smoothed(self):
        from src.detection.temporal_voter import TemporalSlotVoter
        v = TemporalSlotVoter(history=7, stable_min=3)
        slot = {"bbox": (0, 0, 50, 100), "occupied": False}
        # 5 kare boş, 1 kare yanlışlıkla dolu → çoğunluk hâlâ boş
        for _ in range(5):
            out = v.update([slot])
        flip = v.update([{"bbox": (0, 0, 50, 100), "occupied": True}])
        assert flip[0]["occupied"] is False  # tek gürültü bastırıldı

    def test_persistent_change_eventually_flips(self):
        from src.detection.temporal_voter import TemporalSlotVoter
        v = TemporalSlotVoter(history=5)
        slot_empty = {"bbox": (0, 0, 50, 100), "occupied": False}
        for _ in range(5):
            v.update([slot_empty])
        # sürekli dolu gelince çoğunluk dolar
        for _ in range(5):
            out = v.update([{"bbox": (0, 0, 50, 100), "occupied": True}])
        assert out[0]["occupied"] is True

    def test_stable_flag(self):
        from src.detection.temporal_voter import TemporalSlotVoter
        v = TemporalSlotVoter(history=7, stable_min=3)
        slot = {"bbox": (0, 0, 50, 100), "occupied": True}
        out = v.update([slot])
        assert out[0]["stable"] is False  # tek kare → kararsız
        for _ in range(3):
            out = v.update([slot])
        assert out[0]["stable"] is True

    def test_reset_clears_tracks(self):
        from src.detection.temporal_voter import TemporalSlotVoter
        v = TemporalSlotVoter()
        v.update([{"bbox": (0, 0, 10, 10), "occupied": True}])
        v.reset()
        assert v.tracks == []


class TestOverlays:
    """Demo görselleştirme yardımcıları (saf cv2)."""

    def test_nearest_empty_picks_closest(self):
        from src.ui import overlays as ov
        polys = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float),     # uzak
            np.array([[90, 90], [100, 90], [100, 100], [90, 100]], float),  # yakın
        ]
        origin = (95, 95)
        idx, centroid, dist = ov.nearest_empty(polys, origin)
        assert idx == 1
        assert dist < 10

    def test_nearest_empty_empty_list(self):
        from src.ui import overlays as ov
        assert ov.nearest_empty([], (0, 0)) is None

    def test_draw_guidance_no_crash(self):
        from src.ui import overlays as ov
        img = np.zeros((120, 120, 3), np.uint8)
        out = ov.draw_guidance(img, (60, 110), (30, 30), "TEST")
        assert out.shape == (120, 120, 3)

    def test_draw_pseudo_3d_no_crash(self):
        from src.ui import overlays as ov
        img = np.zeros((120, 120, 3), np.uint8)
        poly = np.array([[20, 60], [60, 60], [60, 100], [20, 100]], float)
        out = ov.draw_pseudo_3d(img, poly, (0, 220, 80), lift=10)
        assert out.shape == (120, 120, 3)

    def test_render_minimap_shape(self):
        from src.ui import overlays as ov
        empty = [np.array([[10, 10], [40, 10], [40, 50], [10, 50]], float)]
        occ = [np.array([[60, 10], [90, 10], [90, 50], [60, 50]], float)]
        mm = ov.render_minimap(empty, occ, width=200, height=70)
        assert mm.shape == (70, 200, 3)

    def test_render_minimap_empty(self):
        from src.ui import overlays as ov
        mm = ov.render_minimap([], [], width=200, height=70)
        assert mm.shape == (70, 200, 3)

    def test_paste_minimap_in_bounds(self):
        from src.ui import overlays as ov
        out = np.zeros((300, 400, 3), np.uint8)
        mm = np.full((70, 200, 3), 100, np.uint8)
        ov.paste_minimap(out, mm)
        # sağ üst köşeye yapıştı → orada parlaklık arttı
        assert out[40, 380].mean() > 0


class TestGeometricConsistency:
    """Slot boyut tutarlılık filtresi."""

    def test_outlier_slot_removed(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        # 4 normal (~80 genişlik) + 1 aykırı (300 genişlik)
        slots = [(0, 0, 80, 100), (100, 0, 180, 100), (200, 0, 280, 100),
                 (300, 0, 380, 100), (400, 0, 700, 100)]
        kept = ParkingLineDetector.filter_by_size_consistency(slots)
        assert (400, 0, 700, 100) not in kept
        assert len(kept) == 4

    def test_few_slots_untouched(self):
        from src.detection.parking_line_detector import ParkingLineDetector
        slots = [(0, 0, 80, 100), (100, 0, 400, 100)]
        kept = ParkingLineDetector.filter_by_size_consistency(slots)
        assert len(kept) == 2  # 3'ten az → dokunulmaz


class TestDepthEstimator:
    """Monoküler derinlik tahmini — graceful degrade ve yardımcılar."""

    def test_unavailable_without_model(self):
        from src.detection.depth_estimator import DepthEstimator
        de = DepthEstimator(local_model=None, allow_download=False)
        assert de.available is False

    def test_infer_returns_none_when_unavailable(self):
        from src.detection.depth_estimator import DepthEstimator
        de = DepthEstimator(allow_download=False)
        assert de.infer(_blank(320, 240)) is None

    def test_infer_handles_none_frame(self):
        from src.detection.depth_estimator import DepthEstimator
        de = DepthEstimator(allow_download=False)
        assert de.infer(None) is None

    def test_normalize_range(self):
        from src.detection.depth_estimator import DepthEstimator
        d = np.array([[0.0, 5.0], [10.0, 2.5]], dtype=np.float32)
        n = DepthEstimator._normalize(d)
        assert n.min() == 0.0 and n.max() == pytest.approx(1.0)

    def test_depth_colormap_shape(self):
        from src.detection.depth_estimator import DepthEstimator
        d = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        cmap = DepthEstimator.depth_to_colormap(d)
        assert cmap.shape == (10, 10, 3)

    def test_depth_colormap_none(self):
        from src.detection.depth_estimator import DepthEstimator
        assert DepthEstimator.depth_to_colormap(None) is None

    def test_normalize_flat_map(self):
        from src.detection.depth_estimator import DepthEstimator
        d = np.full((4, 4), 3.0, np.float32)
        n = DepthEstimator._normalize(d)
        assert np.all(n == 0.0)

    def test_region_depth_median(self):
        from src.detection.depth_estimator import DepthEstimator
        dm = np.zeros((100, 100), np.float32)
        dm[10:50, 10:50] = 0.8
        val = DepthEstimator.region_depth(dm, (10, 10, 50, 50))
        assert val == pytest.approx(0.8)

    def test_region_depth_none_map(self):
        from src.detection.depth_estimator import DepthEstimator
        assert DepthEstimator.region_depth(None, (0, 0, 10, 10)) is None

    def test_same_plane_true(self):
        from src.detection.depth_estimator import DepthEstimator
        dm = np.full((100, 100), 0.5, np.float32)
        assert DepthEstimator.same_plane(dm, (0, 0, 10, 10), (50, 50, 60, 60)) is True

    def test_same_plane_false(self):
        from src.detection.depth_estimator import DepthEstimator
        dm = np.zeros((100, 100), np.float32)
        dm[:, 50:] = 1.0  # sağ yarı çok daha yakın
        result = DepthEstimator.same_plane(dm, (0, 0, 10, 10), (60, 0, 70, 10))
        assert result is False

    def test_same_plane_none_map(self):
        from src.detection.depth_estimator import DepthEstimator
        assert DepthEstimator.same_plane(None, (0, 0, 1, 1), (2, 2, 3, 3)) is None


class TestRayCastingOcclusion:
    """Ray-casting tabanlı görüş engeli filtresi birim testleri."""

    def test_line_segments_intersect_crosses(self):
        from src.geometry.roi import line_segments_intersect
        # Kesişen iki doğru parçası (+)
        assert line_segments_intersect((0, 5), (10, 5), (5, 0), (5, 10)) is True

    def test_line_segments_intersect_parallel(self):
        from src.geometry.roi import line_segments_intersect
        # Kesişmeyen paralel doğrular
        assert line_segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False

    def test_segment_intersects_bbox_inside(self):
        from src.geometry.roi import segment_intersects_bbox
        # Tamamen kutunun içinde olan doğru parçası
        bbox = (10, 10, 50, 50)
        assert segment_intersects_bbox((20, 20), (30, 30), bbox) is True

    def test_segment_intersects_bbox_crosses(self):
        from src.geometry.roi import segment_intersects_bbox
        # Kutuyu kesip geçen doğru parçası
        bbox = (10, 10, 50, 50)
        assert segment_intersects_bbox((0, 30), (60, 30), bbox) is True

    def test_segment_intersects_bbox_outside(self):
        from src.geometry.roi import segment_intersects_bbox
        # Kutunun dışındaki doğru parçası
        bbox = (10, 10, 50, 50)
        assert segment_intersects_bbox((0, 0), (5, 5), bbox) is False

    def test_filter_occluded_slots_blocked(self):
        from src.geometry.roi import filter_occluded_slots
        # Kamera (100, 200)'de. Boş slot (100, 50)'de.
        # Engelleyici araç (80, 100, 120, 150)'de.
        # Aracın alt sınırı 150 > 50 (slot y), yani araç slottan daha yakın ve görüş hattında.
        empty_spaces = [(90, 40, 110, 60)]
        vehicle_boxes = [(80, 100, 120, 150)]
        kept = filter_occluded_slots(empty_spaces, vehicle_boxes, (200, 200))
        assert kept == []  # Engellenmiş olmalı

    def test_filter_occluded_slots_clear(self):
        from src.geometry.roi import filter_occluded_slots
        # Kamera (100, 200)'de. Boş slot (100, 50)'de.
        # Araç yan tarafta (150, 100, 190, 150)'de, görüş hattı açık.
        empty_spaces = [(90, 40, 110, 60)]
        vehicle_boxes = [(150, 100, 190, 150)]
        kept = filter_occluded_slots(empty_spaces, vehicle_boxes, (200, 200))
        assert kept == [0]  # Görüşü açık kalmalı
