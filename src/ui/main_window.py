import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog, QGridLayout, QMessageBox, QSlider,
    QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from src.detection.vehicle_detector import VehicleDetector
from src.detection.parking_space_detector import _EMPTY_KEYWORDS
from src.detection.street_parking_detector import StreetParkingDetector
from src.detection.vehicle_tracker import VehicleTracker
from src.parking.learned_slot_memory import LearnedSlotMemory
from src.parking.occupancy_heatmap import OccupancyHeatmap
from src.detection.drivable_area import DrivableAreaSegmenter
from src.parking import ZoneLoader, ParkingAnalyzer
from src.parking import STATUS_AVAILABLE, STATUS_OCCUPIED, STATUS_FORBIDDEN

VEHICLE_CLASSES = {2: "Araba", 3: "Motosiklet", 5: "Otobüs", 7: "Kamyon"}
VEHICLE_COLORS_CV = {
    2: (0, 255, 0),
    3: (0, 165, 255),
    5: (0, 0, 255),
    7: (255, 0, 255),
}
VEHICLE_COLORS_HEX = {
    2: "#00ff00",
    3: "#ffa500",
    5: "#ff4444",
    7: "#ff00ff",
}


def make_section_label(text):
    lbl = QLabel(text)
    lbl.setFont(QFont("Arial", 9, QFont.Bold))
    lbl.setStyleSheet("color: #94a3b8; padding-top: 10px;")
    return lbl


class StatCard(QFrame):
    def __init__(self, icon, label, color):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1e293b;
                border-radius: 10px;
                border-left: 4px solid {color};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        left = QVBoxLayout()
        self.count_lbl = QLabel("0")
        self.count_lbl.setFont(QFont("Arial", 18, QFont.Bold))
        self.count_lbl.setStyleSheet(f"color: {color}; border: none;")
        type_lbl = QLabel(f"{icon} {label}")
        type_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; border: none;")
        left.addWidget(self.count_lbl)
        left.addWidget(type_lbl)

        layout.addLayout(left)
        layout.addStretch()

    def set_count(self, n):
        self.count_lbl.setText(str(n))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Parking AI")
        self.setGeometry(100, 100, 1400, 860)
        self.setMinimumSize(1100, 700)

        self.detector = None
        self.cap = None
        self._last_time = time.time()
        self._fps = 0.0
        self.analyzer = None
        self._last_frame = None
        self._rgb_buf = None
        self.auto_detector = None

        self._conf_thresh = 0.20
        self._iou_thresh  = 0.15

        self._street_mode       = False
        self._min_gap_ratio     = 0.30   # yan kamera: park boşluğu ≈ 0.3-2.0× araç genişliği
        self._row_band_ratio    = 0.80   # tek sıra band genişliği
        self._ignore_top_ratio  = 0.20
        self.street_detector: StreetParkingDetector | None = None
        # Performans: YOLO inference frame skipping.
        # Her N karede bir tam pipeline (YOLO + tracker + analiz);
        # ara karelerde son sonuçlar yeniden kullanılır — bbox'lar 1-2 frame
        # eski olabilir ama görsel akıcı kalır. N=3 → ~3x hızlanma.
        self._inference_period  = 3
        self._inference_tick    = 0
        self._last_detections: list = []
        self._last_obstacles:  list = []
        self._last_result: dict | None = None
        self._last_learned_status: list = []
        self._last_static_mask: list | None = None
        # Araç tracker (hareketli ≠ park etmiş ayrımı)
        self.vehicle_tracker    = VehicleTracker(
            history_len=60,
            min_history=8,
            max_disp_ratio=0.18,
        )
        # Long-term öğrenilmiş slot kütüğü
        self.learned_slots      = LearnedSlotMemory()
        # Bir aracı "kalıcı slot" saymak için minimum statik tracker güncellemesi.
        # Frame skipping (period=3) nedeniyle bu değer × period kadar gerçek
        # frame'e karşılık gelir. 30 × 3 = 90 gerçek frame ≈ 3 sn @30 FPS.
        self._learn_min_frames  = 30
        # Zamansal araç olasılığı haritası — slot adaylarını filtreler ve
        # her slot için 0-1 güven skoru üretir.
        self.heatmap            = OccupancyHeatmap()
        self._slot_min_prob     = 0.04  # gerçek slot için minimum olasılık
        # Drivable area segmentasyonu (YOLOPv2). Model varsa yol yüzeyi
        # maskesi klasik LAB maskesinin yerini alır (çok daha doğru).
        # Ağır (~58 ms) → seyrek çalıştır + cache.
        try:
            self.drivable = DrivableAreaSegmenter("models/yolopv2.pt")
        except Exception:
            self.drivable = None
        self._drivable_period   = 12   # her 12 inference frame'de bir (~1.2 sn)
        self._drivable_tick     = 0
        self._last_drivable_mask = None

        self._log_file = None
        self._log_writer = None
        self._logging = False
        self._frame_count = 0

        self._build_ui()
        self._load_model()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.video_label = QLabel("Kamera / Video Bekleniyor")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color:#0f172a; border-radius:12px; color:#475569; font-size:14px;"
        )
        self.video_label.setMinimumSize(900, 660)

        panel = QVBoxLayout()
        panel.setSpacing(6)
        panel.setContentsMargins(14, 14, 14, 14)

        # Başlık
        title = QLabel("Smart Parking AI")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        title.setStyleSheet("color:#3b82f6;")
        title.setAlignment(Qt.AlignCenter)
        panel.addWidget(title)

        fps_row = QHBoxLayout()
        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setStyleSheet("color:#64748b; font-size:10px;")
        fps_row.addStretch()
        fps_row.addWidget(self.fps_lbl)
        panel.addLayout(fps_row)

        # ── Kaynak ──
        panel.addWidget(make_section_label("KAYNAK"))

        src_grid = QGridLayout()
        src_grid.setSpacing(5)
        self.start_btn = self._btn("▶ Kamera", "#2563eb")
        self.vid_btn   = self._btn("📂 Video",  "#0891b2")
        self.img_btn   = self._btn("🖼 Resim",  "#7c3aed")
        self.stop_btn  = self._btn("■ Durdur", "#dc2626")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_camera)
        self.vid_btn.clicked.connect(self.load_video)
        self.img_btn.clicked.connect(self.load_image)
        self.stop_btn.clicked.connect(self.stop_feed)
        src_grid.addWidget(self.start_btn, 0, 0)
        src_grid.addWidget(self.vid_btn,   0, 1)
        src_grid.addWidget(self.img_btn,   1, 0)
        src_grid.addWidget(self.stop_btn,  1, 1)
        panel.addLayout(src_grid)

        # ── Sokak Modu ──
        panel.addWidget(make_section_label("SOKAK PARK MODU"))

        self.street_btn = self._btn("🚗  Sokak Modunu Ac", "#0e7490")
        self.street_btn.setCheckable(True)
        self.street_btn.clicked.connect(self._toggle_street_mode)
        self.street_btn.setFixedHeight(44)
        panel.addWidget(self.street_btn)

        # Otomatik tespit ayarları
        strip_box = QFrame()
        strip_box.setStyleSheet(
            "QFrame { background:#0c2233; border:1px solid #0e7490; border-radius:8px; }"
        )
        strip_layout = QVBoxLayout(strip_box)
        strip_layout.setContentsMargins(8, 8, 8, 8)
        strip_layout.setSpacing(4)

        strip_layout.addWidget(self._info(
            "Min Bosluk — arac genisligine oran (0.40 = arac genisliginin %40'i)"
        ))
        self._slider_row_into(
            strip_layout, "Bos:", 10, 150, int(self._min_gap_ratio * 100),
            lambda v: self._set_min_gap(v)
        )
        strip_layout.addWidget(self._info(
            "Sira Band — ayni park sirasindaki araclari gruplama toleransi"
        ))
        self._slider_row_into(
            strip_layout, "Sir:", 30, 150, int(self._row_band_ratio * 100),
            lambda v: self._set_row_tol(v)
        )
        strip_layout.addWidget(self._info(
            "Ust Yoksay — cercevenin ust %X'i (gokyuzu, agac) yok sayilir"
        ))
        self._slider_row_into(
            strip_layout, "Yok:", 0, 60, int(self._ignore_top_ratio * 100),
            lambda v: self._set_ignore_top(v)
        )
        panel.addWidget(strip_box)

        # ── Araç Tespiti ──
        panel.addWidget(make_section_label("ARAC TESPİTİ"))

        panel.addLayout(self._slider_row(
            "Conf:", 5, 95, int(self._conf_thresh * 100),
            lambda v: self._set_conf(v)
        ))

        counts_grid = QGridLayout()
        counts_grid.setSpacing(4)
        self.stat_cards = {}
        icons = {2: "🚗", 3: "🏍", 5: "🚌", 7: "🚛"}
        for i, (cls_id, name) in enumerate(VEHICLE_CLASSES.items()):
            card = StatCard(icons[cls_id], name, VEHICLE_COLORS_HEX[cls_id])
            self.stat_cards[cls_id] = card
            counts_grid.addWidget(card, i // 2, i % 2)
        panel.addLayout(counts_grid)

        # ── Park Durumu ──
        panel.addWidget(make_section_label("PARK DURUMU"))

        self.park_cards = {
            STATUS_AVAILABLE: StatCard("🟢", "Bos",  "#00dc50"),
            STATUS_OCCUPIED:  StatCard("🔴", "Dolu", "#3c3cdc"),
            STATUS_FORBIDDEN: StatCard("⚠️",  "Yasak","#c82020"),
        }
        park_grid = QGridLayout()
        park_grid.setSpacing(4)
        for i, card in enumerate(self.park_cards.values()):
            park_grid.addWidget(card, 0, i)
        panel.addLayout(park_grid)

        self.occupancy_lbl = QLabel("")
        self.occupancy_lbl.setStyleSheet(
            "color:#f1f5f9; font-size:12px; font-weight:bold;"
            "background:#1e293b; border-radius:6px; padding:4px 8px;"
        )
        self.occupancy_lbl.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.occupancy_lbl)

        # ── Durum / Log ──
        panel.addWidget(make_section_label("DURUM"))

        self.status_lbl = QLabel("Hazir")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_lbl.setStyleSheet("color:#e2e8f0; font-size:10px; padding:4px;")
        scroll = QScrollArea()
        scroll.setWidget(self.status_lbl)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(80)
        scroll.setStyleSheet("background:#0f172a; border:1px solid #1e293b;")
        panel.addWidget(scroll)

        btn_row = QHBoxLayout()
        self.snapshot_btn = self._btn("📸 Snapshot", "#b45309")
        self.log_btn      = self._btn("⏺ Log", "#0f766e")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.log_btn.clicked.connect(self.toggle_logging)
        btn_row.addWidget(self.snapshot_btn)
        btn_row.addWidget(self.log_btn)
        panel.addLayout(btn_row)

        # Zone (gizli — mevcut kodla uyumluluk için)
        self.zone_lbl       = QLabel("")
        self.auto_det_lbl   = QLabel("")
        self.auto_det_btn   = QPushButton()
        self.auto_clear_btn = QPushButton()
        self.zone_btn       = QPushButton()

        panel.addStretch()

        panel_frame = QFrame()
        panel_frame.setFixedWidth(270)
        panel_frame.setStyleSheet("background-color:#0f172a; border-radius:12px;")
        panel_frame.setLayout(panel)

        root = QHBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(panel_frame)
        self.setLayout(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _info(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#475569; font-size:9px;")
        lbl.setWordWrap(True)
        return lbl

    def _slider_row_into(self, layout, label_text, lo, hi, init, callback):
        row = self._slider_row(label_text, lo, hi, init, callback)
        layout.addLayout(row)
        return row

    def _btn(self, text, color):
        b = QPushButton(text)
        b.setFixedHeight(38)
        b.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background-color: #334155; color: #64748b; }}
        """)
        return b

    def _slider_row(self, label_text, lo, hi, init, callback):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(32)
        lbl.setStyleSheet("color: #e2e8f0; font-size: 10px; font-weight: bold;")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(init)
        slider.setFixedHeight(18)

        val_lbl = QLabel(f"{init/100:.2f}")
        val_lbl.setFixedWidth(34)
        val_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px;")

        def on_change(v):
            val_lbl.setText(f"{v/100:.2f}")
            callback(v)

        slider.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(val_lbl)
        return row

    def _toggle_street_mode(self):
        self._street_mode = self.street_btn.isChecked()
        if self._street_mode:
            self.street_btn.setText("🚗  Sokak Modu ACIK")
            self._rebuild_street_detector()
        else:
            self.street_btn.setText("🚗  Sokak Modunu Ac")
            self.street_detector = None
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _rebuild_street_detector(self):
        """Street detector'ı tüm iyileştirilmiş parametrelerle yeniden oluştur."""
        self.street_detector = StreetParkingDetector(
            min_gap_ratio=self._min_gap_ratio,
            row_band_ratio=self._row_band_ratio,
            ignore_top_ratio=self._ignore_top_ratio,
            # İyileştirilmiş sabit parametreler
            bottom_align_tol=0.35,
            lateral_split_ratio=3.5,
            max_gap_ratio=5.0,
            max_spaces_per_gap=3,
            max_edge_extension_ratio=0.40,
            road_center_reject_ratio=0.0,
            road_color_tol_h=35.0,
            road_color_tol_s=80.0,
            road_color_tol_v=80.0,
        )

    def _set_min_gap(self, v):
        self._min_gap_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_row_tol(self, v):
        self._row_band_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_ignore_top(self, v):
        self._ignore_top_ratio = v / 100
        if self._street_mode:
            self._rebuild_street_detector()
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_conf(self, v):
        self._conf_thresh = v / 100
        if self.detector:
            self.detector.conf = self._conf_thresh
        if self.auto_detector:
            self.auto_detector.conf = self._conf_thresh
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_iou(self, v):
        self._iou_thresh = v / 100
        if self.analyzer:
            self.analyzer.iou_threshold = self._iou_thresh
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── Oto Tespit ────────────────────────────────────────────────
    def _load_auto_detector(self):
        from src.detection.parking_space_detector import ParkingSpaceDetector  # noqa: F811
        path, _ = QFileDialog.getOpenFileName(
            self, "Parking Detector Model Seç", "models/",
            "Model (*.pt)"
        )
        if not path:
            return
        try:
            self.auto_detector = ParkingSpaceDetector(path, conf=self._conf_thresh)
            self.auto_clear_btn.setEnabled(True)
            # Sınıf eşleşmelerini göster
            cmap = self.auto_detector.class_names
            raw_names = self.auto_detector.model.names
            bos_ids  = [f"{k}({raw_names[k]})" for k, v in cmap.items() if v == "BOS"]
            dolu_ids = [f"{k}({raw_names[k]})" for k, v in cmap.items() if v == "DOLU"]
            other_ids = [f"{k}:{v}({raw_names[k]})" for k, v in cmap.items() if v not in ("BOS", "DOLU")]
            info = f"Model: {Path(path).name}\n"
            info += f"BOS  → {bos_ids or 'EŞLEŞMEDİ!'}\n"
            info += f"DOLU → {dolu_ids or 'EŞLEŞMEDİ!'}"
            if other_ids:
                info += f"\nDiğer → {other_ids}"
            self.auto_det_lbl.setText(f"Model: {Path(path).name}")
            self.status_lbl.setText(info)
            if not bos_ids:
                QMessageBox.warning(
                    self, "Sınıf Eşleşme Uyarısı",
                    f"Modelde BOS sınıfı bulunamadı!\n\n"
                    f"Model sınıfları: {dict(raw_names)}\n\n"
                    f"Beklenen anahtar kelimeler: {_EMPTY_KEYWORDS}"
                )
            if self._last_frame is not None and self.cap is None:
                self._process_and_show(self._last_frame)
        except Exception as e:
            self.status_lbl.setText(f"Oto model hatası:\n{e}")

    def _clear_auto_detector(self):
        self.auto_detector = None
        self.auto_det_lbl.setText("Model: yüklenmedi")
        self.auto_clear_btn.setEnabled(False)
        self.status_lbl.setText("Oto tespit modu kapatıldı.")
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    # ── Model ─────────────────────────────────────────────────────
    def _load_model(self):
        self.status_lbl.setText("Model yükleniyor...")
        try:
            self.detector = VehicleDetector(conf=self._conf_thresh)
            self.status_lbl.setText("Model yüklendi.")
        except Exception as e:
            self.status_lbl.setText(f"Model hatası:\n{e}")

    # ── Snapshot ──────────────────────────────────────────────────
    def take_snapshot(self):
        if self._last_frame is None:
            self.status_lbl.setText("Snapshot: görüntü yok.")
            return
        out_dir = Path("outputs/snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = out_dir / f"snapshot_{ts}.jpg"
        frame = self._last_frame.copy()
        if self.analyzer and self.detector:
            dets = self.detector.detect(frame)
            result = self.analyzer.analyze(dets)
            frame = self.analyzer.draw(frame, result, dets)
        cv2.imwrite(str(img_path), frame)
        self.status_lbl.setText(f"Snapshot: {img_path.name}")

    # ── Loglama ───────────────────────────────────────────────────
    def toggle_logging(self):
        if not self._logging:
            out_dir = Path("outputs/metrics")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = out_dir / f"log_{ts}.csv"
            self._log_file = open(log_path, "w", newline="", encoding="utf-8")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "frame", "timestamp", "vehicles",
                "available", "occupied", "forbidden_vehicles"
            ])
            self._frame_count = 0
            self._logging = True
            self.log_btn.setText("⏹  Loglama Durdur")
            self.log_btn.setStyleSheet(self.log_btn.styleSheet().replace("#0f766e", "#dc2626"))
            self.status_lbl.setText(f"Log: {log_path.name}")
        else:
            self._stop_logging()

    def _stop_logging(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        self._logging = False
        self.log_btn.setText("⏺  Loglama Başlat")
        self.log_btn.setStyleSheet(self.log_btn.styleSheet().replace("#dc2626", "#0f766e"))
        self.status_lbl.setText("Loglama durduruldu.")

    def _log_frame(self, vehicle_count, available, occupied, forbidden):
        if not self._logging or self._log_writer is None:
            return
        self._frame_count += 1
        self._log_writer.writerow([
            self._frame_count,
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            vehicle_count, available, occupied, forbidden
        ])

    # ── Zone yükleme ──────────────────────────────────────────────
    def _try_load_json(self, json_path: str) -> bool:
        try:
            loader = ZoneLoader(json_path)
            self.analyzer = ParkingAnalyzer(loader, iou_threshold=self._iou_thresh)
            n_park = len(loader.parking_zones)
            n_forb = len(loader.forbidden_zones)
            self.zone_lbl.setText(
                f"Zone: {Path(json_path).name}\n"
                f"{n_park} park · {n_forb} yasak"
            )
            self.status_lbl.setText(f"Zone yüklendi: {n_park + n_forb} bölge")
            return True
        except Exception as e:
            self.status_lbl.setText(f"Zone hatası:\n{e}")
            return False

    def load_zones_from_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Zone için Resim Seç", "data/",
            "Resim (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        json_path = str(Path(path).with_suffix(".json"))
        json_exists = Path(json_path).exists()

        if json_exists:
            msg = QMessageBox(self)
            msg.setWindowTitle("Zone Mevcut")
            msg.setText(f"{Path(json_path).name} zaten var.\nNe yapmak istersiniz?")
            btn_load = msg.addButton("Yükle", QMessageBox.AcceptRole)
            msg.addButton("Annotator ile Düzenle", QMessageBox.RejectRole)
            msg.exec_()
            if msg.clickedButton() == btn_load:
                if self._try_load_json(json_path):
                    ref = cv2.imread(path)
                    if ref is not None:
                        self._process_and_show(ref)
                return
            open_annotator = True
        else:
            reply = QMessageBox.question(
                self, "Zone Bulunamadı",
                f"{Path(json_path).name} bulunamadı.\nAnnotator ile oluşturulsun mu?",
                QMessageBox.Yes | QMessageBox.No
            )
            open_annotator = (reply == QMessageBox.Yes)

        if not open_annotator:
            return
        try:
            from src.parking.zone_annotator import ZoneAnnotator
            load_existing = json_path if json_exists else None
            ZoneAnnotator(path, output_path=json_path, load_path=load_existing).run()
        except Exception as e:
            self.status_lbl.setText(f"Annotator hatası:\n{e}")
            return
        if Path(json_path).exists():
            if self._try_load_json(json_path):
                ref = cv2.imread(path)
                if ref is not None:
                    self._process_and_show(ref)

    def _auto_load_zones(self, source_path: str):
        json_path = str(Path(source_path).with_suffix(".json"))
        if Path(json_path).exists():
            self._try_load_json(json_path)
        else:
            self.analyzer = None
            self.zone_lbl.setText("Zone: yüklenmedi")
            for card in self.park_cards.values():
                card.set_count(0)
            self.occupancy_lbl.setText("")

    # ── Feed yönetimi ─────────────────────────────────────────────
    def _start_feed(self):
        self._last_time = time.time()
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.vid_btn.setEnabled(False)
        self.img_btn.setEnabled(False)
        if self.street_detector:
            self.street_detector.reset_history()
        self.vehicle_tracker.reset()
        self.learned_slots.reset()
        self.heatmap.reset()
        self._drivable_tick = 0
        self._last_drivable_mask = None
        self._inference_tick = 0
        self._last_detections = []
        self._last_obstacles = []
        self._last_result = None
        self._last_learned_status = []
        self._last_static_mask = None
        self.timer.start(30)

    def start_camera(self):
        self.stop_feed()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_lbl.setText("Kamera bulunamadı.")
            self.cap = None
            return
        self.status_lbl.setText("Kamera aktif.")
        self._start_feed()

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Video Seç", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        self.stop_feed()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_lbl.setText("Video açılamadı.")
            self.cap = None
            return
        self.status_lbl.setText("Video oynatılıyor.")
        self._auto_load_zones(path)
        self._start_feed()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Resim Seç", "", "Resim (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        self.stop_feed()
        frame = cv2.imread(path)
        if frame is None:
            self.status_lbl.setText("Resim açılamadı.")
            return
        # Bug #1-2: Yeni fotoğraf yüklenince tüm önbellek sıfırlanmalı.
        # Aksi halde _last_result bir önceki fotoğrafın analiz sonucunu
        # taşır ve yeni fotoğrafta hiç analiz yapılmaz.
        self._last_result = None
        self._last_obstacles = []
        self._last_drivable_mask = None
        self._last_detections = []
        self._last_static_mask = None
        if self.street_detector is not None:
            self.street_detector.reset_history()
        self._auto_load_zones(path)
        self._last_frame = frame
        self._process_and_show(frame)
        self.status_lbl.setText("Resim yüklendi.")

    def stop_feed(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.vid_btn.setEnabled(True)
        self.img_btn.setEnabled(True)
        self.fps_lbl.setText("FPS: —")
        self.status_lbl.setText("Durduruldu.")

    def closeEvent(self, event):
        self._stop_logging()
        super().closeEvent(event)

    # ── Frame işleme ──────────────────────────────────────────────
    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_feed()
            self.status_lbl.setText("Bitti.")
            return

        self._last_frame = frame
        self._process_and_show(frame)

        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._last_time, 1e-6))
        self._last_time = now
        self.fps_lbl.setText(f"FPS: {self._fps:.1f}")

    def _process_and_show(self, frame):
        counts = {cls_id: 0 for cls_id in VEHICLE_CLASSES}
        detections = []

        # Inference stratejisi:
        #  - Street mode video/kamera → her N frame'de bir YOLO+engel birleşik
        #    çağrı (frame skipping); ara karelerde cache.
        #  - Diğer modlar → her frame YOLO (mevcut davranış).
        if self.detector:
            if self._street_mode and self.cap is not None:
                # Video/kamera: frame skipping ile YOLO+engel çağrısı
                if self._inference_tick % self._inference_period == 0:
                    vehicles, obstacles = self.detector.detect_all(frame)
                    self._last_detections = vehicles
                    self._last_obstacles  = obstacles
                self._inference_tick += 1
                detections = self._last_detections
            elif self._street_mode:
                # Bug #3: Fotoğraf modunda da detect_all çağrılmalı;
                # aksi halde _last_obstacles boş kalır ve engel maskeleme çalışmaz.
                vehicles, obstacles = self.detector.detect_all(frame)
                detections = vehicles
                self._last_obstacles = obstacles
            else:
                detections = self.detector.detect(frame)

            for det in detections:
                cls_id = det.get("class_id")
                if cls_id in VEHICLE_CLASSES:
                    counts[cls_id] += 1

        available = occupied = forbidden = 0
        out = frame.copy()

        if self._street_mode and self.street_detector:
            # Önce araçları normal şekilde çiz
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Tracker + analiz + öğrenme: yalnızca inference frame'inde yenile.
            # Ara karelerde cached result/learned_status kullanılır → 3x hız.
            is_inference_frame = (
                self.cap is not None
                and (self._inference_tick - 1) % self._inference_period == 0
            )

            if self.cap is None:
                static_mask = None
            elif is_inference_frame:
                # Frame'i tracker'a geçir → ego-motion (optik akış) düzeltmesi
                static_mask = self.vehicle_tracker.update(detections, frame=frame)
                self._last_static_mask = static_mask
            else:
                static_mask = self._last_static_mask

            if is_inference_frame or self._last_result is None:
                # Drivable area (YOLOPv2) — seyrek hesapla + cache (ağır model)
                if (self.drivable is not None and self.drivable.available
                        and self.cap is not None):
                    if (self._drivable_tick % self._drivable_period == 0
                            or self._last_drivable_mask is None):
                        try:
                            da_mask, _ll = self.drivable.infer(frame)
                            # Park yerleri yol-kaldırım SINIRINDA → drivable
                            # area'yı dilate ederek kenarı içeri al. Aksi halde
                            # slot bbox'larının yarısı kaldırıma düşüp elenir.
                            if da_mask is not None:
                                # Park yerleri yol kenarındaki şeride uzandığı
                                # için drivable area'yı yatayda DAHA çok
                                # dilate ediyoruz (dikeyde araya kaldırım
                                # girmesin diye yatay > dikey).
                                # Bug #5: Geniş kernel maske bisiklet yolu /
                                # kaldırıma sızıyor. Kernel boyutunu küçülttük.
                                kx = max(10, int(frame.shape[1] * 0.018))
                                ky = max(5, int(frame.shape[0] * 0.010))
                                kernel = cv2.getStructuringElement(
                                    cv2.MORPH_RECT, (kx, ky)
                                )
                                da_mask = cv2.dilate(da_mask, kernel, iterations=1)
                            self._last_drivable_mask = da_mask
                        except Exception:
                            self._last_drivable_mask = None
                    self._drivable_tick += 1

                result = self.street_detector.analyze(
                    frame, detections,
                    obstacles=self._last_obstacles,
                    static_mask=static_mask,
                    external_road_mask=self._last_drivable_mask,
                )

                # Occupancy heatmap güncelle (ego-motion ile kaydırılarak)
                if self.cap is not None:
                    vehicle_bboxes = [d["bbox"] for d in detections]
                    ego_dx, ego_dy = self.vehicle_tracker.last_ego_motion
                    self.heatmap.update(frame.shape, vehicle_bboxes,
                                        ego_dx=ego_dx, ego_dy=ego_dy)

                learned_status: list = []
                if self.cap is not None:
                    static_tracks = self.vehicle_tracker.get_static_tracks(
                        min_frames=self._learn_min_frames
                    )
                    vehicle_bboxes = [d["bbox"] for d in detections]
                    learned_status = self.learned_slots.update(
                        static_tracks, vehicle_bboxes,
                        road_mask=self._last_drivable_mask,
                    )

                # Heuristic boş slot'lardan öğrenilmiş ile çakışanları çıkar
                from src.detection.street_parking_detector import _bbox_iou as _iou_fn
                learned_bboxes = [s["bbox"] for s in learned_status]
                result["empty_spaces"] = [
                    e for e in result["empty_spaces"]
                    if not any(_iou_fn(e, lb) > 0.40 for lb in learned_bboxes)
                ]

                # Heatmap süzgeci: ısınma sonrası, slot'un altındaki bölgede
                # geçmişte hiç araç görülmediyse (yol ortası, kaldırım) ele.
                # Her geçen slot için 0-1 confidence skoru üret.
                slot_confidences: list[float] = []
                if self.cap is not None and self.heatmap.is_warmed_up:
                    filtered = []
                    for s in result["empty_spaces"]:
                        # Slot çevresinde park sırası araç(lar)ı var mı?
                        # Yolun ortasındaki/hayalî slot'lar 0'a yakın olur.
                        p = self.heatmap.slot_neighborhood_max(
                            s, frame.shape, expand=0.6
                        )
                        if p >= self._slot_min_prob:
                            filtered.append(s)
                            slot_confidences.append(min(1.0, p / 0.5))
                    result["empty_spaces"]    = filtered
                    result["slot_confidences"] = slot_confidences
                else:
                    result["slot_confidences"] = [0.5] * len(result["empty_spaces"])

                self._last_result = result
                self._last_learned_status = learned_status
            else:
                result = self._last_result
                learned_status = self._last_learned_status

            # Çizim katmanları — heuristic boş slot'lar confidence-aware
            empty_spaces = result.get("empty_spaces", [])
            confs        = result.get("slot_confidences",
                                      [0.5] * len(empty_spaces))
            if empty_spaces:
                overlay = out.copy()
                for s, c in zip(empty_spaces, confs):
                    x1, y1, x2, y2 = map(int, s)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  (0, 220, 80), -1)
                # Dolgu alfası ortalama confidence ile ölçekli (0.18-0.45)
                avg_c = float(np.mean(confs)) if confs else 0.5
                alpha = 0.18 + 0.27 * avg_c
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
                for s, c in zip(empty_spaces, confs):
                    x1, y1, x2, y2 = map(int, s)
                    # Kontur kalınlığı confidence'a göre 2-3
                    thick = 3 if c >= 0.6 else 2
                    cv2.rectangle(out, (x1, y1), (x2, y2),
                                  (0, 220, 80), thick)
                    label = f"BOS %{int(c * 100)}"
                    (tw, th), _ = cv2.getTextSize(label,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.rectangle(out,
                                  (cx - tw // 2 - 4, cy - th // 2 - 4),
                                  (cx + tw // 2 + 4, cy + th // 2 + 4),
                                  (0, 220, 80), -1)
                    cv2.putText(out, label,
                                (cx - tw // 2, cy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 2, cv2.LINE_AA)

            # Öğrenilmiş kalıcı slot'lar — yalnızca BOS olanları göster.
            # DOLU slot'lar araç bbox'ı ile zaten temsil edildiği için
            # tekrar çizmek görsel kalabalık yaratıyor.
            COLOR_LEARNED_EMPTY = (255, 200, 0)   # cyan
            learned_empty_slots = [s for s in learned_status if not s["occupied"]]
            if learned_empty_slots:
                overlay = out.copy()
                for s in learned_empty_slots:
                    x1, y1, x2, y2 = map(int, s["bbox"])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  COLOR_LEARNED_EMPTY, -1)
                cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
                for s in learned_empty_slots:
                    x1, y1, x2, y2 = map(int, s["bbox"])
                    cv2.rectangle(out, (x1, y1), (x2, y2),
                                  COLOR_LEARNED_EMPTY, 2)
                    # Etiket yalnızca slot yeterince büyükse
                    if (x2 - x1) >= 40 and (y2 - y1) >= 20:
                        cv2.putText(out, "BOS+", (x1 + 4, y2 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    COLOR_LEARNED_EMPTY, 2, cv2.LINE_AA)

            # Engelleri sadece ince konturla; etiket yazma (kalabalık azaltma)
            for ob in self._last_obstacles:
                ox1, oy1, ox2, oy2 = map(int, ob["bbox"])
                cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (60, 60, 220), 1)

            # Sayımlar: öğrenilmiş + dedup'lanmış heuristic
            learned_empty    = sum(1 for s in learned_status if not s["occupied"])
            learned_occupied = sum(1 for s in learned_status if s["occupied"])
            heuristic_empty  = len(result["empty_spaces"])
            # DOLU: tracker'ın gördüğü tüm statik araçlar (heuristic parked
            # zaten learned slot'ları da içerebilir — max ile birleştir)
            available = learned_empty + heuristic_empty
            occupied  = max(result["occupied_count"], learned_occupied)
            total     = available + occupied
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(0)
            if total > 0:
                pct = int(occupied / total * 100)
                self.occupancy_lbl.setText(f"{available} bos · {occupied} dolu  (%{pct})")
            else:
                self.occupancy_lbl.setText("Arac tespit edilemedi")
            for cls_id, card in self.stat_cards.items():
                card.set_count(counts[cls_id])
            self._log_frame(sum(counts.values()), available, occupied, 0)
            self._show_frame(out)
            return

        if self.auto_detector:
            auto_dets = self.auto_detector.detect(frame)
            out = self.auto_detector.draw(out, auto_dets)
            # Araç kutularını üzerine çiz (etiketsiz)
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(out, (x1, y1), (x2, y2),
                              VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0)), 2)
            available = sum(1 for d in auto_dets if d["status"] == "BOS")
            occupied  = sum(1 for d in auto_dets if d["status"] == "DOLU")
            # Model boş yer tespit edemiyorsa zone sistemiyle tamamla
            if available == 0 and self.analyzer:
                zone_result = self.analyzer.analyze(detections)
                available = zone_result.available
                occupied  = max(occupied, zone_result.occupied)
            # Yasaklı zone'lar zone sistemiyle de gösterilebilir
            if self.analyzer:
                zone_result = self.analyzer.analyze(detections)
                forbidden = zone_result.forbidden_vehicles
                for zs in zone_result.zone_statuses:
                    if zs.status == STATUS_FORBIDDEN:
                        pts = zs.zone.polygon
                        overlay = out.copy()
                        cv2.fillPoly(overlay, [pts], (0, 0, 200))
                        cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
                        cv2.polylines(out, [pts], True, (0, 0, 200), 2)
                        cx = int(pts[:, 0].mean())
                        cy = int(pts[:, 1].mean())
                        cv2.putText(out, "YASAK", (cx - 25, cy + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)
            total = available + occupied
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(forbidden)
            if total > 0:
                pct = int(occupied / total * 100)
                self.occupancy_lbl.setText(f"{occupied}/{total} slot dolu  (%{pct})")
            else:
                self.occupancy_lbl.setText("")
        elif self.analyzer:
            result = self.analyzer.analyze(detections)
            out = self.analyzer.draw(out, result, detections)
            available = result.available
            occupied  = result.occupied
            forbidden = result.forbidden_vehicles
            total     = result.total_parking
            self.park_cards[STATUS_AVAILABLE].set_count(available)
            self.park_cards[STATUS_OCCUPIED].set_count(occupied)
            self.park_cards[STATUS_FORBIDDEN].set_count(forbidden)
            if total > 0:
                pct = int(occupied / total * 100)
                self.occupancy_lbl.setText(f"{occupied}/{total} slot dolu  (%{pct})")
            else:
                self.occupancy_lbl.setText("")
        else:
            for card in self.park_cards.values():
                card.set_count(0)
            self.occupancy_lbl.setText("")
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, det["bbox"])
                conf = det["confidence"]
                color = VEHICLE_COLORS_CV.get(cls_id, (0, 255, 0))
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for cls_id, card in self.stat_cards.items():
            card.set_count(counts[cls_id])

        self._log_frame(sum(counts.values()), available, occupied, forbidden)

        self._show_frame(out)

    def _show_frame(self, out):
        self._rgb_buf = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        h, w, ch = self._rgb_buf.shape
        img = QImage(self._rgb_buf.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
