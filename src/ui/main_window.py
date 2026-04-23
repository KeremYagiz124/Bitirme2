import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QFileDialog, QGridLayout, QMessageBox, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from src.detection.vehicle_detector import VehicleDetector
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

        self._conf_thresh = 0.50
        self._iou_thresh  = 0.25

        self._log_file = None
        self._log_writer = None
        self._logging = False
        self._frame_count = 0

        self._build_ui()
        self._load_model()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.video_label = QLabel("Video / Kamera Bekleniyor")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #0f172a;
            border-radius: 12px;
            color: #475569;
            font-size: 14px;
        """)
        self.video_label.setMinimumSize(860, 640)

        panel = QVBoxLayout()
        panel.setSpacing(8)
        panel.setContentsMargins(12, 16, 12, 16)

        title = QLabel("Smart Parking AI")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #3b82f6;")
        title.setAlignment(Qt.AlignCenter)
        panel.addWidget(title)

        subtitle = QLabel("YOLOv8 · IoU Pipeline")
        subtitle.setStyleSheet("color: #64748b; font-size: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        panel.addWidget(subtitle)

        # ── Kontroller ──
        panel.addWidget(make_section_label("KONTROL"))

        self.start_btn    = self._btn("▶  Kamera Başlat", "#2563eb")
        self.stop_btn     = self._btn("■  Durdur",        "#dc2626")
        self.img_btn      = self._btn("🖼  Resim Yükle",   "#7c3aed")
        self.vid_btn      = self._btn("📂  Video Yükle",   "#0891b2")
        self.zone_btn     = self._btn("📍  Zone Tanımla",  "#059669")
        self.snapshot_btn = self._btn("📸  Snapshot Al",   "#b45309")
        self.log_btn      = self._btn("⏺  Loglama Başlat","#0f766e")

        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_feed)
        self.img_btn.clicked.connect(self.load_image)
        self.vid_btn.clicked.connect(self.load_video)
        self.zone_btn.clicked.connect(self.load_zones_from_image)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.log_btn.clicked.connect(self.toggle_logging)

        for b in (self.start_btn, self.stop_btn, self.img_btn,
                  self.vid_btn, self.zone_btn, self.snapshot_btn, self.log_btn):
            panel.addWidget(b)

        self.zone_lbl = QLabel("Zone: yüklenmedi")
        self.zone_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        self.zone_lbl.setWordWrap(True)
        panel.addWidget(self.zone_lbl)

        # ── Sliderlar ──
        panel.addWidget(make_section_label("AYARLAR"))

        conf_desc = QLabel("Tespit hassasiyeti — dusuk: cok tespit, yuksek: emin ol")
        conf_desc.setStyleSheet("color: #475569; font-size: 9px;")
        conf_desc.setWordWrap(True)
        panel.addWidget(conf_desc)
        panel.addLayout(self._slider_row(
            "Conf:", 10, 95, int(self._conf_thresh * 100),
            lambda v: self._set_conf(v)
        ))

        iou_desc = QLabel("Zone ortusme esigi — dusuk: az ortusme yeter, yuksek: tam ortusme")
        iou_desc.setStyleSheet("color: #475569; font-size: 9px;")
        iou_desc.setWordWrap(True)
        panel.addWidget(iou_desc)
        panel.addLayout(self._slider_row(
            "IoU:", 5, 80, int(self._iou_thresh * 100),
            lambda v: self._set_iou(v)
        ))

        # ── Tespit ──
        panel.addWidget(make_section_label("TESPİT"))

        self.stat_cards = {}
        icons = {2: "🚗", 3: "🏍", 5: "🚌", 7: "🚛"}
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (cls_id, name) in enumerate(VEHICLE_CLASSES.items()):
            card = StatCard(icons[cls_id], name, VEHICLE_COLORS_HEX[cls_id])
            self.stat_cards[cls_id] = card
            grid.addWidget(card, i // 2, i % 2)
        panel.addLayout(grid)

        # ── Park Durumu ──
        panel.addWidget(make_section_label("PARK DURUMU"))

        park_grid = QGridLayout()
        park_grid.setSpacing(6)
        self.park_cards = {
            STATUS_AVAILABLE: StatCard("🟢", "Bos Slot",  "#00dc50"),
            STATUS_OCCUPIED:  StatCard("🔴", "Dolu Slot", "#3c3cdc"),
            STATUS_FORBIDDEN: StatCard("⚠️",  "Yasak",    "#c82020"),
        }
        for i, card in enumerate(self.park_cards.values()):
            park_grid.addWidget(card, 0, i)
        panel.addLayout(park_grid)

        # Doluluk oranı
        self.occupancy_lbl = QLabel("")
        self.occupancy_lbl.setStyleSheet(
            "color: #f1f5f9; font-size: 12px; font-weight: bold;"
            "background: #1e293b; border-radius: 6px; padding: 4px 8px;"
        )
        self.occupancy_lbl.setAlignment(Qt.AlignCenter)
        panel.addWidget(self.occupancy_lbl)

        # ── Durum ──
        panel.addWidget(make_section_label("DURUM"))

        self.status_lbl = QLabel("Hazır")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #e2e8f0; font-size: 12px;")
        panel.addWidget(self.status_lbl)

        self.fps_lbl = QLabel("FPS: —")
        self.fps_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        panel.addWidget(self.fps_lbl)

        panel.addStretch()

        panel_frame = QFrame()
        panel_frame.setFixedWidth(280)
        panel_frame.setStyleSheet("background-color: #0f172a; border-radius: 12px;")
        panel_frame.setLayout(panel)

        root = QHBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.video_label, stretch=1)
        root.addWidget(panel_frame)
        self.setLayout(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

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

    def _set_conf(self, v):
        self._conf_thresh = v / 100
        if self.detector:
            self.detector.conf = self._conf_thresh
        if self._last_frame is not None and self.cap is None:
            self._process_and_show(self._last_frame)

    def _set_iou(self, v):
        self._iou_thresh = v / 100
        if self.analyzer:
            self.analyzer.iou_threshold = self._iou_thresh
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

        if self.detector:
            detections = self.detector.detect(frame)
            for det in detections:
                cls_id = det.get("class_id")
                if cls_id in VEHICLE_CLASSES:
                    counts[cls_id] += 1

        available = occupied = forbidden = 0
        out = frame.copy()

        if self.analyzer:
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
