"""IPM kalibrasyon penceresi — zemindeki 4 köşeyi tıklayarak homografi kur.

Görüntü, bilinen bir ölçekte (display_scale) tam pikmap boyutunda bir etikette
gösterilir; böylece tıklama koordinatları doğrusal olarak orijinal görüntü
koordinatlarına çevrilir (merkez/ofset belirsizliği olmaz → koordinat hatası
oluşmaz).

Kullanıcı sırayla 4 köşe işaretler: sol-üst, sağ-üst, sağ-alt, sol-alt.
Ardından dikdörtgenin gerçek genişlik/yükseklik (metre) değerlerini girer.
Pencere kabul edilirse orijinal-koordinat 4 nokta + gerçek boyutları döndürür.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint

_LABELS = ["1: sol-ust", "2: sag-ust", "3: sag-alt", "4: sol-alt"]


class _ClickableImage(QLabel):
    """Pikmapını tam boyutta gösteren, tıklanan noktaları işaretleyen etiket."""

    def __init__(self, pixmap: QPixmap, max_points: int = 4,
                 close_polygon: bool = True):
        super().__init__()
        self._base = pixmap
        self._max_points = max_points
        self._close = close_polygon
        self.setFixedSize(pixmap.size())
        self.setPixmap(pixmap)
        self.points: list[QPoint] = []

    def mousePressEvent(self, event):
        if len(self.points) >= self._max_points:
            return
        self.points.append(event.pos())
        self._redraw()

    def reset(self):
        self.points = []
        self._redraw()

    def _redraw(self):
        pm = QPixmap(self._base)
        painter = QPainter(pm)
        pen = QPen(QColor("#22c55e"), 3)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        for i, p in enumerate(self.points):
            painter.setBrush(QColor("#22c55e"))
            painter.drawEllipse(p, 6, 6)
            painter.drawText(p.x() + 9, p.y() - 9, str(i + 1))
        if len(self.points) >= 2:
            pen2 = QPen(QColor("#3b82f6"), 2)
            painter.setPen(pen2)
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i + 1])
            # Poligonu kapat: 4-nokta modunda tamamlanınca, çok-nokta modunda 3+
            closed = (len(self.points) == 4 if not self._close
                      else len(self.points) >= 3)
            if closed:
                painter.drawLine(self.points[-1], self.points[0])
        painter.end()
        self.setPixmap(pm)


class IPMCalibrationDialog(QDialog):
    """4 köşe + gerçek boyut girerek IPM kalibrasyonu toplar."""

    def __init__(self, frame_bgr: np.ndarray, parent=None, max_w=1000, max_h=620):
        super().__init__(parent)
        self.setWindowTitle("IPM Kalibrasyonu — Zemindeki 4 koseyi tikla")
        self.setStyleSheet("background:#0f172a; color:#e2e8f0;")

        h, w = frame_bgr.shape[:2]
        self.scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        disp = cv2.resize(frame_bgr, (disp_w, disp_h),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._buf = np.ascontiguousarray(rgb)
        qimg = QImage(self._buf.data, disp_w, disp_h, 3 * disp_w,
                      QImage.Format_RGB888)
        self.image = _ClickableImage(QPixmap.fromImage(qimg),
                                     max_points=4, close_polygon=False)

        self.info = QLabel("Sirayla tikla: 1 sol-ust, 2 sag-ust, "
                           "3 sag-alt, 4 sol-alt")
        self.info.setStyleSheet("color:#94a3b8; font-size:12px;")

        # Gerçek boyut girişleri
        dim_row = QHBoxLayout()
        self.w_spin = self._dim_spin(10.0)
        self.h_spin = self._dim_spin(5.0)
        dim_row.addWidget(QLabel("Gercek genislik (m):"))
        dim_row.addWidget(self.w_spin)
        dim_row.addWidget(QLabel("Gercek yukseklik (m):"))
        dim_row.addWidget(self.h_spin)
        dim_row.addStretch()

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Sifirla")
        ok_btn = QPushButton("Onayla")
        cancel_btn = QPushButton("Iptal")
        for b in (reset_btn, ok_btn, cancel_btn):
            b.setFixedHeight(32)
            b.setStyleSheet("QPushButton{background:#1e293b; color:#e2e8f0;"
                            "border-radius:6px; padding:4px 14px;}"
                            "QPushButton:hover{background:#334155;}")
        reset_btn.clicked.connect(self.image.reset)
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.image)
        layout.addLayout(dim_row)
        layout.addLayout(btn_row)

        self.result_points = None
        self.real_w = None
        self.real_h = None

    @staticmethod
    def _dim_spin(value):
        s = QDoubleSpinBox()
        s.setRange(0.5, 100.0)
        s.setSingleStep(0.5)
        s.setValue(value)
        s.setSuffix(" m")
        s.setStyleSheet("QDoubleSpinBox{background:#0f172a; color:#e2e8f0;"
                        "border:1px solid #334155; border-radius:4px; padding:2px;}")
        return s

    def _on_ok(self):
        from PyQt5.QtWidgets import QMessageBox
        if len(self.image.points) != 4:
            QMessageBox.warning(self, "Eksik", "Tam olarak 4 nokta isaretle.")
            return
        # display → orijinal koordinat (birebir ölçek geri alma)
        self.result_points = [(p.x() / self.scale, p.y() / self.scale)
                              for p in self.image.points]
        self.real_w = self.w_spin.value()
        self.real_h = self.h_spin.value()
        self.accept()

    @staticmethod
    def get_calibration(frame_bgr, parent=None):
        """Diyaloğu aç; kabul edilirse (points, real_w, real_h) döndür, yoksa None."""
        dlg = IPMCalibrationDialog(frame_bgr, parent)
        if dlg.exec_() == QDialog.Accepted and dlg.result_points:
            return dlg.result_points, dlg.real_w, dlg.real_h
        return None
