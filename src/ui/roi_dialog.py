"""ROI seçim penceresi — otopark alanını poligonla işaretle.

Kullanıcı görüntü üzerinde 3+ nokta tıklayarak ilgi bölgesini (otopark sınırı)
çizer. Görüntü bilinen ölçekte tam boyutta gösterilir; tıklamalar birebir
orijinal koordinata çevrilir (koordinat hatası olmaz).
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from src.ui.ipm_dialog import _ClickableImage


class RoiSelectionDialog(QDialog):
    """İlgi bölgesi poligonu toplar (en az 3 nokta)."""

    def __init__(self, frame_bgr: np.ndarray, parent=None, max_w=1000, max_h=620):
        super().__init__(parent)
        self.setWindowTitle("ROI Sec — Otopark alaninin kenarlarini tikla (3+ nokta)")
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
                                     max_points=20, close_polygon=True)

        info = QLabel("Otopark alaninin kose noktalarini sirayla tikla. "
                      "En az 3 nokta. Bittiğinde Onayla.")
        info.setStyleSheet("color:#94a3b8; font-size:12px;")

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
        layout.addWidget(info)
        layout.addWidget(self.image)
        layout.addLayout(btn_row)

        self.result_polygon = None

    def _on_ok(self):
        if len(self.image.points) < 3:
            QMessageBox.warning(self, "Eksik", "En az 3 nokta isaretle.")
            return
        self.result_polygon = [(p.x() / self.scale, p.y() / self.scale)
                               for p in self.image.points]
        self.accept()

    @staticmethod
    def get_roi(frame_bgr, parent=None):
        """Diyaloğu aç; kabul edilirse poligon noktaları, yoksa None döndür."""
        dlg = RoiSelectionDialog(frame_bgr, parent)
        if dlg.exec_() == QDialog.Accepted and dlg.result_polygon:
            return dlg.result_polygon
        return None
