"""Video sabitleme — elde-çekim kaymasını kompanze eder (teknik amaçlı).

Telefon elde çekildiğinde kareler titrer/kayar. Bu durum:
  - Tek seferlik IPM kalibrasyonunu geçersiz kılar (homografi referans kareye
    göre kuruluydu),
  - Çizgi-ızgara ve zamansal oylamanın kare-arası eşleşmesini bozar.

Sabitleme her kareyi REFERANS kareye (IPM kalibrasyonunun yapıldığı kare ya da
ilk kare) ORB özellikleri + RANSAC homografisiyle hizalar. Böylece arka planda
IPM ve ızgara geçerli kalır — yalnızca görsel değil, doğruluğu koruyan teknik
bir adımdır.

Graceful: yeterli özellik/eşleşme yoksa kare olduğu gibi geçirilir (success=
False), sistem asla bozulmaz.
"""

from __future__ import annotations

import cv2
import numpy as np


class VideoStabilizer:
    def __init__(self, max_features: int = 600, min_matches: int = 12,
                 ransac_thresh: float = 4.0):
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self._orb = cv2.ORB_create(max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp = None
        self.ref_des = None
        self.ref_shape = None

    def reset(self):
        self.ref_kp = None
        self.ref_des = None
        self.ref_shape = None

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def set_reference(self, frame) -> bool:
        """Hizalama referansını ayarla (ör. IPM kalibrasyon karesi)."""
        gray = self._gray(frame)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_matches:
            return False
        self.ref_kp, self.ref_des = kp, des
        self.ref_shape = frame.shape[:2]
        return True

    def stabilize(self, frame):
        """Kareyi referansa hizala. Döner: (hizalanmış_kare, success).

        Referans yoksa bu kareyi referans yapar ve değiştirmeden döndürür.
        Yeterli eşleşme yoksa kareyi olduğu gibi döndürür (graceful).
        """
        if self.ref_des is None:
            self.set_reference(frame)
            return frame, False

        gray = self._gray(frame)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_matches:
            return frame, False

        matches = self._bf.match(des, self.ref_des)
        if len(matches) < self.min_matches:
            return frame, False
        matches = sorted(matches, key=lambda m: m.distance)

        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return frame, False

        h, w = frame.shape[:2]
        warped = cv2.warpPerspective(frame, H, (w, h))
        return warped, True
