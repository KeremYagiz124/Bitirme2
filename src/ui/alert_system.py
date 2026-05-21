"""Bildirim ve uyarı sistemi.

Üç seviye:
  INFO     — mavi  — bilgilendirme
  WARNING  — sarı  — dikkat gerektiren durum
  CRITICAL — kırmızı — acil müdahale gerektiren durum

Aynı uyarı kodu throttle_sec içinde tekrar tetiklenmez.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


LEVEL_COLORS = {
    AlertLevel.INFO:     ("#1e3a5f", "#60a5fa"),   # (bg, fg)
    AlertLevel.WARNING:  ("#3b2a00", "#fbbf24"),
    AlertLevel.CRITICAL: ("#3b0a0a", "#f87171"),
}


@dataclass
class Alert:
    code:    str
    level:   AlertLevel
    message: str
    ts:      float = field(default_factory=time.time)


class AlertSystem:
    """Uyarı geçmişi tutan ve throttle uygulayan merkezi sistem."""

    def __init__(self, throttle_sec: float = 30.0, max_history: int = 50):
        self._throttle   = throttle_sec
        self._max_hist   = max_history
        self._last_fired: dict[str, float] = {}   # code → timestamp
        self._history:    list[Alert]      = []
        self._listeners:  list             = []    # callable(Alert)

    # ── Dinleyici kaydı ──

    def add_listener(self, fn):
        self._listeners.append(fn)

    # ── Uyarı tetikleme ──

    def fire(self, code: str, level: AlertLevel, message: str) -> bool:
        now = time.time()
        if now - self._last_fired.get(code, 0) < self._throttle:
            return False
        self._last_fired[code] = now
        alert = Alert(code=code, level=level, message=message, ts=now)
        self._history.append(alert)
        if len(self._history) > self._max_hist:
            self._history.pop(0)
        for fn in self._listeners:
            try:
                fn(alert)
            except Exception:
                pass
        return True

    def clear_code(self, code: str):
        self._last_fired.pop(code, None)

    # ── Hazır koşul kontrolleri ──

    def check_occupancy(self, available: int, occupied: int):
        total = available + occupied
        if total == 0:
            return
        pct = occupied / total * 100
        if available == 0:
            self.fire(
                "park_full",
                AlertLevel.CRITICAL,
                "Park alani tamamen dolu! Bos yer kalmadi.",
            )
        elif pct >= 80:
            self.fire(
                "park_high",
                AlertLevel.WARNING,
                f"Park alani %{pct:.0f} dolu. ({available} bos yer kaldi)",
            )
        else:
            self._last_fired.pop("park_full", None)
            self._last_fired.pop("park_high", None)

    def check_forbidden(self, forbidden_count: int):
        if forbidden_count > 0:
            self.fire(
                "forbidden_park",
                AlertLevel.WARNING,
                f"Yasak bolgede {forbidden_count} arac tespit edildi!",
            )

    def check_no_fit(self, fit_count: int, total_empty: int):
        if total_empty > 0 and fit_count == 0:
            self.fire(
                "no_fit",
                AlertLevel.INFO,
                "Araciniz mevcut bos alanlarin hicbirine sigmiyor.",
            )
        else:
            self._last_fired.pop("no_fit", None)
