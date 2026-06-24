# Uygulama Planı: A1 · A4 · C1

---

## A1 — Monoküler Derinlik Tahmini (MiDaS)

### Neden Önemli?
Mevcut sistemde mesafeler tamamen piksel tabanlı heuristik. MiDaS entegrasyonu sonrasında:
- Araç/yaya mesafeleri gerçek metre cinsinden ekranda görünür.
- BSD uyarısı "yaya kör noktaya girdi" yerine **"3.2m'de yaya yaklaşıyor"** der.
- Park slotu boyutları piksel oranından değil metrik ölçümden hesaplanır.

### Kullanılacak Model
```
torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
```
- Boyut: ~14MB  
- CPU inference: ~40-80ms/frame  
- Her 5 karede bir çalıştır (caching ile FPS korunur)

### Yeni Dosya: `src/depth/depth_estimator.py`
```python
import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        self.transform = self.transforms.small_transform
        self.model.eval()
        self.available = True
        self._cache = None
        self._cache_tick = 0

    def infer(self, frame_bgr, tick=0, period=5):
        """Her `period` karede bir yeni derinlik hesaplar, aralarında cache döndürür."""
        if tick % period != 0 and self._cache is not None:
            return self._cache
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode='bicubic', align_corners=False
            ).squeeze()
        depth = pred.cpu().numpy()
        # Normalize 0-1
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
        self._cache = depth_norm
        return depth_norm

    def get_bbox_depth(self, depth_map, bbox):
        """Bir bbox'ın ortasındaki ortalama derinlik değerini döndürür (0-1 arası)."""
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        pad = 8
        roi = depth_map[max(0,cy-pad):cy+pad, max(0,cx-pad):cx+pad]
        return float(roi.mean()) if roi.size > 0 else 0.5

    def depth_to_colormap(self, depth_norm):
        """Derinlik haritasını görselleştirme için BGR colormapa çevirir."""
        d8 = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
```

### `main_window.py` Entegrasyonu

**`__init__`'e ekle:**
```python
try:
    from src.depth.depth_estimator import DepthEstimator
    self.depth_est = DepthEstimator()
except Exception:
    self.depth_est = None
self._depth_tick = 0
self._last_depth_map = None
self._depth_overlay_active = False
```

**Pipeline içinde (YOLO sonrası, çizimden önce):**
```python
if self.depth_est and self.depth_est.available:
    self._last_depth_map = self.depth_est.infer(enhanced_frame, self._depth_tick)
    self._depth_tick += 1
    if self._depth_overlay_active and self._last_depth_map is not None:
        colormap = self.depth_est.depth_to_colormap(self._last_depth_map)
        cv2.addWeighted(colormap, 0.45, out, 0.55, 0, out)
```

**Her YOLO bbox yazısına mesafe ekle:**
```python
if self._last_depth_map is not None:
    d_val = self.depth_est.get_bbox_depth(self._last_depth_map, det['bbox'])
    # Inversted depth → relative distance (yakın=düşük değer)
    rel_dist = 1.0 - d_val
    dist_label = f"~{rel_dist*25:.1f}m"  # kaba metrik tahmin
    cv2.putText(out, dist_label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
```

### UI Değişiklikleri
Sol panele "DERINLIK" bölümüne:
- `depth_toggle_btn` — Derinlik ısı haritasını overlay olarak aç/kapat (Viridis/Inferno colormap)
- `depth_lbl` — "Derinlik: Aktif / Kapalı" durum etiketi

### Sidebar Kart Stili
```python
# border: #7c3aed (mor/violet) — derinlik algısını temsil eder
"QFrame { background:#150a2a; border:1px solid #7c3aed; border-radius:8px; }"
```

---

## A4 — Çok Kriterli Akıllı Slot Seçim Motoru

### Neden Önemli?
Şu an sistem sadece "boş" slot gösteriyor. Bu motor şunu yapar:
**"Sürücünüz için en uygun slot: #3 (Puan: 91/100) · Geniş alan · Çıkışa yakın · Kolay manevra"**

### Skor Formülü
```python
def compute_slot_score(difficulty, distance_px, width_m, ref_width_m, slot_x, map_width):
    """
    difficulty   : 0-100 (mevcut sistem)
    distance_px  : ego araçtan slot merkezine piksel mesafe
    width_m      : slot genişliği metre
    ref_width_m  : sürücünün aracı genişliği
    slot_x       : slot'un x koordinatı (şematik haritada)
    map_width    : harita genişliği (çıkışa yakınlık için proxy)
    """
    w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15  # ağırlıklar

    norm_diff     = difficulty / 100.0
    norm_dist     = max(0.0, 1.0 - distance_px / 1000.0)
    width_margin  = min(1.0, (width_m - ref_width_m) / ref_width_m) if ref_width_m > 0 else 0.5
    exit_prox     = 1.0 - abs(slot_x - map_width) / map_width  # sağ taraf = çıkış varsayımı

    score = w1*norm_diff + w2*norm_dist + w3*width_margin + w4*exit_prox
    return int(score * 100)
```

### Öneri Gerekçe Metni
```python
def slot_reason_text(difficulty, width_margin, distance_px, exit_prox):
    parts = []
    if difficulty >= 75:   parts.append("Kolay manevra")
    elif difficulty >= 45: parts.append("Orta zorluk")
    else:                  parts.append("Dar alan")
    if width_margin > 0.3: parts.append("Geniş slot")
    if distance_px < 300:  parts.append("Yakın mesafe")
    if exit_prox > 0.7:    parts.append("Cikisa yakin")
    return " · ".join(parts) if parts else "Standart slot"
```

### `overlays.py` — Önerilen Slot Özel Çizimi
`render_full_schematic_map` içinde, en yüksek skorlu boş slota özel çizim:
```python
# En yüksek skorlu slotu bul
best_slot_i, best_score = -1, -1
for i, (_, is_occ, sz, fit, _) in enumerate(slots_list):
    if not is_occ and fit:
        d_px = abs((start_x + i*(slot_w+gap) + slot_w//2) - ego_x_schematic)
        w_m = sz[0] if sz else 2.5
        score = compute_slot_score(difficulties[i], d_px, w_m, 1.8,
                                   start_x + i*(slot_w+gap), width)
        if score > best_score:
            best_score, best_slot_i = score, i

# Önerilen slota altın pırıltılı çerçeve
if best_slot_i >= 0:
    bx1 = start_x + best_slot_i*(slot_w+gap) - 3
    bx2 = bx1 + slot_w + 6
    cv2.rectangle(panel, (bx1, slot_y-3), (bx2, slot_y+slot_h+3), (0, 215, 255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"ONERILEN  {best_score}/100",
                (bx1+4, slot_y-8), _FONT, 0.38, (0,215,255), 1, cv2.LINE_AA)
```

### `main_window.py` — Öneri Kartı (Status Bar Altı)
```python
# Her _update_schematic_map_ui sonrasında güncelle
self.recommendation_lbl.setText(
    f"⭐ ÖNERİLEN: SLOT {best_slot_i+1}  ·  Puan: {best_score}/100  ·  {reason_text}"
)
self.recommendation_lbl.setStyleSheet("color:#fbbf24; font-weight:bold; font-size:11px;")
```

### Ağırlık Ayarları (Gelecek genişleme için)
`config.json`'a kaydet:
```json
{
  "slot_score_weights": {
    "difficulty": 0.40,
    "proximity": 0.25,
    "width_margin": 0.20,
    "exit_proximity": 0.15
  }
}
```

---

## C1 — Çevrimdışı Sesli Asistan (Vosk)

### Neden Önemli?
Jüri sunumunda mikrofona "En yakın boş yeri bul" diyerek AVP simülasyonunun
otonom başladığını görmek çarpıcı bir final sahnesidir.

### Gereksinimler
```
pip install vosk sounddevice
```
Türkçe model indir (~50MB): `vosk-model-small-tr-0.3`  
URL: `https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip`

### Yeni Dosya: `src/voice/voice_assistant.py`
```python
import json
import queue
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer

COMMANDS = {
    "bos yer": "find_empty",
    "en yakin": "find_empty",
    "demo basla": "auto_demo",
    "gece": "toggle_night",
    "slam": "toggle_slam",
    "sifirla": "reset_slam",
    "bsd ac": "toggle_bsd",
    "derin": "toggle_depth",
    "dur": "stop_sim",
}

class VoiceAssistant:
    def __init__(self, model_path: str, callback, sample_rate=16000):
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, sample_rate)
        self.callback = callback  # Qt sinyal/slot veya callable
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _audio_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def _listen_loop(self):
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=8000,
                               dtype='int16', channels=1,
                               callback=self._audio_callback):
            while self._running:
                data = self.q.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").lower()
                    for keyword, cmd in COMMANDS.items():
                        if keyword in text:
                            self.callback(cmd, text)
                            break
```

### `main_window.py` Entegrasyonu

**`__init__`'e ekle:**
```python
self._voice_active = False
self.voice_assistant = None
```

**Sesli komut handler:**
```python
def _on_voice_command(self, cmd: str, raw_text: str):
    """VoiceAssistant thread'inden gelen komutları Qt main thread'ine ilet."""
    from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
    QMetaObject.invokeMethod(self, "_execute_voice_cmd",
                             Qt.QueuedConnection,
                             Q_ARG(str, cmd))

@pyqtSlot(str)
def _execute_voice_cmd(self, cmd: str):
    self.status_lbl.setText(f"🎤 Sesli Komut: {cmd}")
    if cmd == "find_empty":
        self._start_auto_demo()
    elif cmd == "auto_demo":
        self._start_auto_demo()
    elif cmd == "toggle_night":
        self.night_btn.click()
    elif cmd == "toggle_slam":
        self.slam_toggle_btn.click()
    elif cmd == "reset_slam":
        self._reset_slam_map()
    elif cmd == "toggle_bsd":
        self.bsd_toggle_btn.click()
    elif cmd == "toggle_depth":
        if hasattr(self, 'depth_toggle_btn'):
            self.depth_toggle_btn.click()
    elif cmd == "stop_sim":
        if self._sim_active:
            self._sim_timer.stop()
            self._sim_active = False

def _toggle_voice_assistant(self):
    if not self._voice_active:
        model_path = "models/vosk-model-small-tr-0.3"
        try:
            self.voice_assistant = VoiceAssistant(model_path, self._on_voice_command)
            self.voice_assistant.start()
            self._voice_active = True
            self.voice_toggle_btn.setText("🎤 Ses KAPAT")
            self.voice_toggle_btn.setStyleSheet(
                "background-color:#059669; color:white; font-weight:bold; border-radius:8px;"
            )
            self.status_lbl.setText("🎤 Sesli Asistan AÇık — Komut bekliyor...")
        except Exception as e:
            self.status_lbl.setText(f"Ses hatası: {e}")
    else:
        if self.voice_assistant:
            self.voice_assistant.stop()
        self._voice_active = False
        self.voice_toggle_btn.setText("🎤 Ses AÇ")
        self.voice_toggle_btn.setStyleSheet(
            "background-color:#475569; color:white; font-weight:bold; border-radius:8px;"
        )
```

### UI — Sesli Asistan Kartı (Sidebar)
```python
# ── Sesli Asistan ──
panel.addWidget(make_section_label("SESLI ASISTAN"))

voice_box = QFrame()
voice_box.setStyleSheet(
    "QFrame { background:#0a1a0f; border:1px solid #059669; border-radius:8px; }"
)
voice_layout = QVBoxLayout(voice_box)
voice_layout.setContentsMargins(8, 8, 8, 8)
voice_layout.setSpacing(6)

voice_desc = QLabel(
    "Sesli komutlar:\n"
    "• 'En yakın boş yer' → AVP\n"
    "• 'Gece görüşü aç'\n"
    "• 'SLAM sıfırla'\n"
    "• 'Demo başla'"
)
voice_desc.setStyleSheet("color:#94a3b8; font-size:9px; background:transparent; border:none;")
voice_layout.addWidget(voice_desc)

self.voice_toggle_btn = self._btn("🎤 Ses AÇ", "#475569")
self.voice_toggle_btn.setFixedHeight(30)
self.voice_toggle_btn.setCheckable(True)
self.voice_toggle_btn.clicked.connect(self._toggle_voice_assistant)
voice_layout.addWidget(self.voice_toggle_btn)

panel.addWidget(voice_box)
```

### Ekran HUD — Dinleme Göstergesi
`_show_frame` içinde, `_voice_active` True ise:
```python
if self._voice_active:
    cv2.circle(out, (out.shape[1]-30, 30), 10, (0,200,80), -1, cv2.LINE_AA)
    cv2.putText(out, "MIC", (out.shape[1]-52, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,80), 1, cv2.LINE_AA)
```

---

## Uygulama Sırası

```
1. A4 (Çok Kriterli Slot) → En kolay, saf Python matematik, dış bağımlılık yok.
2. A1 (MiDaS Derinlik)    → pip install torch (zaten kurulu olabilir) + hub.load.
3. C1 (Vosk Sesli)        → pip install vosk sounddevice + model indir.
```

## Bağımlılıklar

```bash
# A1 - MiDaS (torch zaten kuruluysa sadece model indirilir)
pip install timm  # MiDaS için gerekli

# C1 - Vosk
pip install vosk sounddevice

# Model dosyası (~50MB) - bir kez indir
# https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip
# Çıkart: models/vosk-model-small-tr-0.3/
```
