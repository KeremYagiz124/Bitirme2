"""Çevrimdışı sesli asistan — Vosk (internet gerektirmez).

Jüri sunumunda mikrofona "en yakın boş yeri bul" denildiğinde sistemin otonom
tepki vermesi çarpıcı bir final sahnesidir.

Tasarım kararları:
  - vosk/sounddevice veya model yoksa `available=False` (graceful; UI uyarır).
  - Komut eşleştirme saf fonksiyondur (match_command) → bağımsız test edilir.
  - Yalnızca UYGULAMADA GERÇEKTEN VAR OLAN aksiyonlara komut tanımlanır;
    olmayan özelliklere bağlanmaz (çağıran tarafta bilinmeyen komut no-op).
  - Dinleme ayrı thread'de; komut callback ile iletilir (çağıran Qt ana
    thread'ine marshal etmelidir).
  - TTS: edge-tts (internet varsa, Microsoft neural Türkçe) → pyttsx3 fallback.
    Speak non-blocking; önceki konuşma varsa kesilir.
"""

from __future__ import annotations

import io
import os
import threading

# Anahtar kelime → komut. Yalnızca mevcut aksiyonlar.
COMMANDS = {
    "bos yer":   "find_empty",
    "en yakin":  "find_empty",
    "bos":       "find_empty",
    "izgara":    "toggle_adaptive",
    "adaptif":   "toggle_adaptive",
    "derinlik":  "toggle_depth",
    "isi":       "toggle_depth",
    "gece":      "toggle_night",
    "kus bakisi": "toggle_ipm",
    "harita":    "toggle_ipm",
    "degerlendir": "evaluate",
    "dur":       "stop",
    "kapat":     "stop",
}


CMD_RESPONSES: dict[str, str] = {
    "find_empty":      "Sizin için en yakın boş park yerini buldum, yönlendiriyorum.",
    "toggle_adaptive": "Adaptif çizgi ve ızgara tabanlı park yeri tespit modu değiştirildi.",
    "toggle_depth":    "Derinlik ve otopark yoğunluk analiz katmanı değiştirildi.",
    "toggle_night":    "Gece görüşü iyileştirme filtresi güncellendi.",
    "toggle_ipm":      "Ters perspektif dönüşümü ile kuş bakışı görünüm katmanı değiştirildi.",
    "evaluate":        "Sistem başarı ve doğruluk analizi başlatılıyor.",
    "stop":            "Sesli asistan kapatıldı, güvenli sürüşler dilerim.",
}


class TTSSpeaker:
    """Komut yanıtlarını seslendirir. edge-tts (neural) → pyttsx3 fallback."""

    VOICE = "tr-TR-EmelNeural"

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pygame = None
        self._pyttsx3_engine = None
        self._backend: str = "none"
        self._init_backend()

    def _init_backend(self):
        try:
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
            self._backend = "edge"
        except Exception:
            pass
        if self._backend == "none":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                for v in engine.getProperty("voices"):
                    if "tr" in (v.id or "").lower() or "tolga" in (v.name or "").lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.setProperty("rate", 160)
                self._pyttsx3_engine = engine
                self._backend = "pyttsx3"
            except Exception:
                self._backend = "none"

    def speak(self, text: str):
        if self._backend == "none":
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return  # önceki bitmemişse atla (kalabalık etmesin)
            self._thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
            self._thread.start()

    def _speak_worker(self, text: str):
        if self._backend == "edge":
            self._speak_edge(text)
        elif self._backend == "pyttsx3":
            self._speak_pyttsx3(text)

    def _speak_edge(self, text: str):
        try:
            import asyncio
            import edge_tts

            async def _run():
                communicate = edge_tts.Communicate(text, self.VOICE)
                buf = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                buf.seek(0)
                return buf

            buf = asyncio.run(_run())
            pygame = self._pygame
            if pygame and buf.getbuffer().nbytes > 0:
                pygame.mixer.music.load(buf, "mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    threading.Event().wait(0.05)
        except Exception:
            # internet yoksa pyttsx3'e düş
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str):
        try:
            if self._pyttsx3_engine:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
        except Exception:
            pass


def match_command(text: str, commands: dict | None = None) -> str | None:
    """Tanınan metinde anahtar kelime ara, eşleşen komutu döndür (yoksa None)."""
    commands = commands or COMMANDS
    t = (text or "").lower()
    
    # Türkçe karakterleri ASCII'ye normalize et (hem tanınan metin hem anahtar kelimeler için)
    translation_table = str.maketrans("çğışöüı", "cgisoui")
    t_norm = t.translate(translation_table)
    
    for kw, cmd in commands.items():
        kw_norm = kw.lower().translate(translation_table)
        if kw_norm in t_norm:
            return cmd
    return None


class VoiceAssistant:
    def __init__(self, model_path: str, callback, sample_rate: int = 16000):
        self.model_path = model_path
        self.callback = callback
        self.sample_rate = sample_rate
        self.available = False
        self._running = False
        self._thread = None
        self._sd = None
        self._rec = None
        self._q = None

        if not model_path or not os.path.isdir(model_path):
            return
        try:
            import queue
            import vosk
            import sounddevice
            self._sd = sounddevice
            self._model = vosk.Model(model_path)
            self._rec = vosk.KaldiRecognizer(self._model, sample_rate)
            self._q = queue.Queue()
            self.available = True
        except Exception:
            self.available = False

    def start(self) -> bool:
        if not self.available or self._running:
            return False
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        if self._q is not None:
            self._q.put(bytes(indata))

    def _listen_loop(self):
        import json
        try:
            with self._sd.RawInputStream(
                    samplerate=self.sample_rate, blocksize=8000, dtype="int16",
                    channels=1, callback=self._audio_callback):
                while self._running:
                    data = self._q.get()
                    if self._rec.AcceptWaveform(data):
                        text = json.loads(self._rec.Result()).get("text", "")
                        cmd = match_command(text)
                        if cmd:
                            try:
                                self.callback(cmd, text)
                            except Exception:
                                pass
        except Exception:
            self._running = False
