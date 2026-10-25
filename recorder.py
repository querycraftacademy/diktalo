"""Mikrofon-felvétel 16 kHz mono formátumban (a Whisper ezt várja).

A mikrofon folyamatosan nyitva van, és egy körpuffer mindig tartja az utolsó
néhány tized másodpercet. Így a felvétel a gombnyomás ELŐTTI pillanatokat is
tartalmazza, és nem veszik el az első szótag. Elengedéskor rövid ideig még
várunk, hogy a hangkártya által pufferelt vég is beérkezzen.
"""
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


def level_db(audio: np.ndarray) -> tuple[float, float]:
    """(csúcs, effektív) szint dBFS-ben – a hangerő ellenőrzéséhez."""
    if len(audio) == 0:
        return (-99.0, -99.0)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    to_db = lambda x: float(20 * np.log10(max(x, 1e-6)))
    return (round(to_db(peak), 1), round(to_db(rms), 1))


def normalize(audio: np.ndarray, target_peak: float = 0.85, max_gain: float = 8.0) -> np.ndarray:
    """Halk felvétel felerősítése. A zajt is erősíti, ezért a nyereség korlátos."""
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4 or peak >= target_peak:
        return audio
    return audio * min(target_peak / peak, max_gain)


def device_name(device=None) -> str:
    """A használt bemeneti eszköz neve (None = a rendszer alapértelmezettje)."""
    try:
        if device is None:
            device = sd.default.device[0]
        return str(sd.query_devices(device)["name"])
    except Exception as e:
        return f"(ismeretlen eszköz: {e})"


# Ezek a nevek kameramikrofonra utalnak: távol vesznek, és nagyon rossz az átírás.
SUSPICIOUS = ("cam", "webcam", "logitech", "hd pro", "usb video", "kamera")


def looks_like_webcam(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SUSPICIOUS)


class Recorder:
    def __init__(
        self,
        device=None,
        max_seconds: int = 120,
        preroll_seconds: float = 0.6,
        tail_seconds: float = 0.25,
        keep_mic_open: bool = True,
    ):
        self.device = device
        self.max_seconds = max_seconds
        self.tail_seconds = tail_seconds
        self.keep_mic_open = keep_mic_open
        self._pre_len = max(0, int(preroll_seconds * SAMPLE_RATE))
        self._pre = np.zeros(self._pre_len, dtype=np.float32)
        self._pre_pos = 0
        self._pre_filled = 0
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.recording = False

    # ---------- belső ----------

    def _write_preroll(self, chunk: np.ndarray):
        n = len(chunk)
        if self._pre_len == 0 or n == 0:
            return
        if n >= self._pre_len:
            self._pre[:] = chunk[-self._pre_len:]
            self._pre_pos = 0
            self._pre_filled = self._pre_len
            return
        end = self._pre_pos + n
        if end <= self._pre_len:
            self._pre[self._pre_pos:end] = chunk
        else:
            k = self._pre_len - self._pre_pos
            self._pre[self._pre_pos:] = chunk[:k]
            self._pre[: n - k] = chunk[k:]
        self._pre_pos = end % self._pre_len
        self._pre_filled = min(self._pre_len, self._pre_filled + n)

    def _preroll_snapshot(self) -> np.ndarray:
        if self._pre_filled == 0:
            return np.zeros(0, dtype=np.float32)
        ordered = np.concatenate((self._pre[self._pre_pos:], self._pre[: self._pre_pos]))
        return ordered[-self._pre_filled:].copy()

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        with self._lock:
            if self.recording:
                self._frames.append(chunk)
                if sum(len(f) for f in self._frames) >= self.max_seconds * SAMPLE_RATE:
                    self.recording = False
            else:
                self._write_preroll(chunk)

    def _ensure_stream(self):
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def _close_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._pre_filled = 0
            self._pre_pos = 0

    # ---------- nyilvános ----------

    def open(self):
        """Mikrofon megnyitása induláskor, hogy a körpuffer feltöltődjön."""
        if self.keep_mic_open:
            self._ensure_stream()

    def close(self):
        self._close_stream()

    def start(self):
        self._ensure_stream()
        with self._lock:
            self._frames = [self._preroll_snapshot()] if self._pre_len else []
            self.recording = True

    def stop(self) -> np.ndarray:
        # A hangkártya néhány tized másodpercet puffereli: várjuk be a végét.
        if self.tail_seconds:
            time.sleep(self.tail_seconds)
        with self._lock:
            self.recording = False
            audio = (
                np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)
            )
            self._frames = []
        if not self.keep_mic_open:
            self._close_stream()
        return audio
