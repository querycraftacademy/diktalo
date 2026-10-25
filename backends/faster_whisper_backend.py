"""Whisper faster-whisperrel (CTranslate2): NVIDIA GPU (CUDA), AMD GPU (ROCm-es
CTranslate2-builddel) és processzor."""
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models" / "faster-whisper"


def _split_device(device: str) -> tuple[str, int]:
    dev = (device or "cpu").lower()
    if ":" in dev:
        name, idx = dev.split(":", 1)
        return name, int(idx or 0)
    return dev, 0


class FasterWhisperTranscriber:
    def __init__(self, model: str, device: str = "cpu", compute_type: str = "auto", threads: int = 0):
        from hardware import add_nvidia_dll_dirs

        add_nvidia_dll_dirs()  # cuBLAS/cuDNN DLL-ek a pip-csomagokból (Windows)
        from faster_whisper import WhisperModel

        dev, idx = _split_device(device)
        if compute_type in ("", "auto", None):
            compute_type = "float16" if dev == "cuda" else "int8"
        self.device = dev
        self.name = f"faster-whisper {device} ({compute_type}), {model}"
        t0 = time.time()
        print(f"Whisper betöltése (faster-whisper, {device}, {compute_type}) ...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = WhisperModel(
            model,
            device=dev,
            device_index=idx,
            compute_type=compute_type,
            cpu_threads=int(threads or 0),
            download_root=str(MODELS_DIR),
        )
        # Bemelegítés egy fél másodperc csenddel: a hiányzó cuDNN/cuBLAS DLL sokszor csak
        # az első átíráskor derül ki. Így még a választáskor kiderül, és jöhet a tartalék.
        self._run(np.zeros(8000, dtype=np.float32), "en", None)
        print(f"Whisper betöltve {time.time() - t0:.1f} s alatt.")

    def _run(self, audio, language, initial_prompt) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=language or None,
            task="transcribe",
            initial_prompt=initial_prompt or None,
            # GPU-n belefér a beam search, processzoron a mohó dekódolás jóval gyorsabb
            beam_size=5 if self.device == "cuda" else 1,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe(self, audio: np.ndarray, language: str = "hu", initial_prompt: str | None = None) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return self._run(audio, language, initial_prompt)

    def close(self):
        self.model = None
