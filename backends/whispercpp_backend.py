"""Whisper a whisper.cpp szerverével (whisper-server), alfolyamatként.

A whisper-server egyszer tölti be a modellt, utána minden diktálás egy helyi
HTTP-kérés (127.0.0.1). A GPU-t a bináris dönti el: egy Vulkanos build
(ggml-vulkan.dll / libggml-vulkan.so) AMD-, Intel- és NVIDIA-kártyán is fut,
macOS-en a Metal az Apple GPU-n, a CPU-s build csak processzoron. A "cpu"
eszköz a --no-gpu kapcsolót adja át.
"""
import atexit
import io
import os
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .models import WHISPERCPP_REPO, is_local_path

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models" / "whispercpp"
LOG_PATH = APP_DIR / "whispercpp-server.log"
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def ensure_model(name_or_path: str) -> Path:
    if is_local_path(name_or_path):
        return Path(name_or_path)
    target = MODELS_DIR / name_or_path
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download

    print(f"Modell letöltése: {WHISPERCPP_REPO}/{name_or_path} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(WHISPERCPP_REPO, name_or_path, local_dir=str(MODELS_DIR))
    print("Modell letöltve.")
    return Path(path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_command(binary: str, model: Path, port: int, device: str, language: str,
                  threads: int, extra_args: list) -> list[str]:
    cmd = [
        str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-l", language or "auto", "-t", str(threads), "-nt",
    ]
    dev = (device or "gpu").lower()
    if dev == "cpu":
        cmd.append("-ng")
    elif dev.startswith("gpu:"):
        cmd += ["-dev", dev.split(":", 1)[1]]
    cmd += [str(a) for a in extra_args]
    return cmd


class WhisperCppTranscriber:
    def __init__(self, binary: str | None, model: str, device: str = "gpu", language: str = "hu",
                 threads: int = 0, extra_args: list | None = None, startup_timeout: float = 300):
        import requests

        if not binary or not Path(binary).is_file():
            raise FileNotFoundError(
                "whisper-server nem található (bin/whispercpp/ mappa vagy config: whispercpp.binary)"
            )
        self._requests = requests
        model_path = ensure_model(model)
        threads = threads or max(1, min(8, (os.cpu_count() or 4)))
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.name = f"whisper.cpp {device}, {model_path.name}"
        cmd = build_command(binary, model_path, self.port, device, language, threads, extra_args or [])
        print(f"whisper-server indítása ({device}) ...")
        t0 = time.time()
        self._log = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(Path(binary).parent), creationflags=NO_WINDOW,
        )
        atexit.register(self.close)
        while True:
            if self.proc.poll() is not None:
                self.close()
                raise RuntimeError(f"a whisper-server leállt indulás közben. Napló: {LOG_PATH}\n{self._tail()}")
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - t0 > startup_timeout:
                self.close()
                raise TimeoutError(f"a whisper-server {startup_timeout:.0f} s alatt sem indult el. Napló: {LOG_PATH}")
            time.sleep(0.3)
        self.gpu_info = self._gpu_line()
        print(f"whisper-server kész {time.time() - t0:.1f} s alatt." + (f" ({self.gpu_info})" if self.gpu_info else ""))

    def _tail(self, n: int = 15) -> str:
        try:
            lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def _gpu_line(self) -> str:
        """A naplóból kiolvassa, melyik GPU-t használja (ha egyáltalán)."""
        try:
            for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
                low = line.lower()
                if ("vulkan" in low or "cuda" in low) and ("device" in low or "using" in low):
                    return line.strip()[:160]
        except OSError:
            pass
        return ""

    def transcribe(self, audio: np.ndarray, language: str = "hu", initial_prompt: str | None = None) -> str:
        data = {"response_format": "json", "temperature": "0.0", "language": language or "auto"}
        if initial_prompt:
            data["prompt"] = initial_prompt
        resp = self._requests.post(
            f"{self.url}/inference",
            files={"file": ("audio.wav", wav_bytes(audio), "audio/wav")},
            data=data,
            timeout=600,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "")
        return " ".join(part.strip() for part in text.splitlines() if part.strip())

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()
