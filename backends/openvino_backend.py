"""Whisper OpenVINO GenAI-jal: Intel GPU (Arc, Iris Xe) és bármilyen x86 processzor."""
import os
import time
from pathlib import Path

# A oneDNN kilépéskor ártalmatlan hibalogokat szórhat a konzolra – elnémítjuk.
os.environ.setdefault("ONEDNN_VERBOSE", "0")

import numpy as np

from .models import is_local_path

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models" / "openvino"
CACHE_DIR = APP_DIR / ".ov_cache"


def ensure_model(repo_or_path: str) -> Path:
    """Helyi mappa esetén azt adja vissza, különben letölti a Hugging Face-ről."""
    if is_local_path(repo_or_path):
        return Path(repo_or_path)
    target = MODELS_DIR / repo_or_path.replace("/", "__")
    if (target / "openvino_encoder_model.xml").exists():
        return target
    from huggingface_hub import snapshot_download

    print(f"Modell letöltése: {repo_or_path} -> {target} ...")
    snapshot_download(repo_or_path, local_dir=str(target))
    print("Modell letöltve.")
    return target


class OpenVinoTranscriber:
    def __init__(self, model: str, device: str = "CPU", threads: int = 0):
        import openvino_genai as ov_genai

        self.name = f"OpenVINO {device}, {model}"
        model_dir = ensure_model(model)
        CACHE_DIR.mkdir(exist_ok=True)
        props = {"CACHE_DIR": str(CACHE_DIR)}
        if device.upper() == "CPU" and threads > 0:
            props["INFERENCE_NUM_THREADS"] = int(threads)
        t0 = time.time()
        print(f"Whisper betöltése (OpenVINO, {device}) ... első alkalommal 1-2 percig is tarthat.")
        try:
            self.pipe = ov_genai.WhisperPipeline(str(model_dir), device, **props)
        except Exception as first:
            if device.upper().startswith("GPU"):
                raise  # a GPU-hibát továbbadjuk, hogy a választó a következő jelöltre lépjen
            try:
                self.pipe = ov_genai.WhisperPipeline(str(model_dir), device)
            except Exception:
                raise first
        print(f"Whisper betöltve {time.time() - t0:.1f} s alatt.")

    def transcribe(self, audio: np.ndarray, language: str = "hu", initial_prompt: str | None = None) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        kwargs = {"task": "transcribe", "return_timestamps": False}
        if language:
            kwargs["language"] = f"<|{language}|>"
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        result = self.pipe.generate(audio.tolist(), **kwargs)
        return str(result).strip()

    def close(self):
        self.pipe = None
