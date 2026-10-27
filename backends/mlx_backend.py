"""Whisper mlx-whisperrel: Apple Silicon (M1–M4) Mac GPU-ja, Metalon keresztül."""
import time
from pathlib import Path

import numpy as np

from .models import is_local_path

APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models" / "mlx"


def ensure_model(repo_or_path: str) -> Path:
    """Helyi mappa esetén azt adja vissza, különben letölti a Hugging Face-ről."""
    if is_local_path(repo_or_path):
        return Path(repo_or_path)
    target = MODELS_DIR / repo_or_path.replace("/", "__")
    if (target / "config.json").exists():
        return target
    from huggingface_hub import snapshot_download

    print(f"Modell letöltése: {repo_or_path} -> {target} ...")
    snapshot_download(repo_or_path, local_dir=str(target))
    print("Modell letöltve.")
    return target


class MlxTranscriber:
    def __init__(self, model: str, device: str = "gpu"):
        import mlx.core as mx
        import mlx_whisper

        if device.lower() == "cpu":
            mx.set_default_device(mx.cpu)
        self._mlx = mlx_whisper
        self.name = f"mlx-whisper {device}, {model}"
        self.model_dir = str(ensure_model(model))
        t0 = time.time()
        print(f"Whisper betöltése (mlx-whisper, {device}) ...")
        # Az mlx-whisper az első átíráskor tölti be a modellt: egy fél másodperc csenddel
        # még most betöltjük, így a hiba is a választáskor derül ki, és jöhet a tartalék.
        self._run(np.zeros(8000, dtype=np.float32), "en", None)
        print(f"Whisper betöltve {time.time() - t0:.1f} s alatt.")

    def _run(self, audio, language, initial_prompt) -> str:
        result = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.model_dir,
            language=language or None,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=False,
            verbose=None,
        )
        return str(result.get("text", "")).strip()

    def transcribe(self, audio: np.ndarray, language: str = "hu", initial_prompt: str | None = None) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return self._run(audio, language, initial_prompt)

    def close(self):
        self._mlx = None
