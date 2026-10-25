"""Közös modellnevek leképezése az egyes hátterek saját modelljeire.

A configban elég egy rövid nevet megadni ("large-v3"), és bármelyik háttér
fusson is végül, ugyanannak a Whisper-modellnek a neki való változatát kapja.
"""
from pathlib import Path

# Előre konvertált, int8-ra kvantált OpenVINO-modellek (Hugging Face, OpenVINO szervezet)
OPENVINO = {
    "tiny": "OpenVINO/whisper-tiny-int8-ov",
    "base": "OpenVINO/whisper-base-int8-ov",
    "small": "OpenVINO/whisper-small-int8-ov",
    "medium": "OpenVINO/whisper-medium-int8-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "turbo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
}

# A faster-whisper ezeket a neveket maga is ismeri (Systran / mobiuslabs CTranslate2-modellek)
FASTER_WHISPER = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
}

# ggml-modellek a ggerganov/whisper.cpp Hugging Face-repóból. A nagy modelleknél
# kvantált változat: a large-v3 q5_0 1,1 GB a 3,1 GB-os f16 helyett, alig rosszabb.
WHISPERCPP_REPO = "ggerganov/whisper.cpp"
WHISPERCPP = {
    "tiny": "ggml-tiny-q8_0.bin",
    "base": "ggml-base-q8_0.bin",
    "small": "ggml-small-q8_0.bin",
    "medium": "ggml-medium-q5_0.bin",
    "large-v3": "ggml-large-v3-q5_0.bin",
    "large-v3-turbo": "ggml-large-v3-turbo-q8_0.bin",
    "turbo": "ggml-large-v3-turbo-q8_0.bin",
}

TABLES = {"openvino": OPENVINO, "faster-whisper": FASTER_WHISPER, "whispercpp": WHISPERCPP}
CFG_SECTION = {"openvino": "openvino", "faster-whisper": "faster_whisper", "whispercpp": "whispercpp"}


def resolve(backend: str, cfg: dict) -> str:
    """A háttérnek átadandó modellazonosító.

    Sorrend: a háttér saját "model" mezője, majd a közös "model" rövid név
    leképezve, végül (ha nem ismert rövid név) a közös érték változatlanul,
    pl. egy helyi mappa vagy egy Hugging Face-repó neve.
    """
    own = (cfg.get(CFG_SECTION[backend]) or {}).get("model") or ""
    if own:
        return own
    name = (cfg.get("model") or "large-v3").strip()
    table = TABLES[backend]
    return table.get(name.lower(), name)


def is_local_path(value: str) -> bool:
    try:
        return Path(value).exists()
    except OSError:
        return False
