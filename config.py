"""Konfiguráció betöltése (config.json).

A config.json a felhasználó saját fájlja, nem kerül a repóba. Ha nincs, a
config.example.json-ból (vagy az alábbi alapértékekből) jön létre.
API-kulcsot a fájl nem tárol: a MiniMax-kulcs környezeti változóból jön.
"""
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
EXAMPLE_PATH = APP_DIR / "config.example.json"

DEFAULTS = {
    # Gyorsbillentyű a "keyboard" csomag szintaxisával (pl. "ctrl+windows", "f9", "right shift")
    "hotkey": "ctrl+windows",
    # "hold": amíg nyomva tartod, addig vesz fel | "toggle": egy lenyomás indít, még egy leállít
    "mode": "hold",
    # Diktálás nyelve (ISO-kód). Üres string = automatikus nyelvfelismerés.
    "language": "hu",
    # Átíró háttér: "auto" | "openvino" | "faster-whisper" | "whispercpp"
    "backend": "auto",
    # Eszköz: "auto" | "gpu" | "cpu", vagy a háttér saját neve ("GPU.1", "cuda:1", "gpu:1")
    "device": "auto",
    # Whisper-modell rövid neve: "large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"
    "model": "large-v3",
    # Processzorszálak száma CPU-s futásnál (0 = automatikus)
    "threads": 0,
    # Háttérenkénti finomhangolás. Az üres "model" mező a fenti közös "model"-t használja.
    "openvino": {
        # Hugging Face repó vagy helyi mappa, pl. "OpenVINO/whisper-large-v3-int8-ov"
        "model": "",
    },
    "faster_whisper": {
        # Modellnév vagy helyi CTranslate2-mappa, pl. "large-v3"
        "model": "",
        # "auto": GPU-n float16, CPU-n int8. Lehet még: "int8_float16", "int8", "float32"
        "compute_type": "auto",
    },
    "whispercpp": {
        # A whisper-server(.exe) útja. Üres = bin/whispercpp/whisper-server.exe, majd a PATH.
        "binary": "",
        # ggml-modellfájl neve vagy útja, pl. "ggml-large-v3-q5_0.bin"
        "model": "",
        # További parancssori kapcsolók a whisper-servernek, pl. ["--beam-size", "5"]
        "extra_args": [],
    },
    # Maximális felvételhossz másodpercben (biztonsági korlát toggle módhoz)
    "max_seconds": 120,
    "audio": {
        # A gombnyomás ELŐTTI ennyi másodperc is bekerül a felvételbe (nem vész el az első szótag)
        "preroll_seconds": 0.6,
        # Elengedés után ennyit várunk a hangkártya pufferére (nem vész el az utolsó szótag)
        "tail_seconds": 0.25,
        # A mikrofon folyamatosan nyitva marad (ez kell a preroll-hoz)
        "keep_mic_open": True,
        # Halk felvétel automatikus felerősítése
        "normalize": True,
    },
    # Az előző mondatok átadása a Whispernek kontextusként (méréseink szerint inkább ront)
    "whisper_context": False,
    # Az utolsó N felvétel mentése a recordings mappába hibakereséshez (0 = ki)
    "save_recordings": 0,
    # Opcionális AI-utójavítás (MiniMax, felhős). Alapból kikapcsolva.
    "ai_correction": False,
    "minimax": {
        # A kulcsot ebből a környezeti változóból olvassuk, a fájlba ne írd bele
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/anthropic",
        "model": "MiniMax-M3",
        # "anthropic" (MiniMax) | "openai" (más, OpenAI-kompatibilis szolgáltató)
        "api_style": "anthropic",
        "timeout": 45,
        # "balanced": szócsere is mehet, korlátozottan | "punctuation_only": csak írásjel/ékezet
        "mode": "balanced",
        # Ha a javítás a szavak ennél nagyobb részét átírja, eldobjuk
        "max_word_change": 0.25,
    },
    "output": {
        # "paste": vágólap + Ctrl+V (gyors) | "type": karakterenkénti gépelés (lassabb)
        "method": "paste",
        # Beillesztés után visszaállítsa-e az eredeti vágólap-tartalmat
        "restore_clipboard": False,
        # Szóköz a beillesztett szöveg végére, hogy a folytatás ne ragadjon hozzá
        "append_space": True,
    },
    # Gyakran elrontott nevek/szakszavak: a Whisper súgásába és az AI-javítóba kerülnek
    "vocabulary": [],
    # Mikrofon eszköz indexe vagy neve (null = alapértelmezett). Listázás: python -m sounddevice
    "audio_device": None,
    # Hangjelzés felvétel indításakor/leállításakor
    "beep": True,
}


def _merge(defaults: dict, loaded: dict) -> dict:
    out = dict(defaults)
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load(path: Path | None = None) -> dict:
    path = path or CONFIG_PATH
    if path.exists():
        cfg = _merge(DEFAULTS, _read(path))
    else:
        cfg = _merge(DEFAULTS, _read(EXAMPLE_PATH)) if EXAMPLE_PATH.exists() else dict(DEFAULTS)
        try:
            save(cfg, path)
        except OSError:
            pass
    env_name = cfg["minimax"].get("api_key_env") or "MINIMAX_API_KEY"
    cfg["minimax"]["api_key"] = os.environ.get(env_name, "")
    return cfg


def save(cfg: dict, path: Path | None = None) -> None:
    data = json.loads(json.dumps(cfg))
    data.get("minimax", {}).pop("api_key", None)  # a kulcs sosem kerül fájlba
    with open(path or CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
