"""Hangfájl átírása és hardver-felismerés a parancssorból (mikrofon és gyorsbillentyű nélkül).

Példák:
  python transcribe_file.py --detect                      # mit lát a gép, mit választana (dry-run)
  python transcribe_file.py felvetel.wav                  # átírás a config szerinti háttérrel
  python transcribe_file.py felvetel.wav --backend openvino --device cpu --model small
"""
import argparse
import sys
import time
import wave

import numpy as np

import config as config_mod
import hardware
from backends import create_transcriber, plan

SAMPLE_RATE = 16000


def read_wav(path: str) -> np.ndarray:
    """16 bites PCM WAV beolvasása 16 kHz mono float32-be."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("Csak 16 bites PCM WAV-ot tudok olvasni.")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", help="16 bites PCM WAV-fájl")
    ap.add_argument("--detect", action="store_true", help="csak hardver-felismerés és választási terv")
    ap.add_argument("--backend", help="auto | openvino | faster-whisper | whispercpp")
    ap.add_argument("--device", help="auto | gpu | cpu | háttér saját eszközneve")
    ap.add_argument("--model", help="pl. large-v3, large-v3-turbo, medium, small, base")
    ap.add_argument("--language", help="pl. hu (üres = automatikus)")
    ap.add_argument("--threads", type=int, help="CPU-szálak száma (0 = automatikus)")
    args = ap.parse_args(argv)

    cfg = config_mod.load()
    for key in ("backend", "device", "model", "language", "threads"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.model:  # a parancssori modell felülírja a háttérenkénti beállítást is
        for section in ("openvino", "faster_whisper", "whispercpp"):
            cfg[section]["model"] = ""

    hw = hardware.detect(cfg)
    print(hardware.describe(hw))
    for note in hw.notes:
        print(f"Tipp: {note}")
    candidates, warnings = plan(hw, cfg["backend"], cfg["device"])
    for w in warnings:
        print(f"FIGYELEM: {w}")
    print(f"Választási sorrend (backend={cfg['backend']}, device={cfg['device']}):")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c.backend:<15} {c.device:<6} {c.reason}")
    if args.detect:
        return 0
    if not args.wav:
        ap.error("add meg a WAV-fájlt, vagy használd a --detect kapcsolót")

    audio = read_wav(args.wav)
    print(f"\nHang: {args.wav} ({len(audio) / SAMPLE_RATE:.1f} s)")
    t0 = time.time()
    tr = create_transcriber(cfg, hw=hw)
    t_load = time.time() - t0
    try:
        t0 = time.time()
        text = tr.transcribe(audio, cfg["language"])
        t_stt = time.time() - t0
    finally:
        close = getattr(tr, "close", None)
        if close:
            close()
    print(f"\nHáttér: {tr.candidate.backend} / {tr.candidate.device}  (betöltés {t_load:.1f} s, átírás {t_stt:.1f} s)")
    print(f"Átirat:\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
