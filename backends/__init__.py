"""Az átíró háttér kiválasztása és betöltése.

A plan() egy tiszta függvény: a felismert hardverből és a configból egy
rangsorolt jelöltlistát ad (háttér + eszköz). A create_transcriber() sorban
megpróbálja betölteni őket; ha egy nem indul (hiányzó meghajtó, kevés memória,
hiányzó DLL), a következőre lép, a lista végén mindig a processzor áll.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

BACKENDS = ("openvino", "faster-whisper", "whispercpp")
_ALIASES = {
    "auto": "auto",
    "openvino": "openvino",
    "ov": "openvino",
    "faster-whisper": "faster-whisper",
    "faster_whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "ctranslate2": "faster-whisper",
    "whispercpp": "whispercpp",
    "whisper.cpp": "whispercpp",
    "whisper-cpp": "whispercpp",
}


@dataclass(frozen=True)
class Candidate:
    backend: str  # "openvino" | "faster-whisper" | "whispercpp"
    device: str   # a háttér saját eszközneve: "GPU"/"GPU.1"/"CPU", "cuda"/"cuda:1"/"cpu", "gpu"/"gpu:1"/"cpu"
    reason: str

    def same_as(self, other: "Candidate") -> bool:
        return self.backend == other.backend and self.device.lower() == other.device.lower()

    @property
    def is_cpu(self) -> bool:
        return self.device.lower() == "cpu"


def normalize_backend(value: str) -> str:
    key = (value or "auto").strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Ismeretlen háttér: {value!r} (lehet: auto, {', '.join(BACKENDS)})")
    return _ALIASES[key]


def _installed(hw, backend: str) -> bool:
    return {
        "openvino": hw.openvino,
        "faster-whisper": hw.faster_whisper,
        "whispercpp": bool(hw.whispercpp_binary),
    }[backend]


def _cpu_chain(hw) -> list[Candidate]:
    chain = []
    if hw.openvino:
        chain.append(Candidate("openvino", "CPU", "processzor, OpenVINO (Intel és AMD x86 CPU-n is fut)"))
    if hw.faster_whisper:
        chain.append(Candidate("faster-whisper", "cpu", "processzor, faster-whisper (int8)"))
    if hw.whispercpp_binary:
        chain.append(Candidate("whispercpp", "cpu", "processzor, whisper.cpp"))
    return chain


def _gpu_chain(hw) -> list[Candidate]:
    """Automatikus GPU-sorrend: minden gyártóhoz a rá legjobban illő háttér."""
    chain = []
    if hw.has("nvidia") and hw.faster_whisper and hw.cuda_devices:
        chain.append(Candidate("faster-whisper", "cuda", "NVIDIA GPU, faster-whisper (CUDA)"))
    if hw.has("amd") and hw.whispercpp_binary and hw.whispercpp_gpu == "vulkan":
        chain.append(Candidate("whispercpp", "gpu", "AMD GPU, whisper.cpp (Vulkan)"))
    if hw.has("amd") and not hw.has("nvidia") and hw.faster_whisper and hw.cuda_devices:
        # NVIDIA nélkül a CTranslate2 csak a ROCm-es buildben lát "CUDA" eszközt: ez az AMD-kártya
        chain.append(Candidate("faster-whisper", "cuda", "AMD GPU, faster-whisper (CTranslate2 ROCm)"))
    if hw.openvino and hw.openvino_gpu:
        chain.append(Candidate("openvino", hw.openvino_gpu, f"Intel GPU, OpenVINO ({hw.openvino_gpu_name or hw.openvino_gpu})"))
    if hw.whispercpp_binary and hw.whispercpp_gpu:
        # Bármely más kártya, amire van GPU-s whisper.cpp (pl. Vulkan NVIDIA-n vagy Intelen)
        chain.append(Candidate("whispercpp", "gpu", f"GPU, whisper.cpp ({hw.whispercpp_gpu})"))
    return chain


def _backend_gpu_device(hw, backend: str) -> str:
    return {"openvino": hw.openvino_gpu or "GPU", "faster-whisper": "cuda", "whispercpp": "gpu"}[backend]


def _backend_cpu_device(backend: str) -> str:
    return "CPU" if backend == "openvino" else "cpu"


def plan(hw, backend: str = "auto", device: str = "auto") -> tuple[list[Candidate], list[str]]:
    """(rangsorolt jelöltek, figyelmeztetések)."""
    backend = normalize_backend(backend)
    dev = (device or "auto").strip()
    dev_l = dev.lower()
    warnings: list[str] = []
    out: list[Candidate] = []

    def add(c: Candidate):
        if not any(c.same_as(x) for x in out):
            out.append(c)

    if backend == "auto":
        if dev_l not in ("auto", "gpu", "cpu"):
            warnings.append(
                f'A "{dev}" eszköznév csak megadott háttérrel értelmezhető; "auto"-ként kezelem.'
            )
            dev_l = "auto"
        if dev_l != "cpu":
            gpu = _gpu_chain(hw)
            if dev_l == "gpu" and not gpu:
                warnings.append("GPU-t kértél, de nincs használható GPU-s háttér: processzoron fut.")
            for c in gpu:
                add(c)
        for c in _cpu_chain(hw):
            add(c)
        return out, warnings

    # Megadott háttér
    if not _installed(hw, backend):
        warnings.append(f'A(z) "{backend}" háttér nincs telepítve vagy nem található; automatikus választás.')
        auto, more = plan(hw, "auto", "cpu" if dev_l == "cpu" else "auto")
        return auto, warnings + more

    if dev_l == "cpu":
        add(Candidate(backend, _backend_cpu_device(backend), f"{backend}, processzor (configban kérve)"))
    elif dev_l == "gpu":
        add(Candidate(backend, _backend_gpu_device(hw, backend), f"{backend}, GPU (configban kérve)"))
    elif dev_l == "auto":
        for c in _gpu_chain(hw):
            if c.backend == backend:
                add(c)
        add(Candidate(backend, _backend_cpu_device(backend), f"{backend}, processzor"))
    else:
        add(Candidate(backend, dev, f"{backend}, {dev} (configban kérve)"))
    # Tartalék: ugyanez a háttér processzoron, majd bármely más processzoros háttér
    add(Candidate(backend, _backend_cpu_device(backend), f"{backend}, processzor (tartalék)"))
    for c in _cpu_chain(hw):
        add(c)
    return out, warnings


def _load(candidate: Candidate, cfg: dict):
    from . import models

    model = models.resolve(candidate.backend, cfg)
    threads = int(cfg.get("threads") or 0)
    if candidate.backend == "openvino":
        from .openvino_backend import OpenVinoTranscriber

        return OpenVinoTranscriber(model, candidate.device, threads=threads)
    if candidate.backend == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperTranscriber

        fw = cfg.get("faster_whisper") or {}
        return FasterWhisperTranscriber(
            model, candidate.device, compute_type=fw.get("compute_type", "auto"), threads=threads
        )
    if candidate.backend == "whispercpp":
        from .whispercpp_backend import WhisperCppTranscriber

        wc = cfg.get("whispercpp") or {}
        from hardware import find_whispercpp

        binary, _ = find_whispercpp(wc.get("binary", ""))
        return WhisperCppTranscriber(
            binary, model, candidate.device, language=cfg.get("language", "hu"),
            threads=threads, extra_args=wc.get("extra_args") or [],
        )
    raise ValueError(candidate.backend)


def create_transcriber(
    cfg: dict,
    hw=None,
    log: Callable[[str], None] = print,
    loader: Callable[[Candidate, dict], object] | None = None,
):
    """Az első sikeresen betöltődő háttér. A visszaadott objektumon: .candidate."""
    if hw is None:
        import hardware

        hw = hardware.detect(cfg)
    candidates, warnings = plan(hw, cfg.get("backend", "auto"), cfg.get("device", "auto"))
    for w in warnings:
        log(f"FIGYELEM: {w}")
    if not candidates:
        raise RuntimeError("Nincs telepített átíró háttér. Futtasd az install.bat-ot.")
    loader = loader or _load
    errors = []
    for i, cand in enumerate(candidates):
        log(f"Háttér: {cand.reason} ...")
        try:
            tr = loader(cand, cfg)
        except Exception as e:  # noqa: BLE001 - minden hiba után a következő jelölt jön
            errors.append(f"{cand.backend}/{cand.device}: {e}")
            nxt = candidates[i + 1].reason if i + 1 < len(candidates) else None
            log(f"  nem sikerült ({e})" + (f"; következik: {nxt}" if nxt else ""))
            continue
        try:
            tr.candidate = cand
        except AttributeError:
            pass
        return tr
    raise RuntimeError("Egyik háttér sem indult el:\n  " + "\n  ".join(errors))
