"""Hardver-felismerés: milyen GPU van a gépen, és melyik átíró háttér futtatható.

Minden próba hibatűrő: ha valami nem derül ki (nincs telepítve egy csomag, nem
válaszol egy rendszerparancs), az adott lehetőséget egyszerűen nem kínáljuk fel,
és a választás a következő háttérre lép, végső soron a processzorra.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW

# Ezek nem valódi GPU-k (távoli asztal, virtuális kijelző, alap meghajtó)
_IGNORED = ("microsoft basic", "remote display", "parsec", "virtual", "citrix", "vmware", "spacedesk")


@dataclass(frozen=True)
class GPU:
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "other"


@dataclass
class HardwareInfo:
    gpus: list[GPU] = field(default_factory=list)
    # OpenVINO (openvino-genai) telepítve van-e, és melyik Intel GPU-t látja ("GPU", "GPU.1")
    openvino: bool = False
    openvino_gpu: str | None = None
    openvino_gpu_name: str = ""
    # faster-whisper telepítve van-e, és hány CUDA/HIP eszközt lát a CTranslate2
    faster_whisper: bool = False
    cuda_devices: int = 0
    # whisper.cpp szerver binárisa és a GPU-háttere ("vulkan" | "cuda" | None)
    whispercpp_binary: str | None = None
    whispercpp_gpu: str | None = None
    notes: list[str] = field(default_factory=list)

    def has(self, vendor: str) -> bool:
        return any(g.vendor == vendor for g in self.gpus)


def vendor_from_name(name: str, company: str = "") -> str:
    low = f"{company} {name}".lower()
    if any(k in low for k in ("nvidia", "geforce", "quadro", "tesla")):
        return "nvidia"
    if any(k in low for k in ("advanced micro devices", "amd", "radeon", "ati technologies")):
        return "amd"
    if any(k in low for k in ("intel", "iris", "uhd graphics", "arc(tm)")):
        return "intel"
    return "other"


def parse_video_controllers(raw: str) -> list[GPU]:
    """A PowerShell Get-CimInstance Win32_VideoController JSON-kimenetének feldolgozása."""
    raw = (raw or "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):  # egyetlen kártyánál nem lista jön
        data = [data]
    gpus = []
    for item in data:
        name = (item.get("Name") or "").strip()
        if not name or any(s in name.lower() for s in _IGNORED):
            continue
        gpu = GPU(name, vendor_from_name(name, item.get("AdapterCompatibility") or ""))
        if gpu not in gpus:
            gpus.append(gpu)
    return gpus


def _run(cmd: list[str], timeout: float = 15) -> str:
    out = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
        encoding="utf-8", errors="replace",
    )
    return out.stdout


def list_gpus() -> list[GPU]:
    """A gépben lévő videókártyák listája (gyártóval)."""
    try:
        if sys.platform == "win32":
            raw = _run([
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterCompatibility | ConvertTo-Json -Compress",
            ])
            return parse_video_controllers(raw)
        if shutil.which("lspci"):
            gpus = []
            for line in _run(["lspci"]).splitlines():
                if any(k in line for k in ("VGA compatible", "3D controller", "Display controller")):
                    name = line.split(":", 2)[-1].strip()
                    gpus.append(GPU(name, vendor_from_name(name)))
            return gpus
    except Exception:
        pass
    # Végső tartalék: ha van nvidia-smi, legalább az NVIDIA-kártyát lássuk
    try:
        if shutil.which("nvidia-smi"):
            return [GPU(l.split(":", 1)[-1].split("(")[0].strip(), "nvidia")
                    for l in _run(["nvidia-smi", "-L"]).splitlines() if l.strip()]
    except Exception:
        pass
    return []


# ---------- háttérpróbák ----------

def add_nvidia_dll_dirs() -> None:
    """A pip-ből telepített NVIDIA-könyvtárak (cuBLAS, cuDNN) DLL-mappáinak felvétele.

    Windowson a nvidia-cublas-cu12 / nvidia-cudnn-cu12 csomagok a DLL-eket a
    site-packages/nvidia/*/bin mappákba teszik, de ezeket a Windows magától nem
    keresi. Az importálás előtt hozzá kell adni őket a keresési úthoz.
    """
    if sys.platform != "win32":
        return
    roots = set()
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    roots.add(str(Path(sys.prefix) / "Lib" / "site-packages"))
    for root in roots:
        base = Path(root) / "nvidia"
        if not base.is_dir():
            continue
        for bin_dir in base.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def probe_openvino() -> tuple[bool, str | None, str]:
    """(telepítve, Intel GPU eszközneve vagy None, a GPU neve). Diszkrét kártyát részesít előnyben."""
    if not (importlib.util.find_spec("openvino") and importlib.util.find_spec("openvino_genai")):
        return False, None, ""
    try:
        import openvino as ov

        core = ov.Core()
        gpus = [d for d in core.available_devices if d.startswith("GPU")]
        if not gpus:
            return True, None, ""
        chosen = gpus[0]
        for dev in gpus:
            try:
                if "DISCRETE" in str(core.get_property(dev, "DEVICE_TYPE")).upper():
                    chosen = dev
                    break
            except Exception:
                pass
        try:
            name = str(core.get_property(chosen, "FULL_DEVICE_NAME"))
        except Exception:
            name = chosen
        return True, chosen, name
    except Exception:
        return True, None, ""


def probe_ctranslate2() -> tuple[bool, int]:
    """(faster-whisper telepítve, CUDA/HIP eszközök száma)."""
    if not (importlib.util.find_spec("faster_whisper") and importlib.util.find_spec("ctranslate2")):
        return False, 0
    try:
        add_nvidia_dll_dirs()
        import ctranslate2

        return True, int(ctranslate2.get_cuda_device_count())
    except Exception:
        return True, 0


def _server_name() -> str:
    return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"


def find_whispercpp(binary: str = "") -> tuple[str | None, str | None]:
    """(whisper-server útja, GPU-háttér). A GPU-hátteret a mellette lévő ggml-DLL-ekből látjuk."""
    candidates = []
    if binary:
        candidates.append(Path(binary))
    candidates.append(APP_DIR / "bin" / "whispercpp" / _server_name())
    found = shutil.which(_server_name())
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            folder = path.parent
            names = {p.name.lower() for p in folder.iterdir()}
            if any("ggml-vulkan" in n for n in names):
                return str(path), "vulkan"
            if any("ggml-cuda" in n for n in names):
                return str(path), "cuda"
            return str(path), None
    return None, None


def detect(cfg: dict | None = None) -> HardwareInfo:
    cfg = cfg or {}
    info = HardwareInfo(gpus=list_gpus())
    info.openvino, info.openvino_gpu, info.openvino_gpu_name = probe_openvino()
    info.faster_whisper, info.cuda_devices = probe_ctranslate2()
    binary = (cfg.get("whispercpp") or {}).get("binary", "")
    info.whispercpp_binary, info.whispercpp_gpu = find_whispercpp(binary)

    # Tippek a felhasználónak, ha a hardvere többre képes, mint ami telepítve van
    if info.has("nvidia") and not info.cuda_devices:
        if not info.faster_whisper:
            info.notes.append(
                "NVIDIA GPU van a gépben, de a faster-whisper nincs telepítve: "
                ".venv\\Scripts\\pip install -r requirements-nvidia.txt"
            )
        else:
            info.notes.append(
                "NVIDIA GPU van a gépben, de a CTranslate2 nem lát CUDA-eszközt. "
                "Frissítsd az NVIDIA-meghajtót, és ellenőrizd a requirements-nvidia.txt csomagjait."
            )
    if info.has("amd") and info.whispercpp_gpu != "vulkan" and not (info.cuda_devices and not info.has("nvidia")):
        info.notes.append(
            "AMD GPU van a gépben, de nincs Vulkanos whisper.cpp: a program a processzoron fut. "
            "GPU-hoz lásd a README 'AMD GPU' szakaszát (tools\\build-whispercpp-vulkan.ps1)."
        )
    if not (info.openvino or info.faster_whisper or info.whispercpp_binary):
        info.notes.append("Egyetlen átíró háttér sincs telepítve: futtasd az install.bat-ot.")
    return info


def describe(info: HardwareInfo) -> str:
    lines = ["Videókártyák:"]
    lines += [f"  - {g.name} [{g.vendor}]" for g in info.gpus] or ["  (nem található)"]
    ov = "nincs telepítve"
    if info.openvino:
        ov = f"telepítve, GPU: {info.openvino_gpu} ({info.openvino_gpu_name})" if info.openvino_gpu \
            else "telepítve, Intel GPU nem látszik (csak CPU)"
    fw = "nincs telepítve"
    if info.faster_whisper:
        fw = f"telepítve, CUDA/HIP eszköz: {info.cuda_devices}"
    wc = "nem található"
    if info.whispercpp_binary:
        wc = f"{info.whispercpp_binary} (GPU-háttér: {info.whispercpp_gpu or 'nincs, csak CPU'})"
    lines += [f"OpenVINO:       {ov}", f"faster-whisper: {fw}", f"whisper.cpp:    {wc}"]
    return "\n".join(lines)
