"""Hardver-felismerés: milyen GPU van a gépen, és melyik átíró háttér futtatható.

Minden próba hibatűrő: ha valami nem derül ki (nincs telepítve egy csomag, nem
válaszol egy rendszerparancs), az adott lehetőséget egyszerűen nem kínáljuk fel,
és a választás a következő háttérre lép, végső soron a processzorra.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def is_apple_silicon() -> bool:
    return IS_MAC and platform.machine() == "arm64"


def venv_cmd(tool: str) -> str:
    """A virtuális környezet egy parancsa, ahogy a felhasználó beírná (tippekhez)."""
    return f".venv\\Scripts\\{tool}" if IS_WINDOWS else f".venv/bin/{tool}"


INSTALL_SCRIPT = "install.bat" if IS_WINDOWS else "./install.sh"

# Ezek nem valódi GPU-k (távoli asztal, virtuális kijelző, alap meghajtó)
_IGNORED = ("microsoft basic", "remote display", "parsec", "virtual", "citrix", "vmware", "spacedesk")


@dataclass(frozen=True)
class GPU:
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "other"


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
    # whisper.cpp szerver binárisa és a GPU-háttere ("vulkan" | "cuda" | "metal" | None)
    whispercpp_binary: str | None = None
    whispercpp_gpu: str | None = None
    # mlx-whisper telepítve van-e (csak Apple Silicon Macen fut)
    mlx: bool = False
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
    if "apple" in low:
        return "apple"
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


def parse_system_profiler(raw: str) -> list[GPU]:
    """A macOS `system_profiler SPDisplaysDataType -json` kimenetének feldolgozása."""
    raw = (raw or "").strip()
    if not raw:
        return []
    gpus = []
    for item in json.loads(raw).get("SPDisplaysDataType", []):
        name = (item.get("sppci_model") or item.get("_name") or "").strip()
        if not name:
            continue
        gpu = GPU(name, vendor_from_name(name, item.get("spdisplays_vendor") or ""))
        if gpu not in gpus:
            gpus.append(gpu)
    return gpus


def parse_lspci(raw: str) -> list[GPU]:
    """Az `lspci` kimenetéből a videókártyák (pl. "01:00.0 VGA compatible controller: NVIDIA ...")."""
    gpus = []
    for line in (raw or "").splitlines():
        if any(k in line for k in ("VGA compatible", "3D controller", "Display controller")):
            name = line.split(":", 2)[-1].strip()
            gpu = GPU(name, vendor_from_name(name))
            if gpu not in gpus:
                gpus.append(gpu)
    return gpus


# PCI gyártókódok a /sys/class/drm alatti tartalékhoz (ha nincs lspci)
_PCI_VENDORS = {"0x10de": ("NVIDIA GPU", "nvidia"), "0x1002": ("AMD GPU", "amd"), "0x8086": ("Intel GPU", "intel")}


def _linux_sysfs_gpus() -> list[GPU]:
    gpus = []
    for vendor_file in sorted(Path("/sys/class/drm").glob("card*/device/vendor")):
        try:
            code = vendor_file.read_text().strip().lower()
        except OSError:
            continue
        if code in _PCI_VENDORS:
            gpu = GPU(*_PCI_VENDORS[code])
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
        if IS_MAC:
            return parse_system_profiler(_run(["system_profiler", "SPDisplaysDataType", "-json"]))
        if shutil.which("lspci"):
            return parse_lspci(_run(["lspci"]))
        gpus = _linux_sysfs_gpus()
        if gpus:
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

def _site_roots() -> set[str]:
    roots = set()
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    roots.add(str(Path(sys.prefix) / "Lib" / "site-packages"))
    return roots


_nvidia_libs_loaded = False


def _preload_nvidia_libs() -> None:
    """Linux: a pip-es cuBLAS/cuDNN .so-fájlok előtöltése.

    A CTranslate2 a nevük alapján keresi őket, a site-packages/nvidia/*/lib
    mappákat viszont a dinamikus betöltő nem ismeri, az LD_LIBRARY_PATH pedig
    futás közben már nem módosítható. Ha a könyvtárakat előre, RTLD_GLOBAL-lal
    betöltjük, a későbbi név szerinti keresés a már betöltöttet kapja meg.
    """
    global _nvidia_libs_loaded
    if _nvidia_libs_loaded:
        return
    _nvidia_libs_loaded = True
    import ctypes

    pending = sorted({
        str(p) for root in _site_roots() for p in Path(root).glob("nvidia/*/lib/lib*.so*")
        if p.name.startswith(("libcublas", "libcudnn"))
    })
    # A sorrend számít (pl. cublasLt a cublas előtt), ezért amíg halad, újrapróbáljuk a sikerteleneket.
    while pending:
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if len(failed) == len(pending):
            break
        pending = failed


def add_nvidia_dll_dirs() -> None:
    """A pip-ből telepített NVIDIA-könyvtárak (cuBLAS, cuDNN) elérhetővé tétele.

    Windowson a nvidia-cublas-cu12 / nvidia-cudnn-cu12 csomagok a DLL-eket a
    site-packages/nvidia/*/bin mappákba teszik, de ezeket a Windows magától nem
    keresi. Az importálás előtt hozzá kell adni őket a keresési úthoz. Linuxon
    ugyanezt az előtöltés oldja meg (_preload_nvidia_libs).
    """
    if sys.platform.startswith("linux"):
        _preload_nvidia_libs()
        return
    if sys.platform != "win32":
        return
    for root in _site_roots():
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


def probe_mlx() -> bool:
    """mlx-whisper (Apple Silicon GPU, Metal). Csak arm64-es Macen értelmes."""
    return is_apple_silicon() and bool(importlib.util.find_spec("mlx_whisper"))


def _server_name() -> str:
    return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"


def gpu_from_libs(names: set[str]) -> str | None:
    """A whisper.cpp GPU-háttere a mellette lévő ggml-könyvtárak nevéből (.dll, .so, .dylib)."""
    for kind in ("vulkan", "cuda", "metal"):
        if any(f"ggml-{kind}" in n for n in names):
            return kind
    return None


def find_whispercpp(binary: str = "") -> tuple[str | None, str | None]:
    """(whisper-server útja, GPU-háttér). A GPU-hátteret a mellette lévő ggml-könyvtárakból látjuk."""
    candidates = []
    if binary:
        candidates.append(Path(binary))
    candidates.append(APP_DIR / "bin" / "whispercpp" / _server_name())
    found = shutil.which(_server_name())
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            folder = path.resolve().parent
            names = set()
            for d in (folder, folder.parent / "lib"):  # Linux/macOS: a könyvtárak a ../lib alatt is lehetnek
                if d.is_dir():
                    names.update(p.name.lower() for p in d.iterdir())
            gpu = gpu_from_libs(names)
            if gpu is None and is_apple_silicon():
                gpu = "metal"  # Apple Siliconon a whisper.cpp alapból Metallal fordul (a könyvtárba ágyazva)
            return str(path), gpu
    return None, None


def detect(cfg: dict | None = None) -> HardwareInfo:
    cfg = cfg or {}
    info = HardwareInfo(gpus=list_gpus())
    info.openvino, info.openvino_gpu, info.openvino_gpu_name = probe_openvino()
    info.faster_whisper, info.cuda_devices = probe_ctranslate2()
    binary = (cfg.get("whispercpp") or {}).get("binary", "")
    info.whispercpp_binary, info.whispercpp_gpu = find_whispercpp(binary)
    info.mlx = probe_mlx()

    # Tippek a felhasználónak, ha a hardvere többre képes, mint ami telepítve van
    if is_apple_silicon() and not info.mlx:
        info.notes.append(
            f"Apple Silicon Mac, de az mlx-whisper nincs telepítve (GPU-hoz): {venv_cmd('pip')} install mlx-whisper"
        )
    if info.has("nvidia") and not info.cuda_devices:
        if not info.faster_whisper:
            info.notes.append(
                "NVIDIA GPU van a gépben, de a faster-whisper nincs telepítve: "
                f"{venv_cmd('pip')} install -r requirements-nvidia.txt"
            )
        else:
            info.notes.append(
                "NVIDIA GPU van a gépben, de a CTranslate2 nem lát CUDA-eszközt. "
                "Frissítsd az NVIDIA-meghajtót, és ellenőrizd a requirements-nvidia.txt csomagjait."
            )
    if info.has("amd") and info.whispercpp_gpu != "vulkan" and not (info.cuda_devices and not info.has("nvidia")):
        script = "tools\\build-whispercpp-vulkan.ps1" if IS_WINDOWS else "tools/build-whispercpp.sh"
        info.notes.append(
            "AMD GPU van a gépben, de nincs Vulkanos whisper.cpp: a program a processzoron fut. "
            f"GPU-hoz lásd a README 'AMD GPU' szakaszát ({script})."
        )
    if not (info.openvino or info.faster_whisper or info.whispercpp_binary or info.mlx):
        info.notes.append(f"Egyetlen átíró háttér sincs telepítve: futtasd az {INSTALL_SCRIPT} szkriptet.")
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
    if IS_MAC:
        lines.append(f"mlx-whisper:    {'telepítve (Apple GPU)' if info.mlx else 'nincs telepítve'}")
    return "\n".join(lines)
