"""A háttérválasztás tesztjei, kitalált hardverrel (nincs szükség valódi GPU-ra).

Futtatás:  .venv\\Scripts\\python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import Candidate, create_transcriber, normalize_backend, plan  # noqa: E402
from hardware import GPU, HardwareInfo, parse_video_controllers, vendor_from_name  # noqa: E402

NVIDIA = GPU("NVIDIA GeForce RTX 4070", "nvidia")
AMD = GPU("AMD Radeon RX 7800 XT", "amd")
INTEL_ARC = GPU("Intel(R) Arc(TM) B580 Graphics", "intel")
INTEL_IGPU = GPU("Intel(R) UHD Graphics 770", "intel")


def hw(**kw):
    base = dict(gpus=[], openvino=True)
    base.update(kw)
    return HardwareInfo(**base)


def first(h, backend="auto", device="auto"):
    cands, _ = plan(h, backend, device)
    return (cands[0].backend, cands[0].device)


def pairs(h, backend="auto", device="auto"):
    cands, _ = plan(h, backend, device)
    return [(c.backend, c.device) for c in cands]


class VendorTests(unittest.TestCase):
    def test_vendor_names(self):
        self.assertEqual(vendor_from_name("NVIDIA GeForce RTX 3060"), "nvidia")
        self.assertEqual(vendor_from_name("AMD Radeon RX 6600"), "amd")
        self.assertEqual(vendor_from_name("Radeon (TM) Graphics", "Advanced Micro Devices, Inc."), "amd")
        self.assertEqual(vendor_from_name("Intel(R) Arc(TM) B580 Graphics"), "intel")
        self.assertEqual(vendor_from_name("Intel(R) Iris(R) Xe Graphics"), "intel")
        self.assertEqual(vendor_from_name("Parsec Virtual Display Adapter"), "other")

    def test_parse_single_and_list(self):
        one = json.dumps({"Name": "Intel(R) Arc(TM) B580 Graphics", "AdapterCompatibility": "Intel Corporation"})
        self.assertEqual(parse_video_controllers(one), [INTEL_ARC])
        many = json.dumps([
            {"Name": "AMD Radeon RX 7800 XT", "AdapterCompatibility": "Advanced Micro Devices, Inc."},
            {"Name": "Microsoft Basic Display Adapter", "AdapterCompatibility": "(Standard display types)"},
            {"Name": "Intel(R) UHD Graphics 770", "AdapterCompatibility": "Intel Corporation"},
        ])
        self.assertEqual([g.vendor for g in parse_video_controllers(many)], ["amd", "intel"])
        self.assertEqual(parse_video_controllers(""), [])


class AutoPlanTests(unittest.TestCase):
    def test_nvidia_uses_faster_whisper_cuda(self):
        h = hw(gpus=[NVIDIA], faster_whisper=True, cuda_devices=1)
        self.assertEqual(first(h), ("faster-whisper", "cuda"))
        self.assertIn(("openvino", "CPU"), pairs(h))  # tartalék a végén

    def test_nvidia_without_faster_whisper_falls_back_to_cpu(self):
        h = hw(gpus=[NVIDIA], faster_whisper=False)
        self.assertEqual(pairs(h), [("openvino", "CPU")])

    def test_nvidia_with_broken_cuda_falls_back_to_cpu(self):
        h = hw(gpus=[NVIDIA], faster_whisper=True, cuda_devices=0)
        self.assertEqual(pairs(h), [("openvino", "CPU"), ("faster-whisper", "cpu")])

    def test_amd_with_vulkan_whispercpp(self):
        h = hw(gpus=[AMD], whispercpp_binary="bin/whisper-server.exe", whispercpp_gpu="vulkan")
        self.assertEqual(pairs(h), [("whispercpp", "gpu"), ("openvino", "CPU"), ("whispercpp", "cpu")])

    def test_amd_with_cpu_only_whispercpp_is_not_gpu(self):
        h = hw(gpus=[AMD], whispercpp_binary="bin/whisper-server.exe", whispercpp_gpu=None)
        self.assertEqual(first(h), ("openvino", "CPU"))

    def test_amd_without_binary_runs_on_cpu(self):
        h = hw(gpus=[AMD])
        self.assertEqual(pairs(h), [("openvino", "CPU")])

    def test_amd_with_rocm_ctranslate2(self):
        h = hw(gpus=[AMD], faster_whisper=True, cuda_devices=1)
        self.assertEqual(first(h), ("faster-whisper", "cuda"))

    def test_intel_arc_uses_openvino_gpu(self):
        h = hw(gpus=[INTEL_ARC], openvino_gpu="GPU", openvino_gpu_name="Intel Arc B580")
        self.assertEqual(pairs(h), [("openvino", "GPU"), ("openvino", "CPU")])

    def test_intel_igpu_plus_nvidia_prefers_nvidia(self):
        h = hw(gpus=[INTEL_IGPU, NVIDIA], openvino_gpu="GPU.0", faster_whisper=True, cuda_devices=1)
        self.assertEqual(pairs(h)[:2], [("faster-whisper", "cuda"), ("openvino", "GPU.0")])

    def test_amd_dgpu_plus_intel_igpu_without_vulkan_uses_intel_igpu(self):
        h = hw(gpus=[INTEL_IGPU, AMD], openvino_gpu="GPU")
        self.assertEqual(first(h), ("openvino", "GPU"))

    def test_cpu_only_machine(self):
        h = hw(gpus=[GPU("Microsoft Remote Display Adapter", "other")])
        self.assertEqual(pairs(h), [("openvino", "CPU")])

    def test_forced_cpu_skips_gpus(self):
        h = hw(gpus=[NVIDIA], faster_whisper=True, cuda_devices=1, openvino_gpu=None)
        self.assertEqual(pairs(h, device="cpu"), [("openvino", "CPU"), ("faster-whisper", "cpu")])

    def test_gpu_requested_but_none_warns(self):
        cands, warnings = plan(hw(), "auto", "gpu")
        self.assertEqual([(c.backend, c.device) for c in cands], [("openvino", "CPU")])
        self.assertTrue(warnings)

    def test_nothing_installed(self):
        cands, _ = plan(hw(openvino=False), "auto", "auto")
        self.assertEqual(cands, [])


class ExplicitPlanTests(unittest.TestCase):
    def test_explicit_backend_and_gpu(self):
        h = hw(gpus=[AMD], whispercpp_binary="x.exe", whispercpp_gpu=None)
        # Ha a felhasználó kéri, megpróbáljuk a GPU-t akkor is, ha a felismerés nem látta
        self.assertEqual(pairs(h, "whispercpp", "gpu"),
                         [("whispercpp", "gpu"), ("whispercpp", "cpu"), ("openvino", "CPU")])

    def test_explicit_native_device_name(self):
        h = hw(gpus=[INTEL_IGPU, INTEL_ARC], openvino_gpu="GPU.1")
        self.assertEqual(first(h, "openvino", "GPU.0"), ("openvino", "GPU.0"))
        self.assertEqual(first(h, "openvino", "auto"), ("openvino", "GPU.1"))  # diszkrét kártya
        h2 = hw(gpus=[NVIDIA, NVIDIA], faster_whisper=True, cuda_devices=2)
        self.assertEqual(first(h2, "faster-whisper", "cuda:1"), ("faster-whisper", "cuda:1"))

    def test_explicit_backend_not_installed_falls_back(self):
        h = hw(gpus=[NVIDIA], faster_whisper=False)
        cands, warnings = plan(h, "faster-whisper", "auto")
        self.assertEqual([(c.backend, c.device) for c in cands], [("openvino", "CPU")])
        self.assertTrue(any("nincs telepítve" in w for w in warnings))

    def test_explicit_cpu(self):
        h = hw(gpus=[NVIDIA], faster_whisper=True, cuda_devices=1)
        self.assertEqual(first(h, "faster-whisper", "cpu"), ("faster-whisper", "cpu"))

    def test_aliases(self):
        self.assertEqual(normalize_backend("whisper.cpp"), "whispercpp")
        self.assertEqual(normalize_backend("faster_whisper"), "faster-whisper")
        with self.assertRaises(ValueError):
            normalize_backend("tensorrt")


class FallbackTests(unittest.TestCase):
    def test_create_transcriber_skips_failing_candidates(self):
        h = hw(gpus=[NVIDIA], faster_whisper=True, cuda_devices=1)
        tried = []

        class Fake:
            def __init__(self, cand):
                self.cand = cand

        def loader(cand: Candidate, cfg):
            tried.append((cand.backend, cand.device))
            if cand.device == "cuda":
                raise RuntimeError("Could not load cudnn_ops64_9.dll")
            return Fake(cand)

        logs = []
        tr = create_transcriber({"backend": "auto", "device": "auto"}, hw=h, log=logs.append, loader=loader)
        self.assertEqual(tried, [("faster-whisper", "cuda"), ("openvino", "CPU")])
        self.assertEqual((tr.candidate.backend, tr.candidate.device), ("openvino", "CPU"))
        self.assertTrue(any("cudnn" in line for line in logs))

    def test_create_transcriber_all_fail(self):
        def loader(cand, cfg):
            raise RuntimeError("nope")

        with self.assertRaises(RuntimeError):
            create_transcriber({}, hw=hw(), log=lambda s: None, loader=loader)


class ModelResolveTests(unittest.TestCase):
    def test_aliases_per_backend(self):
        from backends.models import resolve

        cfg = {"model": "large-v3", "openvino": {"model": ""}, "faster_whisper": {"model": ""}, "whispercpp": {"model": ""}}
        self.assertEqual(resolve("openvino", cfg), "OpenVINO/whisper-large-v3-int8-ov")
        self.assertEqual(resolve("faster-whisper", cfg), "large-v3")
        self.assertEqual(resolve("whispercpp", cfg), "ggml-large-v3-q5_0.bin")
        cfg["whispercpp"]["model"] = "ggml-large-v3.bin"
        self.assertEqual(resolve("whispercpp", cfg), "ggml-large-v3.bin")
        cfg["model"] = "D:/sajat/modell"
        self.assertEqual(resolve("openvino", cfg), "D:/sajat/modell")


class DetectDryRunTests(unittest.TestCase):
    """hardware.detect() a rendszerpróbák helyettesítésével (mock)."""

    def _detect(self, gpus, ov=(True, None, ""), ct2=(False, 0), wcpp=(None, None)):
        from unittest import mock

        import hardware

        with mock.patch.object(hardware, "list_gpus", return_value=gpus), \
             mock.patch.object(hardware, "probe_openvino", return_value=ov), \
             mock.patch.object(hardware, "probe_ctranslate2", return_value=ct2), \
             mock.patch.object(hardware, "find_whispercpp", return_value=wcpp), \
             mock.patch.object(hardware, "probe_mlx", return_value=False), \
             mock.patch.object(hardware, "is_apple_silicon", return_value=False):
            return hardware.detect({})

    def test_amd_machine_without_vulkan(self):
        info = self._detect([AMD])
        self.assertTrue(any("AMD GPU" in n for n in info.notes))
        self.assertEqual(first(info), ("openvino", "CPU"))

    def test_amd_machine_with_vulkan(self):
        info = self._detect([AMD], wcpp=("bin/whispercpp/whisper-server.exe", "vulkan"))
        self.assertFalse(any("AMD GPU" in n for n in info.notes))
        self.assertEqual(first(info), ("whispercpp", "gpu"))

    def test_nvidia_machine_missing_packages(self):
        info = self._detect([NVIDIA])
        self.assertTrue(any("requirements-nvidia.txt" in n for n in info.notes))
        self.assertEqual(first(info), ("openvino", "CPU"))

    def test_nvidia_machine_ready(self):
        info = self._detect([NVIDIA], ct2=(True, 1))
        self.assertEqual(info.notes, [])
        self.assertEqual(first(info), ("faster-whisper", "cuda"))


class WhisperCppCommandTests(unittest.TestCase):
    def test_command_flags(self):
        from backends.whispercpp_backend import build_command

        cpu = build_command("ws.exe", Path("m.bin"), 1234, "cpu", "hu", 4, [])
        self.assertIn("-ng", cpu)
        self.assertEqual(cpu[cpu.index("-l") + 1], "hu")
        gpu1 = build_command("ws.exe", Path("m.bin"), 1234, "gpu:1", "hu", 4, ["--beam-size", "5"])
        self.assertNotIn("-ng", gpu1)
        self.assertEqual(gpu1[gpu1.index("-dev") + 1], "1")
        self.assertEqual(gpu1[-2:], ["--beam-size", "5"])


class MacLinuxDetectTests(unittest.TestCase):
    def test_system_profiler(self):
        from hardware import parse_system_profiler

        raw = json.dumps({"SPDisplaysDataType": [
            {"_name": "kHW_AppleM2Item", "sppci_model": "Apple M2", "spdisplays_vendor": "sppci_vendor_Apple"},
        ]})
        self.assertEqual(parse_system_profiler(raw), [GPU("Apple M2", "apple")])
        intel_mac = json.dumps({"SPDisplaysDataType": [
            {"sppci_model": "Intel Iris Plus Graphics", "spdisplays_vendor": "Intel"},
            {"sppci_model": "AMD Radeon Pro 5300M", "spdisplays_vendor": "sppci_vendor_amd"},
        ]})
        self.assertEqual([g.vendor for g in parse_system_profiler(intel_mac)], ["intel", "amd"])
        self.assertEqual(parse_system_profiler(""), [])

    def test_lspci(self):
        from hardware import parse_lspci

        raw = (
            "00:02.0 VGA compatible controller: Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] (rev 04)\n"
            "00:14.0 USB controller: Intel Corporation Alder Lake-S PCH USB 3.2\n"
            "01:00.0 VGA compatible controller: NVIDIA Corporation AD104 [GeForce RTX 4070] (rev a1)\n"
            "03:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7800 XT]\n"
        )
        self.assertEqual([g.vendor for g in parse_lspci(raw)], ["intel", "nvidia", "amd"])

    def test_whispercpp_gpu_from_libs(self):
        from hardware import gpu_from_libs

        self.assertEqual(gpu_from_libs({"whisper-server", "libggml-vulkan.so", "libggml.so"}), "vulkan")
        self.assertEqual(gpu_from_libs({"ggml-cuda.dll", "whisper.dll"}), "cuda")
        self.assertEqual(gpu_from_libs({"libggml-metal.dylib"}), "metal")
        self.assertIsNone(gpu_from_libs({"libggml-cpu.so", "libwhisper.so.1"}))

    def test_apple_silicon_prefers_mlx(self):
        h = hw(gpus=[GPU("Apple M2", "apple")], openvino=False, faster_whisper=True, mlx=True)
        self.assertEqual(pairs(h), [("mlx", "gpu"), ("faster-whisper", "cpu")])

    def test_apple_silicon_without_mlx_uses_metal_whispercpp(self):
        h = hw(gpus=[GPU("Apple M1", "apple")], openvino=False, faster_whisper=True,
               whispercpp_binary="/x/whisper-server", whispercpp_gpu="metal")
        self.assertEqual(first(h), ("whispercpp", "gpu"))

    def test_intel_mac_runs_on_cpu(self):
        h = hw(gpus=[GPU("Intel Iris Plus Graphics", "intel")], openvino=False, faster_whisper=True)
        self.assertEqual(pairs(h), [("faster-whisper", "cpu")])

    def test_mlx_alias_and_explicit(self):
        self.assertEqual(normalize_backend("mlx-whisper"), "mlx")
        h = hw(openvino=False, faster_whisper=True, mlx=True)
        self.assertEqual(pairs(h, "mlx", "cpu")[0], ("mlx", "cpu"))

    def test_mlx_model_names(self):
        from backends.models import resolve

        self.assertEqual(resolve("mlx", {"model": "large-v3", "mlx": {"model": ""}}), "mlx-community/whisper-large-v3-mlx")
        self.assertEqual(resolve("mlx", {"model": "turbo"}), "mlx-community/whisper-large-v3-turbo")


class FakeKey:
    def __init__(self, name=None, char=None):
        self.name, self.char = name, char


class HotkeyTests(unittest.TestCase):
    def test_parse(self):
        from hotkeys import parse_hotkey

        self.assertEqual(parse_hotkey("ctrl+windows"), ["ctrl", "cmd"])
        self.assertEqual(parse_hotkey("Ctrl + Super"), ["ctrl", "cmd"])
        self.assertEqual(parse_hotkey("right shift"), ["shift_r"])
        self.assertEqual(parse_hotkey("left  alt+f9"), ["alt_l", "f9"])
        self.assertEqual(parse_hotkey("option+space"), ["alt", "space"])
        self.assertEqual(parse_hotkey("caps lock"), ["caps_lock"])
        with self.assertRaises(ValueError):
            parse_hotkey("ctrl+")

    def test_key_names(self):
        from hotkeys import key_names

        self.assertEqual(key_names(FakeKey("ctrl_r")), {"ctrl_r", "ctrl"})
        self.assertEqual(key_names(FakeKey("cmd")), {"cmd"})
        self.assertEqual(key_names(FakeKey("caps_lock")), {"caps_lock"})
        self.assertEqual(key_names(FakeKey(char="V")), {"v"})

    def test_hold_combo(self):
        from hotkeys import ComboTracker

        t = ComboTracker(["ctrl", "cmd"])
        self.assertFalse(t.press("ctrl_l", {"ctrl", "ctrl_l"}))
        self.assertTrue(t.press("cmd", {"cmd"}))           # teljes: indul
        self.assertFalse(t.press("cmd", {"cmd"}))          # ismétlődő lenyomás: semmi
        self.assertFalse(t.release("x"))                   # idegen billentyű
        self.assertTrue(t.release("ctrl_l"))               # egy tag felengedve: leáll
        self.assertFalse(t.release("cmd"))                 # a második felengedés már nem számít
        self.assertFalse(t.press("ctrl_r", {"ctrl", "ctrl_r"}))  # újra: jobb Ctrl is jó
        self.assertTrue(t.press("cmd", {"cmd"}))

    def test_side_specific(self):
        from hotkeys import ComboTracker

        t = ComboTracker(["shift_r"])
        self.assertFalse(t.press("shift_l", {"shift", "shift_l"}))
        self.assertTrue(t.press("shift_r", {"shift", "shift_r"}))


class ControlTests(unittest.TestCase):
    def test_roundtrip(self):
        import threading
        from unittest import mock

        import control

        with mock.patch.object(control, "PORT_FILE", Path(__file__).resolve().parent / ".test.port"):
            hit = threading.Event()
            server = control.ControlServer({"toggle": hit.set})
            try:
                self.assertEqual(control.send("toggle"), "ok")
                self.assertTrue(hit.wait(2))
                self.assertEqual(control.send("valami"), "ismeretlen parancs")
            finally:
                server.close()
            self.assertFalse(control.PORT_FILE.exists())
            with self.assertRaises(RuntimeError):
                control.send("toggle")


if __name__ == "__main__":
    unittest.main()
