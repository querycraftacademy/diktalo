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
             mock.patch.object(hardware, "find_whispercpp", return_value=wcpp):
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


if __name__ == "__main__":
    unittest.main()
