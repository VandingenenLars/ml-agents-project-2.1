"""
before running: install psutil pynvml GPUtil amdsmi

System Metrics Collector 

Samples CPU, RAM, and GPU usage every 5 seconds 

Writes JSON Lines to data/raw/run_YYYY-MM-DD_HH-MM-SS/system_metrics.json

GPU support:
  NVIDIA
  AMD
  Intel
  Apple

records cpu, gpu and ram usage percentage

"""

from __future__ import annotations
import json
import os
import signal
import sys
import time
import shutil
import subprocess
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class system_metrics_collector:
    _psutil = None
    _pynvml = None
    _GPUtil = None
    _amdsmi = None


def collect_cpu_cores(self) -> int:
    cores = self._psutil.cpu_count(logical=False)
    if not cores:
        cores = self._psutil.cpu_count(logical=True) or 0
    self.cpu_cores = int(cores)
    return self.cpu_cores

def collect_ram_total_gb(self) -> float:
    vm = self._psutil.virtual_memory()
    self.ram_total_gb = round(vm.total / (1024 * 1024 * 1024), 2)
    return self.ram_total_gb

    def __init__(self, run_dir: Optional[str] = None, interval: float = 5.0, file_name: str = "system_metrics.jason") -> None:
        self._lazy_imports()
        self.interval = max(0.1, float(interval))
        self.run_dir = run_dir or self._default_run_dir()
        self.file_path = os.path.join(self.run_dir, file_name)
        self._stop = False
        os.makedirs(self.run_dir, exist_ok=True)
        try:
            self._psutil.cpu_percent(interval=None)
        except Exception:
            pass
        signal.signal(signal.SIGINT, self._handle_sig)
        signal.signal(signal.SIGTERM, self._handle_sig)

    def _lazy_imports(self):
        if self._psutil is None:
            try:
                import psutil as _psutil
                self._psutil = _psutil
            except Exception as e:
                print("ERROR: psutil is required. Install with: pip install psutil", file=sys.stderr)
                raise e

        if self._pynvml is None:
            try:
                import pynvml as _pynvml
                self._pynvml = _pynvml
            except Exception:
                self._pynvml = None

        if self._GPUtil is None:
            try:
                import GPUtil as _GPUtil
                self._GPUtil = _GPUtil
            except Exception:
                self._GPUtil = None

        if self._amdsmi is None:
            try:
                import amdsmi as _amdsmi
                self._amdsmi = _amdsmi
            except Exception:
                self._amdsmi = None

    def _iso_now(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def _collect_nvidia_nvml(self) -> List[Dict[str, Any]]:
        if self._pynvml is None:
            return []
        try:
            self._pynvml.nvmlInit()
        except Exception:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            count = self._pynvml.nvmlDeviceGetCount()
            for idx in range(count):
                h = self._pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                mem_pct = round((mem.used / mem.total) * 100.0, 2) if getattr(mem, "total", 0) else None
                gpus.append({
                    "vendor": "NVIDIA",
                    "index": idx,
                    "name": self._safe_str(self._pynvml.nvmlDeviceGetName(h)),
                    "utilization_percent": float(getattr(util, "gpu", 0.0)),
                    "memory_utilization_percent": mem_pct,
                })
        except Exception:
            pass
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpus

    def _collect_nvidia_smi(self) -> List[Dict[str, Any]]:
        cmd = shutil.which("nvidia-smi")
        if not cmd:
            return []
        q = "utilization.gpu,memory.used,memory.total,name"
        try:
            out = subprocess.check_output(
                [cmd, "--query-gpu="+q, "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            gpus: List[Dict[str, Any]] = []
            for idx, line in enumerate(l for l in out.strip().splitlines() if l.strip()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                util = self._to_float(parts[0])
                mem_used = self._to_float(parts[1])
                mem_total = self._to_float(parts[2])
                mem_pct = round((mem_used / mem_total) * 100.0, 2) if (mem_used is not None and mem_total) else None
                gpus.append({
                    "vendor": "NVIDIA",
                    "index": idx,
                    "name": parts[3],
                    "utilization_percent": util if util is not None else 0.0,
                    "memory_utilization_percent": mem_pct,
                })
            return gpus
        except Exception:
            return []

    def _collect_nvidia_gputil(self) -> List[Dict[str, Any]]:
        if self._GPUtil is None:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            for g in self._GPUtil.getGPUs():
                mem_total = getattr(g, "memoryTotal", None)
                mem_used = getattr(g, "memoryUsed", None)
                mem_pct = round((mem_used / mem_total) * 100.0, 2) if mem_total else None
                gpus.append({
                    "vendor": "NVIDIA",
                    "index": getattr(g, "id", None),
                    "name": getattr(g, "name", None),
                    "utilization_percent": round(getattr(g, "load", 0.0) * 100.0, 2),
                    "memory_utilization_percent": mem_pct,
                })
        except Exception:
            pass
        return gpus

    def _collect_amd_amdsmi(self) -> List[Dict[str, Any]]:
        if self._amdsmi is None:
            return []
        try:
            self._amdsmi.amdsmi_init()
        except Exception:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            sockets = self._amdsmi.amdsmi_get_socket_handles()
            for s in sockets:
                for dev in self._amdsmi.amdsmi_get_processor_handles(s):
                    try:
                        name = self._safe_str(self._amdsmi.amdsmi_get_gpu_vendor_name(dev)) + " " + self._safe_str(self._amdsmi.amdsmi_get_gpu_name(dev))
                    except Exception:
                        name = "AMD GPU"
                    try:
                        util = self._amdsmi.amdsmi_get_gpu_activity(dev).gfx_activity
                    except Exception:
                        util = None
                    try:
                        mem = self._amdsmi.amdsmi_get_gpu_memory_usage(dev)
                        mem_pct = round((mem.used / mem.total) * 100.0, 2) if getattr(mem, "total", 0) else None
                    except Exception:
                        mem_pct = None
                    gpus.append({
                        "vendor": "AMD",
                        "index": self._safe_idx(dev),
                        "name": name.strip(),
                        "utilization_percent": float(util) if util is not None else None,
                        "memory_utilization_percent": mem_pct,
                    })
        except Exception:
            pass
        return gpus

    def _collect_amd_rocm_smi(self) -> List[Dict[str, Any]]:
        cmd = shutil.which("rocm-smi")
        if not cmd:
            return []
        try:
            out = subprocess.check_output(
                [cmd, "--showuse", "--showmemuse", "--json"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            )
            data = json.loads(out)
            gpus: List[Dict[str, Any]] = []
            cards = data.get("card", data.get("cards", data))
            if isinstance(cards, dict):
                cards = list(cards.values())
            if isinstance(cards, list):
                for idx, c in enumerate(cards):
                    try:
                        name = c.get("Card series", c.get("Card series ", "AMD GPU"))
                        util = self._extract_pct_any(c, ["GPU use (%)", "GPU use %", "GPU use"])
                        mem_used = self._to_float(c.get("VRAM use (MiB)")) or self._to_float(c.get("VRAM use (MB)"))
                        mem_total = self._to_float(c.get("VRAM total (MiB)")) or self._to_float(c.get("VRAM total (MB)"))
                        mem_pct = round((mem_used / mem_total) * 100.0, 2) if (mem_used and mem_total) else None
                        gpus.append({
                            "vendor": "AMD",
                            "index": idx,
                            "name": name,
                            "utilization_percent": util,
                            "memory_utilization_percent": mem_pct,
                        })
                    except Exception:
                        continue
            if gpus:
                return gpus
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [cmd, "--showuse", "--showmemuse"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            )
            gpus: List[Dict[str, Any]] = []
            idx = -1
            current = {}
            for line in out.splitlines():
                ls = line.strip()
                if ls.startswith("GPU") and ":" in ls:
                    if current:
                        gpus.append(current)
                    idx += 1
                    current = {"vendor": "AMD", "index": idx, "name": "AMD GPU",
                               "utilization_percent": None, "memory_utilization_percent": None}
                if "%" in ls and "GPU use" in ls:
                    current["utilization_percent"] = self._extract_first_pct(ls)
                if "VRAM" in ls and "(" in ls and ")" in ls and "%" in ls:
                    current["memory_utilization_percent"] = self._extract_first_pct(ls)
            if current:
                gpus.append(current)
            return gpus
        except Exception:
            return []

    def _collect_intel_gpu_top(self) -> List[Dict[str, Any]]:
        cmd = shutil.which("intel_gpu_top")
        if not cmd:
            return []
        try:
            out = subprocess.check_output(
                [cmd, "-J", "-s", "100", "-d", "1"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            out = out.strip()
            data = json.loads(out)
            engines = data.get("engines", [])
            busy_vals = []
            for e in engines:
                b = e.get("busy", None)
                if isinstance(b, (int, float)):
                    busy_vals.append(float(b))
            util = min(100.0, sum(busy_vals)) if busy_vals else None
            return [{
                "vendor": "Intel",
                "index": 0,
                "name": "Intel GPU",
                "utilization_percent": util,
                "memory_utilization_percent": None,
            }]
        except Exception:
            return []

    def _collect_apple_powermetrics(self) -> List[Dict[str, Any]]:
        cmd = shutil.which("powermetrics")
        if platform.system() != "Darwin" or not cmd:
            return []
        try:
            out = subprocess.check_output(
                [cmd, "--show-gpu", "-n", "1"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            util = None
            for line in out.splitlines():
                if "GPU Active" in line and "%" in line:
                    util = self._extract_first_pct(line)
                    break
            return [{
                "vendor": "Apple",
                "index": 0,
                "name": "Apple GPU",
                "utilization_percent": util,
                "memory_utilization_percent": None,
            }]
        except Exception:
            return []

    def _safe_str(self, x):
        try:
            if isinstance(x, bytes):
                return x.decode(errors="ignore")
            return str(x)
        except Exception:
            return "GPU"

    def _safe_idx(self, x):
        try:
            return int(x)
        except Exception:
            return None

    def _to_float(self, x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace("%", "").replace("MiB", "").replace("MB", "").replace("GiB", "").replace("GB", "")
            return float(s)
        except Exception:
            return None

    def _extract_first_pct(self, s: str) -> Optional[float]:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
        return float(m.group(1)) if m else None

    def _extract_pct_any(self, d: dict, keys: List[str]) -> Optional[float]:
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            f = self._to_float(v)
            if f is not None:
                return f
        return None

    def _collect_gpus(self) -> List[Dict[str, Any]]:
        all_gpus: List[Dict[str, Any]] = []
        nvidia_paths = [self._collect_nvidia_nvml, self._collect_nvidia_smi, self._collect_nvidia_gputil]
        amd_paths = [self._collect_amd_amdsmi, self._collect_amd_rocm_smi]
        intel_paths = [self._collect_intel_gpu_top]
        apple_paths = [self._collect_apple_powermetrics]

        for fn in nvidia_paths:
            g = fn()
            if g:
                all_gpus.extend(g)
                break

        for fn in amd_paths:
            g = fn()
            if g:
                all_gpus.extend(g)
                break

        for fn in intel_paths:
            g = fn()
            if g:
                all_gpus.extend(g)
                break

        for fn in apple_paths:
            g = fn()
            if g:
                all_gpus.extend(g)
                break

        return all_gpus

    def _default_run_dir(self) -> str:
        local = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join("data", "raw", f"run_{local}")

    def _handle_sig(self, signum, frame):
        print("\nReceived signal, stopping...", file=sys.stderr)
        self._stop = True

    def sample_once(self) -> Dict[str, Any]:
        vm = self._psutil.virtual_memory()
        cpu_pct = self._psutil.cpu_percent(interval=None)
        gpus = self._collect_gpus()
        gpu_utils = [g.get("utilization_percent") for g in gpus if isinstance(g.get("utilization_percent"), (int, float))]
        gpu_util = int(round(max(gpu_utils))) if gpu_utils else 0
        record = {
            "timestamp": self._iso_now(),
            "cpu_utilization": round(float(cpu_pct), 1),
            "ram_used_mb": round(vm.used / (1024 * 1024), 1),
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "gpu_utilization": gpu_util,
        }
        return record

    def run(self, verbose: bool = True) -> None:
        print(f"Writing to {self.file_path}")
        written = 0
        with open(self.file_path, "a", encoding="utf-8") as f:
            while not self._stop:
                rec = self.sample_once()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                written += 1
                if verbose:
                    print(f"[{rec['timestamp']}] CPU {rec['cpu_utilization']:.1f}% | RAM used {rec['ram_used_mb']:.1f} MB avail {rec['ram_available_mb']:.1f} MB | GPU {rec['gpu_utilization']}%")
                time.sleep(self.interval)
        print(f"Stopped. Samples written: {written}")

    def stop(self):
        self._stop = True
