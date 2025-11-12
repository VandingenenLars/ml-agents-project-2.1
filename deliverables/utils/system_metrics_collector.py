
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
import argparse
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


_psutil = None
_pynvml = None
_GPUtil = None
_amdsmi = None

def _lazy_imports():
    global _psutil, _pynvml, _GPUtil, _amdsmi
    if _psutil is None:
        try:
            import psutil as _psutil  
        except Exception as e:
            print("ERROR: psutil is required. Install with: pip install psutil", file=sys.stderr)
            raise e

    if _pynvml is None:
        try:
            import pynvml as _pynvml 
        except Exception:
            _pynvml = None 

    if _GPUtil is None:
        try:
            import GPUtil as _GPUtil  
        except Exception:
            _GPUtil = None 

    if _amdsmi is None:
        try:
            import amdsmi as _amdsmi  
        except Exception:
            _amdsmi = None  

def _iso_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")




# GPU Collectors:

class _BaseGPUCollector:
    vendor: str = "Unknown"
    name: str = "base"

    def available(self) -> bool:
        return False

    def collect(self) -> List[Dict[str, Any]]:
        return []


# NVIDIA

class NvidiaNVMLCollector(_BaseGPUCollector):
    vendor = "NVIDIA"
    name = "nvml"

    def __init__(self):
        self.ready = False

    def available(self) -> bool:
        if _pynvml is None:
            return False
        try:
            _pynvml.nvmlInit()
            self.ready = True
            return True
        except Exception:
            return False

    def collect(self) -> List[Dict[str, Any]]:
        if not self.ready:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            count = _pynvml.nvmlDeviceGetCount()
            for idx in range(count):
                h = _pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = _pynvml.nvmlDeviceGetUtilizationRates(h)  # .gpu, .memory
                mem = _pynvml.nvmlDeviceGetMemoryInfo(h)        # .used, .total
                mem_pct = round((mem.used / mem.total) * 100.0, 2) if getattr(mem, "total", 0) else None
                gpus.append({
                    "vendor": self.vendor,
                    "index": idx,
                    "name": _safe_str(_pynvml.nvmlDeviceGetName(h)),
                    "utilization_percent": float(getattr(util, "gpu", 0.0)),
                    "memory_utilization_percent": mem_pct,
                })
        except Exception:
            pass
        return gpus


class NvidiaSmiCLICollector(_BaseGPUCollector):
    vendor = "NVIDIA"
    name = "nvidia-smi"

    def __init__(self):
        self.cmd = shutil.which("nvidia-smi")

    def available(self) -> bool:
        return self.cmd is not None

    def collect(self) -> List[Dict[str, Any]]:
        if not self.cmd:
            return []
        # Query util and memory , returns numbers without units
        q = "utilization.gpu,memory.used,memory.total,name"
        try:
            out = subprocess.check_output(
                [self.cmd, "--query-gpu="+q, "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            gpus: List[Dict[str, Any]] = []
            for idx, line in enumerate(l for l in out.strip().splitlines() if l.strip()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                util = _to_float(parts[0])
                mem_used = _to_float(parts[1])
                mem_total = _to_float(parts[2])
                mem_pct = round((mem_used / mem_total) * 100.0, 2) if (mem_used is not None and mem_total) else None
                gpus.append({
                    "vendor": self.vendor,
                    "index": idx,
                    "name": parts[3],
                    "utilization_percent": util if util is not None else 0.0,
                    "memory_utilization_percent": mem_pct,
                })
            return gpus
        except Exception:
            return []


class NvidiaGPUtilCollector(_BaseGPUCollector):
    vendor = "NVIDIA"
    name = "gputil"

    def available(self) -> bool:
        return _GPUtil is not None

    def collect(self) -> List[Dict[str, Any]]:
        if _GPUtil is None:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            for g in _GPUtil.getGPUs():
                mem_total = getattr(g, "memoryTotal", None)
                mem_used = getattr(g, "memoryUsed", None)
                mem_pct = round((mem_used / mem_total) * 100.0, 2) if mem_total else None
                gpus.append({
                    "vendor": self.vendor,
                    "index": getattr(g, "id", None),
                    "name": getattr(g, "name", None),
                    "utilization_percent": round(getattr(g, "load", 0.0) * 100.0, 2),
                    "memory_utilization_percent": mem_pct,
                })
        except Exception:
            pass
        return gpus


# AMD

class AMDAMDSMICollector(_BaseGPUCollector):
    vendor = "AMD"
    name = "amdsmi"

    def __init__(self):
        self.initialized = False

    def available(self) -> bool:
        if _amdsmi is None:
            return False
        try:
            _amdsmi.amdsmi_init()
            self.initialized = True
            return True
        except Exception:
            return False

    def collect(self) -> List[Dict[str, Any]]:
        if not self.initialized:
            return []
        gpus: List[Dict[str, Any]] = []
        try:
            # amdsmi python API (simplified best-effort)
            sockets = _amdsmi.amdsmi_get_socket_handles()
            for s in sockets:
                for dev in _amdsmi.amdsmi_get_processor_handles(s):
                    try:
                        name = _safe_str(_amdsmi.amdsmi_get_gpu_vendor_name(dev)) + " " + _safe_str(_amdsmi.amdsmi_get_gpu_name(dev))
                    except Exception:
                        name = "AMD GPU"
                    try:
                        util = _amdsmi.amdsmi_get_gpu_activity(dev).gfx_activity  # percent
                    except Exception:
                        util = None
                    try:
                        mem = _amdsmi.amdsmi_get_gpu_memory_usage(dev)  # returns bytes used & total
                        mem_pct = round((mem.used / mem.total) * 100.0, 2) if getattr(mem, "total", 0) else None
                    except Exception:
                        mem_pct = None
                    gpus.append({
                        "vendor": self.vendor,
                        "index": _safe_idx(dev),
                        "name": name.strip(),
                        "utilization_percent": float(util) if util is not None else None,
                        "memory_utilization_percent": mem_pct,
                    })
        except Exception:
            pass
        return gpus


class AMDROCmSMICollector(_BaseGPUCollector):
    vendor = "AMD"
    name = "rocm-smi"

    def __init__(self):
        self.cmd = shutil.which("rocm-smi")

    def available(self) -> bool:
        return self.cmd is not None

    def collect(self) -> List[Dict[str, Any]]:
        if not self.cmd:
            return []
       
        try:
            out = subprocess.check_output(
                [self.cmd, "--showuse", "--showmemuse", "--json"],
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
                        util = _extract_pct_any(c, ["GPU use (%)", "GPU use %", "GPU use"])
                        mem_used = _to_float(c.get("VRAM use (MiB)")) or _to_float(c.get("VRAM use (MB)"))
                        mem_total = _to_float(c.get("VRAM total (MiB)")) or _to_float(c.get("VRAM total (MB)"))
                        mem_pct = round((mem_used / mem_total) * 100.0, 2) if (mem_used and mem_total) else None
                        gpus.append({
                            "vendor": self.vendor,
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

        # Fallback
        try:
            out = subprocess.check_output(
                [self.cmd, "--showuse", "--showmemuse"],
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
                    
                    # Start of new GPU block
                    if current:
                        gpus.append(current)
                    idx += 1
                    current = {"vendor": self.vendor, "index": idx, "name": "AMD GPU",
                               "utilization_percent": None, "memory_utilization_percent": None}
                if "%" in ls and "GPU use" in ls:
                    current["utilization_percent"] = _extract_first_pct(ls)
                if "VRAM" in ls and "(" in ls and ")" in ls and "%" in ls:
                    current["memory_utilization_percent"] = _extract_first_pct(ls)
            if current:
                gpus.append(current)
            return gpus
        except Exception:
            return []


# Intel

class IntelGpuTopCollector(_BaseGPUCollector):
    vendor = "Intel"
    name = "intel_gpu_top"

    def __init__(self):
        self.cmd = shutil.which("intel_gpu_top")

    def available(self) -> bool:
        return self.cmd is not None

    def collect(self) -> List[Dict[str, Any]]:
        if not self.cmd:
            return []
        
        try:
            
            out = subprocess.check_output(
                [self.cmd, "-J", "-s", "100", "-d", "1"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            # Find first complete JSON object
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
                "vendor": self.vendor,
                "index": 0,
                "name": "Intel GPU",
                "utilization_percent": util,
                "memory_utilization_percent": None,  
            }]
        except Exception:
            return []


# macOS

class ApplePowermetricsCollector(_BaseGPUCollector):
    vendor = "Apple"
    name = "powermetrics"

    def __init__(self):
        self.cmd = shutil.which("powermetrics")
        self.is_macos = (platform.system() == "Darwin")

    def available(self) -> bool:
        return self.is_macos and self.cmd is not None

    def collect(self) -> List[Dict[str, Any]]:
        if not self.available():
            return []
     
        try:
            out = subprocess.check_output(
                [self.cmd, "--show-gpu", "-n", "1"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            util = None
            for line in out.splitlines():
                if "GPU Active" in line and "%" in line:
                    util = _extract_first_pct(line)
                    break
            return [{
                "vendor": self.vendor,
                "index": 0,
                "name": "Apple GPU",
                "utilization_percent": util,
                "memory_utilization_percent": None,
            }]
        except Exception:
            return []


# Aggregator

def _safe_str(x):
    try:
        if isinstance(x, bytes):
            return x.decode(errors="ignore")
        return str(x)
    except Exception:
        return "GPU"

def _safe_idx(x):
    try:
        return int(x)
    except Exception:
        return None

def _to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "").replace("MiB", "").replace("MB", "").replace("GiB", "").replace("GB", "")
        return float(s)
    except Exception:
        return None

def _extract_first_pct(s: str) -> Optional[float]:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None

def _extract_pct_any(d: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        f = _to_float(v)
        if f is not None:
            return f
    return None

class GPUBackend:
   
    #Try each vendor path in order; for each vendor, stop at the first working collector.
  
    def __init__(self) -> None:
        self.nvidia_chain = [NvidiaNVMLCollector(), NvidiaSmiCLICollector(), NvidiaGPUtilCollector()]
        self.amd_chain    = [AMDAMDSMICollector(), AMDROCmSMICollector()]
        self.intel_chain  = [IntelGpuTopCollector()]
        self.apple_chain  = [ApplePowermetricsCollector()]

    def collect(self) -> List[Dict[str, Any]]:
        all_gpus: List[Dict[str, Any]] = []

        
        for c in self.nvidia_chain:
            if c.available():
                g = c.collect()
                if g:
                    all_gpus.extend(g)
                break

      
        for c in self.amd_chain:
            if c.available():
                g = c.collect()
                if g:
                    all_gpus.extend(g)
                break

       
        for c in self.intel_chain:
            if c.available():
                g = c.collect()
                if g:
                    all_gpus.extend(g)
                break

        
        for c in self.apple_chain:
            if c.available():
                g = c.collect()
                if g:
                    all_gpus.extend(g)
                break

        return all_gpus


# Main Collector

class SystemMetricsCollector:
    def __init__(self, run_dir: Optional[str] = None, interval: float = 5.0, file_name: str = "system_metrics.json") -> None:
        _lazy_imports()
        self.interval = max(0.1, float(interval))
        self.run_dir = run_dir or self._default_run_dir()
        self.file_path = os.path.join(self.run_dir, file_name)
        self._stop = False
        self._gpu = GPUBackend()
        os.makedirs(self.run_dir, exist_ok=True)

        
        try:
            _psutil.cpu_percent(interval=None)
        except Exception:
            pass

       
        signal.signal(signal.SIGINT, self._handle_sig)
        signal.signal(signal.SIGTERM, self._handle_sig)

    def _default_run_dir(self) -> str:
        
        local = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join("data", "raw", f"run_{local}")

    def _handle_sig(self, signum, frame):
        print("\nReceived signal, stopping...", file=sys.stderr)
        self._stop = True

    def sample_once(self) -> Dict[str, Any]:
        vm = _psutil.virtual_memory()
        cpu_pct = _psutil.cpu_percent(interval=None)

        record = {
            "timestamp": _iso_now(),
            "cpu": {"utilization_percent": float(cpu_pct)},
            "ram": {"percent": float(vm.percent)},
            "gpus": self._gpu.collect(),  
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
                    cpu = rec["cpu"]["utilization_percent"]
                    ram_pct = rec["ram"]["percent"]
                    gpu_str = " / ".join(
                        f"{g.get('vendor')}[{g.get('index')}]: {g.get('utilization_percent')}% (mem {g.get('memory_utilization_percent')}%)"
                        if g.get("memory_utilization_percent") is not None
                        else f"{g.get('vendor')}[{g.get('index')}]: {g.get('utilization_percent')}%"
                        for g in rec.get("gpus", [])
                        if g.get("utilization_percent") is not None
                    ) or "no-gpu"
                    print(f"[{rec['timestamp']}] CPU {cpu:.1f}% | RAM {ram_pct:.1f}% | {gpu_str}")

                time.sleep(self.interval)

        print(f"Stopped. Samples written: {written}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect CPU/GPU/RAM utilization percentages to JSON Lines"
    )
    p.add_argument("--interval", type=float, default=5.0, help="Sampling interval in 5 sec")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Existing or new run directory under data/raw/. Default: creates a new timestamped run dir.")
    p.add_argument("--quiet", action="store_true", help="Less console output.")
    return p.parse_args()

def main():
    args = parse_args()
    collector = SystemMetricsCollector(run_dir=args.run_dir, interval=args.interval)
    collector.run(verbose=not args.quiet)

if __name__ == "__main__":
    main()

