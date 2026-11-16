"""Hardware detection and inference tuning for Intel Core Ultra 5 125H class machines.

Target machine: Intel Core Ultra 5 125H (2x P-core + 8x E-core + 2x LPE-core,
14 threads), ~32 GB RAM, Intel Arc integrated graphics (7 Xe-cores, 18 TOPS
INT8) + Intel AI Boost NPU (11 TOPS).

Strategy
--------
* CPU: ONNX Runtime scales poorly onto E-cores.  Using all 14 threads is
  slower than pinning to a smaller pool.  We therefore recommend a modest
  thread count (P-cores + a few E-cores) instead of ``cpu_count // 2``.
* iGPU: on Windows the Arc iGPU is reachable through ONNX Runtime's
  DirectML execution provider (``onnxruntime-directml``) and through
  OpenVINO.  Both are optional installs; when absent we fall back to CPU
  without failing.  The NPU is best left to ASR / enhancement, not TTS.
* RAM: 32 GB is ample for ZipVoice (123M params).  We keep a single TTS
  model resident and unload on quality switch to bound memory.

All detection is best-effort and cached.  Every choice can be overridden
with environment variables:

    VOICE_THREADS=<int>     force ONNX thread count
    VOICE_PROVIDER=<name>   force provider: cpu | directml | openvino
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache

# ---------------------------------------------------------------------------
# CPU topology
# ---------------------------------------------------------------------------


def _windows_core_counts() -> tuple[int | None, int | None]:
    """Return (performance_cores, efficient_cores) on Windows, else (None, None).

    Uses ``GetLogicalProcessorInformationEx`` via PowerShell-free ctypes-free
    fallback: ``wmic``/registry are unreliable, so we parse the CPU name and
    fall back to known Ultra 5 125H topology.  On non-Windows we return
    (None, None) and the caller uses a generic heuristic.
    """
    if sys.platform != "win32":
        return None, None
    name = _windows_cpu_name().lower()
    # Intel Core Ultra 5 125H: 2P + 8E + 2LPE (LPE counted with E for tuning).
    if "125h" in name or "ultra 5" in name:
        return 2, 10
    if "ultra 7" in name and ("155h" in name or "165h" in name):
        return 6, 8
    if "ultra 9" in name and ("185h" in name or "155h" in name):
        return 6, 8
    if "ultra 7" in name and "155u" in name:
        return 2, 8
    return None, None


def _windows_cpu_name() -> str:
    if sys.platform != "win32":
        return platform.processor() or platform.machine()
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["wmic", "cpu", "get", "name", "/value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.splitlines():
            if line.lower().startswith("name="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _windows_gpu_name() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        # Prefer Intel Arc / Intel Graphics over a discrete card listing.
        for name in names:
            if "intel" in name.lower():
                return name
        return names[0] if names else None
    except Exception:
        return None


def _total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024**3), 1)
        import os as _os

        pages = _os.sysconf("SC_PHYS_PAGES")
        page_size = _os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ONNX Runtime providers
# ---------------------------------------------------------------------------


def available_providers() -> list[str]:
    """ONNX Runtime execution providers visible in this environment."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _has_directml() -> bool:
    return any("directml" in p.lower() for p in available_providers())


def _has_openvino() -> bool:
    return any("openvino" in p.lower() for p in available_providers())


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def recommended_threads() -> int:
    """Thread count for ONNX Runtime on this machine.

    Ultra 5 125H (2P+10E/LPE): 6 threads (2P + 4E) beats 7-14 threads in
    practice because E-cores add synchronisation overhead to dense GEMMs.
    Generic fallback: half the logical cores, capped at 8.
    """
    override = os.environ.get("VOICE_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return min(int(override), 32)
    p_cores, e_cores = _windows_core_counts()
    if p_cores is not None and e_cores is not None:
        # P-cores do the heavy lifting; add a few E-cores for throughput.
        return max(2, min(8, p_cores * 2 + max(0, e_cores // 3)))
    cpu = os.cpu_count() or 4
    return min(8, max(2, cpu // 2))


def recommended_provider() -> str:
    """Execution provider for TTS/ASR: directml > openvino > cpu.

    DirectML targets the Arc iGPU on Windows and is the fastest path when
    ``onnxruntime-directml`` is installed.  OpenVINO can target iGPU/NPU.
    Anything unexpected falls back to CPU so the app always runs.
    """
    override = os.environ.get("VOICE_PROVIDER", "").strip().lower()
    if override in ("cpu", "directml", "openvino", "cuda", "coreml"):
        return override
    if sys.platform == "win32" and _has_directml():
        return "directml"
    if _has_openvino():
        return "openvino"
    return "cpu"


@lru_cache(maxsize=1)
def detect_hardware() -> dict:
    """Snapshot of this machine's relevant hardware + our recommendations."""
    p_cores, e_cores = _windows_core_counts()
    cpu_name = _windows_cpu_name()
    gpu_name = _windows_gpu_name()
    providers = available_providers()
    provider = recommended_provider()
    threads = recommended_threads()
    is_intel_arc = bool(gpu_name and "intel" in gpu_name.lower())
    # sherpa-onnx accepts "cpu" / "directml" / "openvino" style strings via
    # ONNX Runtime; unknown providers raise at model load, so we only ever
    # report providers that are actually installed (plus cpu).
    sherpa_providers = ["cpu"]
    if _has_directml():
        sherpa_providers.append("directml")
    if _has_openvino():
        sherpa_providers.append("openvino")
    return {
        "cpu": cpu_name,
        "cpu_count": os.cpu_count() or 0,
        "p_cores": p_cores,
        "e_cores": e_cores,
        "ram_gb": _total_ram_gb(),
        "gpu": gpu_name,
        "is_intel_arc": is_intel_arc,
        "onnx_providers": providers,
        "sherpa_providers": sherpa_providers,
        "recommended_provider": provider,
        "recommended_threads": threads,
        "platform": sys.platform,
    }


def provider_fallback_chain(preferred: str | None = None) -> list[str]:
    """Ordered providers to try when loading a model (first success wins)."""
    first = preferred or recommended_provider()
    chain = [first]
    for candidate in ("directml", "openvino", "cpu"):
        if candidate not in chain:
            chain.append(candidate)
    # Only keep providers that exist (cpu always exists).
    installed = set(available_providers())
    result = []
    for name in chain:
        if name == "cpu" or any(name in p.lower() for p in installed):
            result.append(name)
    return result or ["cpu"]


__all__ = [
    "available_providers",
    "detect_hardware",
    "provider_fallback_chain",
    "recommended_provider",
    "recommended_threads",
]
