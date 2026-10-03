"""Small NVIDIA/process telemetry helpers used by benchmark commands."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading
import time


def gpu_snapshot() -> list[dict[str, str]]:
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,pci.bus_id,name,driver_version,memory.total,memory.used"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode:
        return []
    keys = [
        "index",
        "pci_bus_id",
        "name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
    ]
    output: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == len(keys):
            output.append(dict(zip(keys, values)))
    return output


def process_rss_kib(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


class ResourceSampler:
    def __init__(self, pid: int | None, period_s: float = 0.5) -> None:
        self.pid = pid
        self.period_s = period_s
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="llama-model-manager-resource-sampler",
            daemon=True,
        )

    def take_sample(self) -> None:
        self.samples.append({
            "monotonic_s": time.monotonic(),
            "gpus": gpu_snapshot(),
            "rss_kib": process_rss_kib(self.pid),
        })

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.take_sample()
            self._stop.wait(self.period_s)

    def start(self) -> None:
        self.take_sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.period_s + 2)
        self.take_sample()

    def peak(self) -> dict[str, object]:
        gpu_peaks: dict[str, int] = {}
        for sample in self.samples:
            gpus = sample.get("gpus")
            if not isinstance(gpus, list):
                continue
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                try:
                    used = int(gpu["memory_used_mib"])
                    index = str(gpu["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                gpu_peaks[index] = max(gpu_peaks.get(index, 0), used)
        rss = [
            sample["rss_kib"]
            for sample in self.samples
            if isinstance(sample.get("rss_kib"), int)
        ]
        return {
            "sample_count": len(self.samples),
            "gpu_vram_peak_mib_by_index": gpu_peaks,
            "server_rss_peak_kib": max(rss) if rss else None,
        }
