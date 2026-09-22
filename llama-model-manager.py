#!/usr/bin/env python3
"""Explicit llama.cpp model launcher and reproducible benchmark runner.

Reads paths and installation settings from ~/.config/llama-model-manager/config.ini.
Daily presets default to ~/llm-workbench/configs/presets.yaml and fall back to the
legacy ~/.config/llama.cpp/model-presets.ini when no YAML presets are found.  The loader supports a ``defaults`` section: every preset
merges the defaults dict, overriding only the keys it defines.

The original INI-based loader (configparser, ``[Section]`` syntax) is preserved
as a compatibility fallback so existing presets keep working.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import configparser
import copy
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOME = Path.home()
XDG_CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")).expanduser()
MANAGER_CONFIG_PATH = Path(os.environ.get(
    "LLAMA_MODEL_MANAGER_CONFIG",
    XDG_CONFIG_HOME / "llama-model-manager" / "config.ini",
)).expanduser()

DEFAULT_MANAGER_CONFIG = {
    "install": {
        "install_dir": str(HOME / ".local/share/llama-model-manager"),
        "bin_dir": str(HOME / ".local/bin"),
    },
    "paths": {
        "models_dir": str(HOME / "Models"),
        "presets_yaml": str(HOME / "llm-workbench/configs/presets.yaml"),
        "presets_ini": str(HOME / ".config/llama.cpp/model-presets.ini"),
        "llama_server_bin": str(HOME / "Applications/llama.cpp/build/bin/llama-server"),
        "state_dir": str(HOME / ".local/state/llama-model-launcher"),
        "benchmark_case": str(HOME / ".config/llama.cpp/benchmarks/flash-next-v1.json"),
        "benchmark_results": str(HOME / ".local/share/llama-benchmarks/results.jsonl"),
    },
    "server": {
        "host": "0.0.0.0",
        "port": "1919",
    },
}


def _manager_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict(DEFAULT_MANAGER_CONFIG)
    if MANAGER_CONFIG_PATH.is_file():
        parser.read(MANAGER_CONFIG_PATH, encoding="utf-8")
    return parser


def _config_value(section: str, key: str, env_var: str | None = None) -> str:
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    parser = _manager_config()
    return parser.get(section, key, fallback=DEFAULT_MANAGER_CONFIG[section][key])


def _config_path(section: str, key: str, env_var: str | None = None) -> Path:
    return Path(_config_value(section, key, env_var)).expanduser()


def _write_manager_config(parser: configparser.ConfigParser) -> None:
    MANAGER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANAGER_CONFIG_PATH.with_suffix(MANAGER_CONFIG_PATH.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    os.replace(temp_path, MANAGER_CONFIG_PATH)


def _prompt_path(label: str, current: str) -> str:
    answer = input(f"{label} [{current}]: ").strip()
    return str(Path(answer or current).expanduser())


def setup_manager_config() -> int:
    parser = _manager_config()
    print("LLAMA MODEL MANAGER SETUP")
    print()
    print(f"Config: {MANAGER_CONFIG_PATH}")
    print("Press Enter to keep the current value.")
    print()

    parser["install"]["install_dir"] = _prompt_path(
        "Install directory", parser["install"]["install_dir"]
    )
    parser["install"]["bin_dir"] = _prompt_path(
        "Command directory", parser["install"]["bin_dir"]
    )
    parser["paths"]["models_dir"] = _prompt_path(
        "Models directory", parser["paths"]["models_dir"]
    )
    parser["paths"]["presets_yaml"] = _prompt_path(
        "Presets YAML", parser["paths"]["presets_yaml"]
    )
    parser["paths"]["presets_ini"] = _prompt_path(
        "Legacy presets INI", parser["paths"]["presets_ini"]
    )
    parser["paths"]["llama_server_bin"] = _prompt_path(
        "llama-server binary", parser["paths"]["llama_server_bin"]
    )
    parser["paths"]["state_dir"] = _prompt_path(
        "State directory", parser["paths"]["state_dir"]
    )
    parser["paths"]["benchmark_case"] = _prompt_path(
        "Benchmark case", parser["paths"]["benchmark_case"]
    )
    parser["paths"]["benchmark_results"] = _prompt_path(
        "Benchmark results", parser["paths"]["benchmark_results"]
    )

    host = input(f"Server host [{parser['server']['host']}]: ").strip()
    port = input(f"Server port [{parser['server']['port']}]: ").strip()
    if host:
        parser["server"]["host"] = host
    if port:
        try:
            parsed_port = int(port)
        except ValueError as exc:
            raise LauncherError(f"Server port must be an integer, got {port!r}") from exc
        if not 1 <= parsed_port <= 65535:
            raise LauncherError("Server port must be between 1 and 65535")
        parser["server"]["port"] = str(parsed_port)

    _write_manager_config(parser)
    print()
    print(f"Wrote configuration to {MANAGER_CONFIG_PATH}")
    print("Re-run install.sh if you changed the install or command directory.")
    return 0


def show_manager_config() -> int:
    parser = _manager_config()
    print(f"Config: {MANAGER_CONFIG_PATH}")
    print()
    for section in parser.sections():
        print(f"[{section}]")
        for key, value in parser[section].items():
            print(f"{key} = {value}")
        print()
    return 0


MODELS_DIR = _config_path("paths", "models_dir", "LLAMA_MODELS_DIR")
YAML_PATH = _config_path("paths", "presets_yaml", "LLAMA_MODELS_YAML")
INI_PATH = _config_path("paths", "presets_ini", "LLAMA_MODELS_INI")
SERVER_BIN = _config_path("paths", "llama_server_bin", "LLAMA_SERVER_BIN")
STATE_DIR = _config_path("paths", "state_dir", "LLAMA_MODEL_STATE_DIR")
BENCH_CASE = _config_path("paths", "benchmark_case", "LLAMA_BENCH_CASE")
BENCH_RESULTS = _config_path("paths", "benchmark_results", "LLAMA_BENCH_RESULTS")
DEFAULT_HOST = _config_value("server", "host", "LLAMA_SERVER_HOST")
try:
    DEFAULT_PORT = int(_config_value("server", "port", "LLAMA_SERVER_PORT"))
except ValueError as exc:
    raise RuntimeError("Configured server port is not an integer") from exc

# These affect launcher behavior rather than becoming llama-server arguments.
PRESET_ONLY_KEYS = {
    "load-on-startup", "stop-timeout", "dedup-cache-models", "version", "server-bin",
    "alias", "host", "port",
}
SUPPORTED_KEYS = {
    "model", "ctx-size", "parallel", "threads", "threads-batch", "n-gpu-layers", "device",
    "split-mode", "tensor-split", "main-gpu", "flash-attn", "cache-type-k",
    "cache-type-v", "batch-size", "ubatch-size", "n-cpu-moe", "fit",
    "fit-target", "mmap", "no-mmap", "mlock", "repack", "no-repack",
    "load-mode", "lazy-mode", "spec-type", "spec-draft-n-max", "model-draft",
    "spec-draft-model", "jinja", "metrics", "slot-save-path", "server-bin",
    "phase-aware-workspace", "live-context-workspace", "backend-sampling",
    "decode-overlap", "decode-boundary-overlap", "ple-prefetch", "experimental-logs",
    "moe-expert-cache-size", "moe-expert-cache-host-pinned-mb",
}
BOOL_FLAGS = {
    "mmap": ("--mmap", "--no-mmap"),
    "no-mmap": ("--no-mmap", "--mmap"),
    "mlock": ("--mlock", None),
    "repack": ("--repack", "--no-repack"),
    "no-repack": ("--no-repack", "--repack"),
    "jinja": ("--jinja", "--no-jinja"),
    "metrics": ("--metrics", None),
    "phase-aware-workspace": ("--phase-aware-workspace", None),
    "live-context-workspace": ("--live-context-workspace", None),
    "backend-sampling": ("--backend-sampling", None),
    "decode-overlap": ("--decode-overlap", None),
    "decode-boundary-overlap": ("--decode-boundary-overlap", None),
    "ple-prefetch": ("--ple-prefetch", None),
    "experimental-logs": ("--experimental-logs", None),
}
TRUE_VALUES = {"1", "true", "on", "yes", "enabled"}
FALSE_VALUES = {"0", "false", "off", "no", "disabled"}


class LauncherError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# YAML preset loader (primary)
# ---------------------------------------------------------------------------

def _yaml_load_presets(yaml_path: Path) -> dict[str, dict[str, str]]:
    """Parse the YAML preset file and return {preset_name: {key: value}}.

    The file must have a ``defaults`` mapping and one or more named sections.
    Each named section inherits the defaults, overriding any keys it defines.
    """
    # Minimal YAML loader — no external dependencies.
    # Supports:
    #   key: value                (string, number, quoted)
    #   key: "value"              (double-quoted string)
    #   key: 'value'              (single-quoted string)
    #   key: "true"/"on"/etc     (booleans stored as strings — caller decides)
    #   # comments                (ignored)
    #   # blank lines             (ignored)
    #
    # This intentionally avoids the PyYAML dependency; the preset format is
    # simple enough for a hand-rolled parser.

    if not yaml_path.is_file():
        raise LauncherError(f"YAML preset file not found: {yaml_path}")

    content = yaml_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Parse into top-level sections
    current_section: str | None = None
    sections: dict[str, dict[str, str]] = {}
    current_items: dict[str, str] = {}

    def flush_section():
        nonlocal current_section, current_items
        if current_section is not None:
            sections[current_section] = dict(current_items)
        current_section = None
        current_items = {}

    for raw_line in lines:
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Detect top-level key (no leading whitespace in original line)
        if raw_line[0] not in (" ", "\t") and line.endswith(":"):
            flush_section()
            current_section = line[:-1].strip()
            current_items = {}
            continue

        # Key-value pair within current section
        if current_section is None:
            continue

        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            if key and value is not None:
                current_items[key] = value

    flush_section()

    if not sections:
        raise LauncherError(f"No presets found in {yaml_path}")

    # Extract defaults
    defaults = sections.get("defaults", {})

    # Merge: each preset = defaults + override
    presets: dict[str, dict[str, str]] = {}
    for name, items in sections.items():
        if name == "defaults":
            continue
        merged = dict(defaults)
        merged.update(items)
        if "model" not in merged:
            raise LauncherError(f"Preset {name!r} has no model path")
        # Validate keys
        unknown = sorted(set(merged) - SUPPORTED_KEYS - PRESET_ONLY_KEYS)
        if unknown:
            raise LauncherError(f"Preset {name!r} has unsupported keys: {', '.join(unknown)}")
        presets[name] = merged

    if not presets:
        raise LauncherError(f"No named presets found in {yaml_path}")

    return presets


# ---------------------------------------------------------------------------
# Legacy INI preset loader (compatibility fallback)
# ---------------------------------------------------------------------------

def _ini_load_presets(ini_path: Path) -> dict[str, dict[str, str]]:
    """Parse the original INI-style presets file."""
    if not ini_path.is_file():
        raise LauncherError(f"Preset INI not found: {ini_path}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with ini_path.open(encoding="utf-8") as handle:
        parser.read_file(handle)

    global_values = dict(parser["*"]) if parser.has_section("*") else {}
    presets: dict[str, dict[str, str]] = {}
    for name in parser.sections():
        if name in {"*", "default"}:
            continue
        values = dict(global_values)
        values.update(dict(parser[name]))
        unknown = sorted(set(values) - SUPPORTED_KEYS - PRESET_ONLY_KEYS)
        if unknown:
            raise LauncherError(f"Preset {name!r} has unsupported keys: {', '.join(unknown)}")
        if not values.get("model"):
            raise LauncherError(f"Preset {name!r} has no model path")
        presets[name] = values
    if not presets:
        raise LauncherError(f"No named presets found in {ini_path}")
    return presets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_presets() -> dict[str, dict[str, str]]:
    """Load presets, trying YAML first, then legacy INI.

    YAML is preferred because it supports a ``defaults`` section that is
    automatically inherited by every preset, making per-model overrides
    shorter to write and easier to maintain.
    """
    try:
        return _yaml_load_presets(YAML_PATH)
    except LauncherError:
        pass
    try:
        return _ini_load_presets(INI_PATH)
    except LauncherError:
        pass
    raise LauncherError(
        "No presets found. Tried YAML at "
        f"{YAML_PATH} and INI at {INI_PATH}"
    )


def selected_preset(name: str) -> dict[str, str]:
    presets = load_presets()
    if name not in presets:
        raise LauncherError(f"Unknown preset {name!r}. Available: {', '.join(presets)}")
    return presets[name]


def argv_for_preset(name: str, values: dict[str, str]) -> list[str]:
    # A curated preset may opt into a separately built llama.cpp binary. This
    # keeps a tested fork tied to the model that needs it without changing the
    # global default used by other presets.
    server_bin = Path(values.get("server-bin", str(SERVER_BIN))).expanduser()
    if not server_bin.is_file() or not os.access(server_bin, os.X_OK):
        raise LauncherError(f"llama-server is not executable: {server_bin}")
    argv = [str(server_bin), "--host", DEFAULT_HOST, "--port", str(DEFAULT_PORT), "--alias", name]
    for key, raw_value in values.items():
        if key in PRESET_ONLY_KEYS:
            continue
        if key == "preset-name":
            continue
        if key in BOOL_FLAGS:
            positive, negative = BOOL_FLAGS[key]
            if parse_bool(raw_value):
                argv.append(positive)
            elif negative:
                argv.append(negative)
            continue
        argv.extend([f"--{key}", raw_value])
    return argv


def preset_description(name: str) -> dict[str, object]:
    values = selected_preset(name)
    model_path = Path(values["model"]).expanduser()
    return {
        "preset": name,
        "ini_path": str(INI_PATH),
        "yaml_path": str(YAML_PATH) if YAML_PATH.is_file() else None,
        "model_path": str(model_path),
        "model_exists": model_path.is_file(),
        "model_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "server_bin": str(Path(values.get("server-bin", str(SERVER_BIN))).expanduser()),
        "parameters": values,
        "argv": argv_for_preset(name, values),
    }


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=False, **kwargs)


def listener_pids(port: int = DEFAULT_PORT) -> list[int]:
    completed = run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True)
    return sorted({int(pid) for pid in re.findall(r"pid=(\d+)", completed.stdout)})


def commandline_for_pid(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return ""


def is_llama_server(pid: int) -> bool:
    cmdline = commandline_for_pid(pid)
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except FileNotFoundError:
        executable = ""
    return "llama-server" in cmdline or Path(executable).name == "llama-server"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_listener(port: int = DEFAULT_PORT, timeout_s: int = 30) -> None:
    pids = listener_pids(port)
    if not pids:
        return
    non_llama = [pid for pid in pids if not is_llama_server(pid)]
    if non_llama:
        raise LauncherError(
            f"Refusing to stop non-llama process(es) listening on :{port}: {non_llama}"
        )
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        live = [pid for pid in pids if pid_alive(pid)]
        if not live and not listener_pids(port):
            return
        time.sleep(0.25)
    live = [pid for pid in pids if pid_alive(pid)]
    if live:
        raise LauncherError(f"llama-server did not stop within {timeout_s}s: {live}")


def http_get(path: str, timeout_s: int = 10) -> object:
    with urllib.request.urlopen(f"http://127.0.0.1:{DEFAULT_PORT}{path}", timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post(path: str, payload: dict[str, object], timeout_s: int = 300) -> object:
    request = urllib.request.Request(
        f"http://127.0.0.1:{DEFAULT_PORT}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_active(name: str, process: subprocess.Popen[object], timeout_s: int = 900) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error = "not yet queried"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LauncherError(f"llama-server exited before readiness (exit {process.returncode})")
        try:
            data = http_get("/v1/models", timeout_s=5)
            models = data.get("data", []) if isinstance(data, dict) else []
            if any(model.get("id") == name for model in models if isinstance(model, dict)):
                return data
            last_error = f"server ready but selected alias {name!r} not reported"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise LauncherError(f"Timed out waiting for {name!r} on :{DEFAULT_PORT}: {last_error}")


def state_paths() -> tuple[Path, Path]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / "server.pid", STATE_DIR / "server.log"


def start_preset(name: str, background: bool) -> int:
    description = preset_description(name)
    model_path = Path(description["model_path"])
    if not model_path.is_file():
        raise LauncherError(f"Configured model file is not present: {model_path}")
    stop_listener()
    pid_path, log_path = state_paths()
    argv = description["argv"]
    if background:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        try:
            wait_until_active(name, process)
        except Exception:
            if process.poll() is None:
                process.terminate()
            raise
        print(json.dumps({"status": "active", "preset": name, "pid": process.pid, "url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}", "log": str(log_path)}))
        return 0

    process = subprocess.Popen(argv)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    try:
        wait_until_active(name, process)
        print(f"Active preset: {name} on http://{DEFAULT_HOST}:{DEFAULT_PORT}", flush=True)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return process.wait()
    finally:
        if process.poll() is not None:
            pid_path.unlink(missing_ok=True)


def choose_with_fzf(query: str | None) -> str | None:
    names = list(load_presets())
    command = ["fzf", "--prompt=Model > ", "--height=80%", "--reverse", "--border"]
    if query:
        command.extend(["--query", query])
    try:
        completed = run(command, input="\n".join(names) + "\n", capture_output=True)
    except FileNotFoundError as exc:
        raise LauncherError("fzf is required for interactive model selection") from exc
    selected = completed.stdout.strip()
    return selected or None


def model_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Select and start an explicit llama.cpp preset")
    parser.add_argument("query", nargs="?", help="initial fzf query")
    parser.add_argument("--preset", help="start/show this exact preset without fzf")
    parser.add_argument("--background", action="store_true", help="start, verify, and detach")
    parser.add_argument("--list", action="store_true", help="list explicit preset names")
    parser.add_argument("--show", action="store_true", help="print resolved parameters and argv as JSON")
    parser.add_argument("--setup", action="store_true", help="interactively create or update manager configuration")
    parser.add_argument("--config", action="store_true", help="show resolved manager configuration")
    parser.add_argument(
        "--download",
        metavar="REPO",
        help="download a model from HuggingFace",
    )
    parser.add_argument(
        "--file",
        default="*.gguf",
        help="file or glob to download (default: *.gguf)",
    )
    parser.add_argument(
        "--quant",
        help="quantization directory name; inferred when possible",
    )
    parser.add_argument(
        "--name",
        help="preset name for promotion",
    )
    parser.add_argument(
        "--promote",
        metavar="LAB_PATH",
        help="promote a lab experiment to a daily preset",
    )
    args = parser.parse_args(argv)

    if args.setup:
        return setup_manager_config()
    if args.config:
        return show_manager_config()
    if args.download is not None:
        return download_model(
            args.download,
            file_pattern=args.file,
            quant=args.quant,
            name=args.name,
        )
    if args.promote is not None:
        return promote_model(
            [args.promote] + (["--name", args.name] if args.name else [])
        )
    if args.list:
        print("\n".join(load_presets()))
        return 0
    if args.show:
        if not args.preset:
            parser.error("--show requires --preset")
        print(json.dumps(preset_description(args.preset), indent=2, sort_keys=True))
        return 0
    if args.preset:
        name = args.preset
    else:
        name = choose_with_fzf(args.query)
        if not name:
            return 0
    return start_preset(name, args.background)


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise LauncherError(f"Expected boolean value, got {value!r}")


def command_output(args: list[str]) -> str:
    completed = run(args, capture_output=True)
    return "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)


def gpu_snapshot() -> list[dict[str, str]]:
    query = "index,pci.bus_id,name,driver_version,memory.total,memory.used"
    completed = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True)
    if completed.returncode:
        return []
    keys = ["index", "pci_bus_id", "name", "driver_version", "memory_total_mib", "memory_used_mib"]
    output = []
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
    except FileNotFoundError:
        return None
    return None


def active_pid() -> int | None:
    pid_path, _ = state_paths()
    try:
        pid = int(pid_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if pid_alive(pid) else None


class ResourceSampler:
    def __init__(self, pid: int | None, period_s: float = 0.5) -> None:
        self.pid = pid
        self.period_s = period_s
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

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
            for gpu in sample["gpus"]:
                try:
                    used = int(gpu["memory_used_mib"])
                except (KeyError, ValueError):
                    continue
                gpu_peaks[gpu["index"]] = max(gpu_peaks.get(gpu["index"], 0), used)
        rss = [sample["rss_kib"] for sample in self.samples if sample["rss_kib"] is not None]
        return {
            "sample_count": len(self.samples),
            "gpu_vram_peak_mib_by_index": gpu_peaks,
            "server_rss_peak_kib": max(rss) if rss else None,
        }


def prometheus_metrics() -> dict[str, float]:
    request = urllib.request.Request(f"http://127.0.0.1:{DEFAULT_PORT}/metrics")
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, value = line.rsplit(" ", 1)
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def metric_delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if key not in after:
        return None
    return after[key] - before.get(key, 0.0)


def current_model_meta() -> dict[str, object] | None:
    data = http_get("/v1/models")
    if not isinstance(data, dict):
        return None
    items = data.get("data", [])
    if not items:
        return None
    first = items[0]
    return first if isinstance(first, dict) else None


def llama_build_info() -> dict[str, str]:
    return {
        "git_commit": command_output(["git", "-C", str(SERVER_BIN.parent.parent.parent), "rev-parse", "HEAD"]),
        "server_version": command_output([str(SERVER_BIN), "--version"]),
    }


def standard_performance_matrix() -> list[dict[str, object]]:
    """Luke-style fixed scenarios; targets are tokenizer-padded prompt content."""
    return [
        {"id": "decode-64in-256out", "prompt_tokens": 64, "max_tokens": 256, "warmup_runs": 1, "measured_trials": 3, "kind": "decode"},
        {"id": "prefill-512in-1out", "prompt_tokens": 512, "max_tokens": 1, "warmup_runs": 1, "measured_trials": 3, "kind": "prefill"},
        {"id": "prefill-2k-in-1out", "prompt_tokens": 2048, "max_tokens": 1, "warmup_runs": 1, "measured_trials": 3, "kind": "prefill"},
        {"id": "prefill-8k-in-1out", "prompt_tokens": 8192, "max_tokens": 1, "warmup_runs": 1, "measured_trials": 3, "kind": "prefill"},
        {"id": "combined-4k-in-128out", "prompt_tokens": 4096, "max_tokens": 128, "warmup_runs": 1, "measured_trials": 3, "kind": "combined"},
    ]


def median_numeric(trials: list[dict[str, object]], section: str, key: str) -> float | int | None:
    values = [trial.get(section, {}).get(key) for trial in trials]
    numeric = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(numeric) if numeric else None


def cached_performance_matrix() -> list[dict[str, object]]:
    """Cache-hit companion scenarios for representative medium/long inputs."""
    return [
        {
            "id": "cache-2k-in-1out",
            "prompt_tokens": 2048,
            "max_tokens": 1,
            "warmup_runs": 1,
            "measured_trials": 3,
        },
        {
            "id": "cache-8k-in-1out",
            "prompt_tokens": 8192,
            "max_tokens": 1,
            "warmup_runs": 1,
            "measured_trials": 3,
        },
        {
            "id": "cache-4k-in-128out",
            "prompt_tokens": 4096,
            "max_tokens": 128,
            "warmup_runs": 1,
            "measured_trials": 3,
        },
    ]


def summarize_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    if not trials:
        raise LauncherError("Cannot summarize zero measured trials")
    return {
        "measured_trials": len(trials),
        "prompt_tokens_median": median_numeric(trials, "usage", "prompt_tokens"),
        "generated_tokens_median": median_numeric(trials, "usage", "generated_tokens"),
        "cached_tokens_median": median_numeric(trials, "usage", "cached_tokens"),
        "prompt_tokens_per_second_median": median_numeric(trials, "native_timings", "prompt_per_second"),
        "generation_tokens_per_second_median": median_numeric(trials, "native_timings", "predicted_per_second"),
        "prompt_ms_median": median_numeric(trials, "native_timings", "prompt_ms"),
        "generation_ms_median": median_numeric(trials, "native_timings", "generation_ms"),
    }


def summarize_cached_trials(trials: list[dict[str, object]], target_prompt_tokens: int) -> dict[str, object]:
    if target_prompt_tokens <= 0:
        raise LauncherError("Cached-trial target prompt tokens must be positive")
    summary = summarize_trials(trials)
    cached_tokens = summary["cached_tokens_median"]
    server_prompt_tokens = summary["prompt_tokens_median"]
    summary["cache_hit_ratio_denominator"] = "server_prompt_tokens_median"
    summary["cache_hit_ratio_median"] = (
        cached_tokens / server_prompt_tokens
        if isinstance(cached_tokens, (int, float)) and isinstance(server_prompt_tokens, (int, float)) and server_prompt_tokens > 0
        else None
    )
    return summary


def pad_prompt_to_tokens(target_tokens: int, token_count: callable, filler: str = " benchmark") -> str:
    """Construct filler text whose server-tokenized content exactly hits target_tokens."""
    if target_tokens <= 0:
        raise LauncherError("Prompt token target must be positive")
    if token_count(filler) <= 0:
        raise LauncherError("Benchmark filler does not tokenize to a positive token count")
    low, high = 0, 1
    while token_count(filler * high) < target_tokens:
        low, high = high, high * 2
    while low <= high:
        mid = (low + high) // 2
        candidate = filler * mid
        count = token_count(candidate)
        if count == target_tokens:
            return candidate
        if count < target_tokens:
            low = mid + 1
        else:
            high = mid - 1
    raise LauncherError(f"Could not construct an exact {target_tokens}-token benchmark prompt")


def matrix_request(
    preset: str,
    prompt: str,
    scenario: dict[str, object],
    cache_prompt: bool,
    slot_id: int,
) -> dict[str, object]:
    return {
        "model": preset,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": scenario["max_tokens"],
        "stream": False,
        "cache_prompt": cache_prompt,
        "id_slot": slot_id,
    }


def trial_from_response(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        raise LauncherError("Unexpected non-object chat completion response")
    usage = response.get("usage")
    timings = response.get("timings")
    if not isinstance(usage, dict) or not isinstance(timings, dict):
        raise LauncherError("Response omitted native usage or timings")
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    return {
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "cached_tokens": cached_tokens,
        },
        "native_timings": {
            "prompt_per_second": timings.get("prompt_per_second"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "generation_ms": timings.get("predicted_ms"),
        },
        "finish_reason": finish_reason,
    }


def run_performance_matrix(
    preset: str,
    token_count: callable,
    execute: callable,
    reset_slot: callable,
    slot_id: int = 0,
) -> dict[str, object]:
    """Run cold and cache-hit cases using injected server operations."""
    cold_rows: list[dict[str, object]] = []
    cached_rows: list[dict[str, object]] = []

    for scenario in standard_performance_matrix():
        prompt = pad_prompt_to_tokens(int(scenario["prompt_tokens"]), token_count)
        payload = matrix_request(preset, prompt, scenario, cache_prompt=False, slot_id=slot_id)
        reset_slot()
        warmup = trial_from_response(execute(payload))
        reset_slot()
        trials = [trial_from_response(execute(payload)) for _ in range(int(scenario["measured_trials"]))]
        cold_rows.append({
            "scenario": scenario,
            "content_tokens": token_count(prompt),
            "warmup": warmup,
            "trials": trials,
            "summary": summarize_trials(trials),
        })

    for scenario in cached_performance_matrix():
        prompt = pad_prompt_to_tokens(int(scenario["prompt_tokens"]), token_count)
        payload = matrix_request(preset, prompt, scenario, cache_prompt=True, slot_id=slot_id)
        reset_slot()
        prime = trial_from_response(execute(payload))
        trials = [trial_from_response(execute(payload)) for _ in range(int(scenario["measured_trials"]))]
        cached_rows.append({
            "scenario": scenario,
            "content_tokens": token_count(prompt),
            "prime": prime,
            "trials": trials,
            "summary": summarize_cached_trials(trials, int(scenario["prompt_tokens"])),
        })

    return {
        "protocol": "luke-style-matrix-plus-cache-v1",
        "preset": preset,
        "slot_id": slot_id,
        "cold": cold_rows,
        "cached": cached_rows,
    }


def matrix_benchmark(args: argparse.Namespace) -> int:
    """Run the matrix against the already active server without restarting it."""
    active = current_model_meta()
    if active.get("id") != args.preset:
        raise LauncherError(
            f"Matrix requires active preset {args.preset!r}; current model is {active.get('id')!r}. "
            "Start the explicit benchmark preset first."
        )
    resolved = preset_description(args.preset)

    def token_count(text: str) -> int:
        tokenized = http_post("/tokenize", {"content": text}, timeout_s=120)
        if not isinstance(tokenized, dict) or not isinstance(tokenized.get("tokens"), list):
            raise LauncherError("Server /tokenize response omitted a token list")
        return len(tokenized["tokens"])

    def execute(payload: dict[str, object]) -> object:
        return http_post("/v1/chat/completions", payload, timeout_s=900)

    def reset_slot() -> None:
        http_post(f"/slots/{args.slot}?action=erase", {}, timeout_s=60)

    pre_metrics = prometheus_metrics()
    pre_gpus = gpu_snapshot()
    pid = active_pid()
    sampler = ResourceSampler(pid)
    sampler.start()
    try:
        matrix = run_performance_matrix(args.preset, token_count, execute, reset_slot, args.slot)
    finally:
        sampler.stop()
    post_metrics = prometheus_metrics()
    post_gpus = gpu_snapshot()

    cache_medians = [row["summary"].get("cached_tokens_median") for row in matrix["cached"]]
    cache_verified = all(isinstance(value, (int, float)) and value > 0 for value in cache_medians)
    result = {
        "schema_version": 2,
        "kind": "luke-style-performance-matrix",
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_reference": "https://github.com/lukesdevlab/youtube/blob/main/performance-benchmark-overview.html",
        "matrix": matrix,
        "preset": args.preset,
        "model": {
            "path": resolved["model_path"],
            "filename": Path(str(resolved["model_path"])).name,
            "bytes": resolved["model_bytes"],
        },
        "llama_cpp": llama_build_info(),
        "resolved_launch": {
            "ini_path": resolved["ini_path"],
            "parameters": resolved["parameters"],
            "argv": resolved["argv"],
        },
        "server": {
            "url": f"http://127.0.0.1:{DEFAULT_PORT}",
            "model_info_before": active,            "model_info_after": current_model_meta(),
            "server_pid": pid,
        },
        "cache_validation": {
            "cache_prompt": True,
            "slot_id": args.slot,
            "cached_tokens_medians": cache_medians,
            "verified_nonzero_cache_hits": cache_verified,
        },
        "metrics_delta": {
            "prompt_tokens": metric_delta(pre_metrics, post_metrics, "llamacpp:prompt_tokens_total"),
            "prompt_seconds": metric_delta(pre_metrics, post_metrics, "llamacpp:prompt_seconds_total"),
            "generated_tokens": metric_delta(pre_metrics, post_metrics, "llamacpp:tokens_predicted_total"),
            "generated_seconds": metric_delta(pre_metrics, post_metrics, "llamacpp:tokens_predicted_seconds_total"),
        },
        "resources": {
            "gpu_before": pre_gpus,
            "gpu_after": post_gpus,
            "during_matrix_peak": sampler.peak(),
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps({
        "written": str(args.output),
        "run_id": result["run_id"],
        "preset": args.preset,
        "cold_cases": len(matrix["cold"]),
        "cached_cases": len(matrix["cached"]),
        "cache_verified": cache_verified,
    }, sort_keys=True))
    return 0


def benchmark_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a fixed llama.cpp benchmark and append JSONL")
    parser.add_argument("--preset", default="Qwen3.8-Flash-Next", help="explicit preset to benchmark")
    parser.add_argument("--matrix", action="store_true", help="run the fixed cold+cache performance matrix against the active server")
    parser.add_argument("--slot", type=int, default=0, help="dedicated llama-server slot for matrix cache tests")
    parser.add_argument("--case", type=Path, default=BENCH_CASE, help="versioned JSON test case")
    parser.add_argument("--output", type=Path, default=BENCH_RESULTS, help="JSONL results file")
    args = parser.parse_args(argv)
    if args.matrix:
        return matrix_benchmark(args)
    if not args.case.is_file():
        raise LauncherError(f"Benchmark case not found: {args.case}")
    case = json.loads(args.case.read_text(encoding="utf-8"))
    request_template = case.get("request")
    if not isinstance(request_template, dict):
        raise LauncherError("Benchmark case must contain a request object")

    start = run([sys.executable, str(Path(__file__)), "model", "--background", "--preset", args.preset], capture_output=True)
    if start.returncode:
        raise LauncherError(f"Could not start preset {args.preset!r}: {start.stderr or start.stdout}")
    resolved = preset_description(args.preset)
    pre_metrics = prometheus_metrics()
    pre_gpus = gpu_snapshot()
    pid = active_pid()
    sampler = ResourceSampler(pid)
    request_payload = copy.deepcopy(request_template)
    request_payload["model"] = args.preset
    sampler.start()
    try:
        response = http_post("/v1/chat/completions", request_payload, timeout_s=900)
    finally:
        sampler.stop()
    post_metrics = prometheus_metrics()
    post_gpus = gpu_snapshot()
    if not isinstance(response, dict):
        raise LauncherError("Unexpected non-object chat completion response")
    usage = response.get("usage", {})
    timings = response.get("timings", {})
    if not isinstance(usage, dict) or not isinstance(timings, dict):
        raise LauncherError("Response omitted native usage or timings")

    draft_tokens = metric_delta(pre_metrics, post_metrics, "llamacpp:spec_decode_num_draft_tokens_total")
    accepted_tokens = metric_delta(pre_metrics, post_metrics, "llamacpp:spec_decode_num_accepted_tokens_total")
    draft_steps = metric_delta(pre_metrics, post_metrics, "llamacpp:spec_decode_num_drafts_total")
    result = {
        "schema_version": 1,
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case": case,
        "preset": args.preset,
        "model": {
            "path": resolved["model_path"],
            "filename": Path(str(resolved["model_path"])).name,
            "bytes": resolved["model_bytes"],
        },
        "llama_cpp": llama_build_info(),
        "resolved_launch": {
            "ini_path": resolved["ini_path"],
            "parameters": resolved["parameters"],
            "argv": resolved["argv"],
        },
        "server": {
            "url": f"http://127.0.0.1:{DEFAULT_PORT}",
            "model_info": current_model_meta(),
            "server_pid": pid,
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_prompt_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens") if isinstance(usage.get("prompt_tokens_details"), dict) else None,
        },
        "native_timings": timings,
        "rates_tokens_per_second": {
            "prompt_prefill": timings.get("prompt_per_second"),
            "generation": timings.get("predicted_per_second"),
        },
        "metrics_delta": {
            "prompt_tokens": metric_delta(pre_metrics, post_metrics, "llamacpp:prompt_tokens_total"),
            "prompt_seconds": metric_delta(pre_metrics, post_metrics, "llamacpp:prompt_seconds_total"),
            "generated_tokens": metric_delta(pre_metrics, post_metrics, "llamacpp:tokens_predicted_total"),
            "generated_seconds": metric_delta(pre_metrics, post_metrics, "llamacpp:tokens_predicted_seconds_total"),
        },
        "speculative": {
            "spec_type": resolved["parameters"].get("spec-type", "none"),
            "spec_draft_n_max": resolved["parameters"].get("spec-draft-n-max"),
            "model_draft": resolved["parameters"].get("model-draft") or resolved["parameters"].get("spec-draft-model"),
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "verification_steps": draft_steps,
            "accepted_per_verification_step": (accepted_tokens / draft_steps) if draft_steps else None,
        },
        "resources": {
            "gpu_before": pre_gpus,
            "gpu_after": post_gpus,
            "during_request_peak": sampler.peak(),
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps({
        "written": str(args.output),
        "run_id": result["run_id"],
        "preset": args.preset,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "generated_tokens": result["usage"]["generated_tokens"],
        "prompt_tps": result["rates_tokens_per_second"]["prompt_prefill"],
        "generation_tps": result["rates_tokens_per_second"]["generation"],
    }, sort_keys=True))
    return 0


def main() -> int:
    program = Path(sys.argv[0]).name
    argv = sys.argv[1:]

    if argv and argv[0] in {"model", "benchmark"}:
        mode = argv[0]
        argv = argv[1:]
    elif program in {"model", "llama-model-manager"}:
        mode = "model"
    elif program in {"benchmark", "llama-benchmark"}:
        mode = "benchmark"
    else:
        print(
            "Usage: llama_models.py {model|benchmark} [options]\n"
            "   or: model [options]",
            file=sys.stderr,
        )
        return 2

    try:
        if mode == "model":
            return model_main(argv)
        return benchmark_main(argv)
    except LauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1



# ---------------------------------------------------------------------------
# --download implementation
# ---------------------------------------------------------------------------

def _infer_quant_dir(model_name: str) -> str:
    upper = model_name.upper()

    known = (
        "NVFP4",
        "MXFP4",
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ4_XS",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    )

    for quant in known:
        if quant in upper:
            return quant

    return "unknown"


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{value} B"


class _DownloadUI:
    """Small terminal status UI matching the lab-runner's event/progress style."""

    SYMBOLS = {
        "active": "→",
        "success": "✓",
        "warning": "!",
        "failure": "✗",
    }

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._live = False
        self._last_plain_percent = -10

    def elapsed(self) -> str:
        seconds = max(0, int(time.monotonic() - self.started))
        if seconds >= 3600:
            return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _clear_live(self) -> None:
        if self.is_tty and self._live:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
            self._live = False

    def event(self, kind: str, message: str, detail: str | None = None) -> None:
        self._clear_live()
        symbol = self.SYMBOLS.get(kind, self.SYMBOLS["active"])
        suffix = f"  {detail}" if detail else ""
        print(f"{self.elapsed()}  {symbol} {message}{suffix}", flush=True)

    def progress(self, label: str, done: int, total: int, rate_bps: float = 0.0) -> None:
        percent = (100.0 * done / total) if total else 0.0
        rate = f"  {_format_bytes(int(rate_bps))}/s" if rate_bps > 0 else ""
        line = (
            f"{self.elapsed()}  → {label}  "
            f"{_format_bytes(done)} / {_format_bytes(total)}  {percent:5.1f}%{rate}"
        )
        if self.is_tty:
            sys.stdout.write("\r\033[2K" + line)
            sys.stdout.flush()
            self._live = True
            return

        whole_percent = int(percent)
        if whole_percent >= self._last_plain_percent + 10 or done >= total:
            print(line, flush=True)
            self._last_plain_percent = whole_percent

    def finish_progress(self) -> None:
        self._clear_live()
        self._last_plain_percent = -10


def _copy2_with_progress(source: Path, target: Path, ui: _DownloadUI, label: str) -> None:
    """Keep shutil.copy2's fast-copy path while reporting destination growth."""
    total = source.stat().st_size
    started = time.monotonic()
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.5):
            try:
                done = min(target.stat().st_size, total)
            except FileNotFoundError:
                done = 0
            elapsed = max(time.monotonic() - started, 0.001)
            ui.progress(label, done, total, done / elapsed)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        shutil.copy2(source, target)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    elapsed = max(time.monotonic() - started, 0.001)
    ui.progress(label, total, total, total / elapsed)
    ui.finish_progress()


def download_model(
    repo: str,
    *,
    file_pattern: str = "*.gguf",
    quant: str | None = None,
    name: str | None = None,
) -> int:
    """Download GGUF model files locally via the Hugging Face CLI."""
    hf_bin = shutil.which("hf")
    if not hf_bin:
        print(
            "error: Hugging Face CLI 'hf' is not installed or not in PATH",
            file=sys.stderr,
        )
        return 1

    if name:
        print(
            "warning: --name is ignored with --download; test the model and "
            "use --promote separately",
            file=sys.stderr,
        )

    ui = _DownloadUI()
    requested_quant = quant or "auto"
    ui.event("active", "Preparing Hugging Face download", f"quant={requested_quant}")
    print(f"       repo:  {repo}", flush=True)
    print(f"       files: {file_pattern}", flush=True)

    cmd = [hf_bin, "download", repo]

    if any(ch in file_pattern for ch in "*?["):
        cmd.extend(["--include", file_pattern])
    else:
        cmd.append(file_pattern)

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        cmd.extend(["--token", token])

    ui.event("active", "Running hf download", "native Hugging Face progress follows")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )

    if result.returncode != 0:
        ui.event("failure", "Hugging Face download failed", f"exit {result.returncode}")
        return 1

    output_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        ui.event("failure", "Hugging Face did not report a downloaded path")
        return 1

    path_line = next(
        (line for line in reversed(output_lines) if line.startswith("path: ")),
        output_lines[-1],
    )
    downloaded_path = Path(path_line.removeprefix("path: ").strip()).expanduser()
    ui.event("success", "Hugging Face download complete", str(downloaded_path))

    if downloaded_path.is_file():
        gguf_files = [downloaded_path]
    elif downloaded_path.is_dir():
        gguf_files = sorted(downloaded_path.rglob("*.gguf"))
    else:
        ui.event("failure", "Downloaded path does not exist", str(downloaded_path))
        return 1

    if not gguf_files:
        ui.event("failure", "No GGUF files found", str(downloaded_path))
        return 1

    total_bytes = sum(path.stat().st_size for path in gguf_files)
    ui.event(
        "success",
        f"Resolved {len(gguf_files)} GGUF file(s)",
        f"{_format_bytes(total_bytes)} total",
    )
    for index, path in enumerate(gguf_files, start=1):
        print(
            f"       [{index}/{len(gguf_files)}] {path.name}  {_format_bytes(path.stat().st_size)}",
            flush=True,
        )

    model_basename = gguf_files[0].stem
    family_match = re.match(
        r"^(.+?)-\d{5}-of-\d{5}$",
        model_basename,
    )
    family_name = family_match.group(1) if family_match else model_basename
    quant_dir = quant or _infer_quant_dir(model_basename)

    target_dir = MODELS_DIR / family_name / quant_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    ui.event("active", "Staging model", str(target_dir))

    for index, source in enumerate(gguf_files, start=1):
        target = target_dir / source.name
        size = source.stat().st_size
        label = f"Staging {index}/{len(gguf_files)} {source.name}"

        if target.exists() and target.stat().st_size == size:
            ui.event(
                "success",
                f"Already staged {index}/{len(gguf_files)}",
                f"{source.name}  {_format_bytes(size)}",
            )
            continue

        ui.event(
            "active",
            f"Copying shard {index}/{len(gguf_files)}",
            f"{source.name}  {_format_bytes(size)}",
        )
        _copy2_with_progress(source, target, ui, label)
        ui.event(
            "success",
            f"Staged shard {index}/{len(gguf_files)}",
            source.name,
        )

    meta = {
        "repo_id": repo,
        "downloaded": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": [f.name for f in gguf_files],
        "quantization": quant_dir,
        "family": family_name,
    }

    meta_path = target_dir / ".meta.json"
    ui.event("active", "Writing model metadata", str(meta_path))
    meta_path.write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    ui.event(
        "success",
        "Model ready",
        f"{len(gguf_files)} file(s), {_format_bytes(total_bytes)} · {target_dir}",
    )
    return 0


# ---------------------------------------------------------------------------
# --promote implementation
# ---------------------------------------------------------------------------

def _build_promote_args(argv):
    p = argparse.ArgumentParser(description="Promote a lab experiment result to a daily preset")
    p.add_argument("lab_path", nargs="?", default=None, help="Path to the lab experiment run directory")
    p.add_argument("--name", default=None, help="Preset name (default: inferred from experiment)")
    args, remaining = p.parse_known_args(argv)
    if not args.lab_path:
        p.error("lab path is required")
    return args


def _map_flag_to_yaml_key(flag):
    direct = {
        "model": "model",
        "ctx-size": "ctx-size",
        "n-gpu-layers": "n-gpu-layers",
        "device": "device",
        "split-mode": "split-mode",
        "tensor-split": "tensor-split",
        "threads": "threads",
        "threads-batch": "threads-batch",
        "n-cpu-moe": "n-cpu-moe",
        "moe-expert-cache-size": "moe-expert-cache-size",
        "moe-expert-cache-host-pinned-mb": "moe-expert-cache-host-pinned-mb",
        "batch-size": "batch-size",
        "ubatch-size": "ubatch-size",
        "cache-type-k": "cache-type-k",
        "cache-type-v": "cache-type-v",
        "server-bin": "server-bin",
        "fit": "fit",
        "load-mode": "load-mode",
        "repack": "repack",
        "jinja": "jinja",
        "metrics": "metrics",
        "slot-save-path": "slot-save-path",
        "phase-aware-workspace": "phase-aware-workspace",
        "live-context-workspace": "live-context-workspace",
        "backend-sampling": "backend-sampling",
        "decode-overlap": "decode-overlap",
        "decode-boundary-overlap": "decode-boundary-overlap",
        "ple-prefetch": "ple-prefetch",
        "experimental-logs": "experimental-logs",
        "parallel": "parallel",
    }
    return direct.get(flag, flag)


def _extract_flags_from_run(run_data):
    flags = {}
    candidates = run_data.get("candidates", [])
    if candidates:
        cmd_array = candidates[0].get("command", [])
        if isinstance(cmd_array, list):
            i = 0
            while i < len(cmd_array):
                item = cmd_array[i]
                if isinstance(item, str) and item.startswith("--"):
                    if "=" in item:
                        key, _, value = item[2:].partition("=")
                        if value:
                            flags[key] = value
                        else:
                            flags[key] = "true"
                    else:
                        key = item[2:]
                        if i + 1 < len(cmd_array) and not cmd_array[i + 1].startswith("--"):
                            flags[key] = cmd_array[i + 1]
                            i += 1
                        else:
                            flags[key] = "true"
                i += 1
            if flags:
                return flags

    command_str = run_data.get("command_line", run_data.get("command", ""))
    if isinstance(command_str, str) and command_str:
        tokens = shlex.split(command_str)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("--"):
                if "=" in token:
                    key, _, value = token[2:].partition("=")
                    flags[key] = value or None
                else:
                    key = token[2:]
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        flags[key] = tokens[i + 1]
                        i += 1
                    else:
                        flags[key] = "true"
            i += 1
        if flags:
            return flags

    return flags


def promote_model(argv, after_download=False, name=None):
    args = _build_promote_args(argv)
    lab_path = Path(args.lab_path)

    if not lab_path.is_dir():
        print(f"error: lab path does not exist: {lab_path}", file=sys.stderr)
        return 1

    run_json_path = lab_path / "run.json"
    if not run_json_path.is_file():
        runs_dir = lab_path
        if lab_path.name.startswith("20"):
            runs_dir = lab_path / "runs"
            if runs_dir.is_dir():
                run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
                if run_dirs:
                    run_json_path = run_dirs[-1] / "run.json"
                    if run_json_path.is_file():
                        print(f"Using latest run: {run_dirs[-1].name}")
                    else:
                        print(f"error: no run.json in latest run {run_dirs[-1]}", file=sys.stderr)
                        return 1
                else:
                    print(f"error: no run directories in {runs_dir}", file=sys.stderr)
                    return 1
            else:
                print(f"error: no runs/ directory in {lab_path}", file=sys.stderr)
                return 1
        else:
            print(f"error: run.json not found at {run_json_path}", file=sys.stderr)
            return 1

    try:
        run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: cannot read run.json: {exc}", file=sys.stderr)
        return 1

    flags = _extract_flags_from_run(run_data)
    print(f"Extracted {len(flags)} flags from run.json")

    preset_dict = {}
    for key, value in flags.items():
        yaml_key = _map_flag_to_yaml_key(key)
        if yaml_key:
            preset_dict[yaml_key] = str(value)

    if "model" not in preset_dict:
        print("error: no model path found in run.json", file=sys.stderr)
        return 1

    preset_name = args.name
    if not preset_name:
        model_path = Path(preset_dict["model"])
        if model_path.is_file():
            family = model_path.parent.parent.name
            family_safe = family.replace("-", "").replace(" ", "").lower()
            quant_dir = model_path.parent.name
            preset_name = f"{family_safe}_{quant_dir}"
        else:
            preset_name = lab_path.name.replace("-", "_")

    print(f"\nPreset name: {preset_name}")
    print(f"\nMerged configuration ({len(preset_dict)} keys):")
    for key, value in sorted(preset_dict.items()):
        print(f"  {key}: {value}")

    presets_yaml_path = YAML_PATH
    if not presets_yaml_path.is_file():
        print("error: presets.yaml not found", file=sys.stderr)
        return 1

    content = presets_yaml_path.read_text(encoding="utf-8")
    preset_header = f"# -- {preset_name}"

    if preset_header in content:
        print(f"\nReplacing existing preset: {preset_name}")
        lines = content.splitlines()
        new_lines = []
        in_preset = False
        for line in lines:
            if line.startswith(preset_header):
                in_preset = True
                continue
            if in_preset:
                if line.strip() and not line.startswith(" "):
                    in_preset = False
                else:
                    continue
            new_lines.append(line)

        while new_lines and new_lines[-1].strip() == "":
            new_lines.pop()

        new_lines.append("")
        new_lines.append(preset_header)
        new_lines.append(preset_name + ":")
        new_lines.append(f"# Promoted on {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for key, value in sorted(preset_dict.items()):
            if " " in value or value in ("true", "false", "on", "off", "yes", "no"):
                value = shlex.quote(value)
            new_lines.append(f"  {key}: {value}")

        content = "\n".join(new_lines) + "\n"
    else:
        print(f"\nAdding new preset: {preset_name}")
        lines = content.rstrip().splitlines()
        lines.append("")
        lines.append(preset_header)
        lines.append(preset_name + ":")
        lines.append(f"# Promoted on {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for key, value in sorted(preset_dict.items()):
            if " " in value or value in ("true", "false", "on", "off", "yes", "no"):
                value = shlex.quote(value)
            lines.append(f"  {key}: {value}")
        content = "\n".join(lines) + "\n"

    # Always backup before writing
    backup_path = presets_yaml_path.with_suffix(".yaml.bak")
    import shutil
    shutil.copy2(str(presets_yaml_path), str(backup_path))

    presets_yaml_path.write_text(content, encoding="utf-8")
    print(f"\nWrote preset '{preset_name}' to {presets_yaml_path}")
    print(f"  backup saved to {backup_path}")

    if not after_download:
        # Inline verification: check preset exists in the written file
        with open(presets_yaml_path, encoding="utf-8") as f:
            written = f.read()
        if f"{preset_name}:" in written:
            print(f"✓ Verified: '{preset_name}' written to presets.yaml")
        else:
            print(f"✗ Error: '{preset_name}' NOT found in presets.yaml after write")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())