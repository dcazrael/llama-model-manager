"""Readable foreground rendering for llama-server logs.

The raw server stream is always written to disk. This module only controls the
human-facing terminal view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import threading

from .ui import TerminalUI


TASK_RE = re.compile(r"\btask\s+(-?\d+)\b")
NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

PROGRESS_RE = re.compile(
    rf"\bn_gen\s*=\s*(\d+),\s*tg\s*=\s*({NUM_RE})\s*t/s,\s*tg_3s\s*=\s*({NUM_RE})\s*t/s"
)
PROMPT_TIMING_RE = re.compile(
    rf"prompt eval time\s*=\s*({NUM_RE})\s*ms\s*/\s*(\d+)\s*tokens.*?({NUM_RE})\s*tokens per second"
)
EVAL_TIMING_RE = re.compile(
    rf"(?<!prompt )\beval time\s*=\s*({NUM_RE})\s*ms\s*/\s*(\d+)\s*tokens.*?({NUM_RE})\s*tokens per second"
)
TOTAL_TIMING_RE = re.compile(
    rf"total time\s*=\s*({NUM_RE})\s*ms\s*/\s*(\d+)\s*tokens"
)
GRAPHS_RE = re.compile(r"graphs reused\s*=\s*(\d+)")
DRAFT_RE = re.compile(
    rf"draft acceptance\s*=\s*({NUM_RE})\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\),\s*mean len\s*=\s*({NUM_RE})"
)
RELEASE_RE = re.compile(
    r"\brelease:.*?\btask\s+(-?\d+)\b.*?n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)"
)
PROCESS_RE = re.compile(r"\btask\s+(-?\d+)\s*\|\s*processing task\b")
MODEL_RE = re.compile(r"loading model '([^']+)'")
LISTEN_RE = re.compile(r"listening on (https?://\S+)")
KV_RE = re.compile(r"([A-Za-z0-9_-]+)=([^\s]+)")


def _kv_fields(text: str) -> dict[str, str]:
    return dict(KV_RE.findall(text))


def _int(fields: dict[str, str], key: str) -> int | None:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        return None


def _human_int(value: int) -> str:
    return f"{value:,}"


def _seconds(ms: float | None) -> str:
    if ms is None:
        return "?"
    return f"{ms / 1000:.2f} s"


def _gib(value: int | None) -> str:
    if value is None:
        return "?"
    return f"{value / (1024 ** 3):.2f} GiB"


@dataclass
class RequestStats:
    task: int
    prompt_tokens: int | None = None
    prompt_ms: float | None = None
    prompt_tps: float | None = None
    generated_tokens: int | None = None
    generation_ms: float | None = None
    generation_tps: float | None = None
    total_tokens: int | None = None
    total_ms: float | None = None
    graphs_reused: int | None = None
    draft_acceptance: float | None = None
    draft_accepted: int | None = None
    draft_generated: int | None = None
    draft_mean_len: float | None = None
    released_tokens: int | None = None
    truncated: bool = False
    moe: dict[str, int | float] = field(default_factory=dict)
    paths: dict[str, int | float] = field(default_factory=dict)
    resources: dict[str, int | float] = field(default_factory=dict)


class LlamaServerLogFormatter:
    """Turn verbose llama-server diagnostics into compact terminal events."""

    def __init__(self, ui: TerminalUI, *, raw_output: bool = False) -> None:
        self.ui = ui
        self.raw_output = raw_output
        self.requests: dict[int, RequestStats] = {}
        self.current_task: int | None = None
        self._seen_prepack_reasons: set[str] = set()
        self._model_loaded_seen = False
        self._listening_seen = False

    def server_ready(self, preset: str, url: str, log_path: Path) -> None:
        self.ui.clear_live()
        self.ui.event("success", f"Active preset: {preset}", url)
        self.ui.line(f"  {self.ui.styled('Log'.ljust(12), 'dim')}  {log_path}")

    def formatter_failed(self, exc: Exception) -> None:
        if self.raw_output:
            return
        self.ui.clear_live()
        self.ui.event(
            "warning",
            "Log formatter failed; switching to raw output",
            f"{type(exc).__name__}: {exc}",
        )
        self.raw_output = True

    def _stats(self, task: int | None = None) -> RequestStats | None:
        task = self.current_task if task is None else task
        if task is None or task < 0:
            return None
        if task not in self.requests:
            self.requests[task] = RequestStats(task=task)
        return self.requests[task]

    def _warning_or_error(self, text: str) -> bool:
        lower = text.lower()
        severe = (
            re.search(r"^\S+\s+E\s+", text) is not None
            or "cuda error" in lower
            or "out of memory" in lower
            or "fatal error" in lower
            or "assertion" in lower
            or "segmentation fault" in lower
            or "traceback" in lower
            or re.search(r"(?:^|\s)e\s+\w+", text) is not None
        )
        failed = " failed" in lower or "failure" in lower
        warning = re.search(r"(?:^|\s)W\s+\w+", text) is not None or "warning" in lower

        if severe or failed:
            message = text if len(text) <= 220 else text[:217] + "..."
            self.ui.event("failure", message)
            return True
        if warning:
            message = text if len(text) <= 220 else text[:217] + "..."
            self.ui.event("warning", message)
            return True
        return False

    def _emit_request_summary(self, stats: RequestStats) -> None:
        self.ui.clear_live()
        rows: list[tuple[str, str] | tuple[str, str, str]] = []

        if stats.prompt_tokens is not None:
            value = (
                f"{_human_int(stats.prompt_tokens)} tok · {_seconds(stats.prompt_ms)}"
                + (f" · {stats.prompt_tps:.2f} t/s" if stats.prompt_tps is not None else "")
            )
            rows.append(("Prompt", value, "bright_green"))

        if stats.generated_tokens is not None:
            value = (
                f"{_human_int(stats.generated_tokens)} tok · {_seconds(stats.generation_ms)}"
                + (f" · {stats.generation_tps:.2f} t/s" if stats.generation_tps is not None else "")
            )
            rows.append(("Generate", value, "bright_green"))

        if stats.total_tokens is not None:
            rows.append((
                "Total",
                f"{_human_int(stats.total_tokens)} tok · {_seconds(stats.total_ms)}",
            ))

        if stats.draft_acceptance is not None:
            accepted = (
                f"{_human_int(stats.draft_accepted)}/{_human_int(stats.draft_generated)}"
                if stats.draft_accepted is not None and stats.draft_generated is not None
                else "?"
            )
            mean = f" · mean {stats.draft_mean_len:.2f}" if stats.draft_mean_len is not None else ""
            rows.append((
                "MTP",
                f"{stats.draft_acceptance * 100:.2f}% · {accepted}{mean}",
                "magenta",
            ))

        if stats.graphs_reused is not None:
            rows.append(("Graphs", f"{_human_int(stats.graphs_reused)} reused"))

        slots = int(stats.moe["populated_slots"]) if "populated_slots" in stats.moe else None
        capacity = int(stats.moe["slot_capacity"]) if "slot_capacity" in stats.moe else None
        payload = int(stats.moe["populated_payload_bytes"]) if "populated_payload_bytes" in stats.moe else None
        payload_capacity = int(stats.moe["payload_capacity_bytes"]) if "payload_capacity_bytes" in stats.moe else None
        if slots is not None and capacity:
            pct = slots / capacity * 100.0
            payload_text = (
                f" · {_gib(payload)}/{_gib(payload_capacity)}"
                if payload is not None and payload_capacity is not None
                else ""
            )
            rows.append(("MoE cache", f"{_human_int(slots)}/{_human_int(capacity)} slots · {pct:.1f}%{payload_text}"))

        hits = int(stats.paths["cache_hits"]) if "cache_hits" in stats.paths else None
        misses = int(stats.paths["cache_misses"]) if "cache_misses" in stats.paths else None
        if hits is not None and misses is not None:
            total = hits + misses
            hit_rate = hits / total * 100.0 if total else 0.0
            rows.append((
                "MoE reuse",
                f"cumulative · {_human_int(hits)} hits · {_human_int(misses)} misses · {hit_rate:.1f}%",
            ))

        health_keys = ("fallback", "rollback", "prepare_error", "finish_error", "required_unsupported")
        health = {key: int(stats.moe.get(key, 0)) for key in health_keys}
        if any(health.values()):
            rows.append((
                "MoE health",
                (
                    f"{health['fallback']} fallback · {health['rollback']} rollback · "
                    f"{health['prepare_error'] + health['finish_error']} errors · "
                    f"{health['required_unsupported']} unsupported"
                ),
                "red",
            ))

        if stats.truncated:
            rows.append(("Context", "truncated", "yellow"))

        if rows:
            self.ui.line()
            self.ui.rows(f"REQUEST {stats.task}", rows)
        else:
            detail = (
                f"{_human_int(stats.released_tokens)} tokens"
                if stats.released_tokens is not None
                else None
            )
            self.ui.event("success", f"Request {stats.task} complete", detail)

    def handle(self, line: str) -> None:
        if self.raw_output:
            self.ui.raw(line)
            return

        text = line.strip()
        if not text:
            return

        process = PROCESS_RE.search(text)
        if process:
            task = int(process.group(1))
            if task >= 0:
                self.current_task = task
                self.requests[task] = RequestStats(task=task)
                self.ui.event("active", f"Request {task}", "processing")
            return

        progress = PROGRESS_RE.search(text)
        if progress:
            task_match = TASK_RE.search(text)
            task = int(task_match.group(1)) if task_match else self.current_task
            stats = self._stats(task)
            if stats is not None:
                stats.generated_tokens = int(progress.group(1))
                stats.generation_tps = float(progress.group(2))
                self.ui.live(
                    f"Request {stats.task} · {_human_int(stats.generated_tokens)} generated"
                    f" · {stats.generation_tps:.2f} t/s · recent {float(progress.group(3)):.2f} t/s"
                )
            return

        match = PROMPT_TIMING_RE.search(text)
        if match:
            stats = self._stats()
            if stats:
                stats.prompt_ms = float(match.group(1))
                stats.prompt_tokens = int(match.group(2))
                stats.prompt_tps = float(match.group(3))
            return

        match = EVAL_TIMING_RE.search(text)
        if match:
            stats = self._stats()
            if stats:
                stats.generation_ms = float(match.group(1))
                stats.generated_tokens = int(match.group(2))
                stats.generation_tps = float(match.group(3))
            return

        match = TOTAL_TIMING_RE.search(text)
        if match:
            stats = self._stats()
            if stats:
                stats.total_ms = float(match.group(1))
                stats.total_tokens = int(match.group(2))
            return

        match = GRAPHS_RE.search(text)
        if match:
            stats = self._stats()
            if stats:
                stats.graphs_reused = int(match.group(1))
            return

        match = DRAFT_RE.search(text)
        if match:
            stats = self._stats()
            if stats:
                stats.draft_acceptance = float(match.group(1))
                stats.draft_accepted = int(match.group(2))
                stats.draft_generated = int(match.group(3))
                stats.draft_mean_len = float(match.group(4))
            return

        if text.startswith("moe-grouped-decode:"):
            stats = self._stats()
            if stats:
                fields = _kv_fields(text)
                for key in (
                    "fallback", "rollback", "prepare_error", "finish_error",
                    "required_unsupported", "populated_slots", "slot_capacity",
                    "populated_payload_bytes", "payload_capacity_bytes",
                ):
                    value = _int(fields, key)
                    if value is not None:
                        stats.moe[key] = value
            return

        if text.startswith("moe-grouped-paths:"):
            stats = self._stats()
            if stats:
                fields = _kv_fields(text)
                for key in ("cache_hits", "cache_misses", "captures", "replays"):
                    value = _int(fields, key)
                    if value is not None:
                        stats.paths[key] = value
            return

        if text.startswith("moe-grouped-resources:"):
            stats = self._stats()
            if stats:
                fields = _kv_fields(text)
                for key in ("payload_capacity_bytes", "payload_allocation_bytes", "host_staging_bytes"):
                    value = _int(fields, key)
                    if value is not None:
                        stats.resources[key] = value
            return

        if text.startswith("moe-prepack: disabled reason="):
            reason = text.partition("reason=")[2].strip()
            if reason and reason not in self._seen_prepack_reasons:
                self._seen_prepack_reasons.add(reason)
                if reason == "requires single-row independent or bounded speculative MAIN decode":
                    self.ui.event("info", "MoE prepack", "disabled for current speculative decode path")
                else:
                    self.ui.event("warning", "MoE prepack disabled", reason)
            return

        if text.startswith((
            "moe-early-router:",
            "moe-grouped-prefetch:",
            "moe-cache-experts:",
            "moe-cache-experts-hot:",
            "moe-cache-tensor-hot-decode-miss:",
            "moe-cache-phase:",
            "moe-cache-mm:",
            "moe-device-prefetch:",
        )):
            return

        release = RELEASE_RE.search(text)
        if release:
            task = int(release.group(1))
            stats = self._stats(task)
            if stats:
                stats.released_tokens = int(release.group(2))
                stats.truncated = bool(int(release.group(3)))
                self._emit_request_summary(stats)
                self.requests.pop(task, None)
            if self.current_task == task:
                self.current_task = None
            return

        model = MODEL_RE.search(text)
        if model:
            self.ui.event("active", "Loading model", Path(model.group(1)).name)
            return

        if "model loaded" in text.lower() and not self._model_loaded_seen:
            self._model_loaded_seen = True
            self.ui.event("success", "Model loaded")
            return

        listening = LISTEN_RE.search(text)
        if listening and not self._listening_seen:
            self._listening_seen = True
            self.ui.event("success", "llama-server listening", listening.group(1))
            return

        if "MTP recurrent-plane policy:" in text:
            detail = text.split("MTP recurrent-plane policy:", 1)[1].strip()
            self.ui.event("special", "MTP enabled", detail)
            return

        if "decode overlap:" in text:
            detail = text.split("decode overlap:", 1)[1].strip()
            if "unsupported" in detail.lower() or "normal decode" in detail.lower():
                self.ui.event("warning", "Decode overlap", detail)
            return

        if self._warning_or_error(text):
            return

        # Routine INFO/DEBUG lines stay in the raw log. The launcher terminal is
        # deliberately an operational summary, not a second copy of server.log.


class ServerLogPump:
    """Drain llama-server stdout without blocking startup and tee it to disk."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        log_path: Path,
        formatter: LlamaServerLogFormatter,
    ) -> None:
        self.process = process
        self.log_path = log_path
        self.formatter = formatter
        self._thread = threading.Thread(
            target=self._run,
            name="llama-server-log-pump",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        assert self.process.stdout is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as handle:
            for line in self.process.stdout:
                handle.write(line)
                handle.flush()
                try:
                    self.formatter.handle(line)
                except Exception as exc:
                    # Never let presentation code stop draining llama-server's
                    # pipe. A blocked stdout pipe can stall the server itself.
                    self.formatter.formatter_failed(exc)
