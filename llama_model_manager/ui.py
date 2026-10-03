"""Dependency-free terminal rendering for the model launcher."""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import TextIO


class TerminalUI:
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "bright_green": "\033[1;92m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "red": "\033[31m",
    }

    SYMBOLS = {
        "active": "●",
        "success": "✓",
        "special": "⚡",
        "warning": "!",
        "failure": "✗",
        "divider": "─",
    }

    ASCII_SYMBOLS = {
        "active": "*",
        "success": "+",
        "special": "*",
        "warning": "!",
        "failure": "x",
        "divider": "-",
    }

    def __init__(
        self,
        stream: TextIO = sys.stdout,
        *,
        no_color: bool = False,
        plain: bool = False,
        raw_output: bool = False,
    ) -> None:
        self.stream = stream
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self.color = self.is_tty and not no_color and "NO_COLOR" not in os.environ
        self.interactive = self.is_tty and not plain and not raw_output
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            "✓⚡─".encode(encoding)
            self.symbols = self.SYMBOLS
        except (UnicodeEncodeError, LookupError):
            self.symbols = self.ASCII_SYMBOLS
        self.width = max(32, shutil.get_terminal_size((88, 24)).columns)
        self.started = time.monotonic()
        self._lock = threading.RLock()
        self._live_text = ""

    def styled(self, text: str, style: str) -> str:
        if not self.color:
            return text
        return f"{self.COLORS[style]}{text}{self.COLORS['reset']}"

    def _clear_live_locked(self) -> None:
        if self.interactive and self._live_text:
            self.stream.write("\r\033[2K")
            self._live_text = ""

    def line(self, text: str = "") -> None:
        with self._lock:
            self._clear_live_locked()
            self.stream.write(text + "\n")
            self.stream.flush()

    def raw(self, text: str) -> None:
        with self._lock:
            self._clear_live_locked()
            self.stream.write(text)
            if text and not text.endswith("\n"):
                self.stream.write("\n")
            self.stream.flush()

    def elapsed(self) -> str:
        seconds = max(0, int(time.monotonic() - self.started))
        if seconds >= 3600:
            return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def event(self, kind: str, message: str, detail: str | None = None) -> None:
        color = {
            "active": "cyan",
            "success": "green",
            "special": "magenta",
            "warning": "yellow",
            "failure": "red",
        }.get(kind, "reset")
        stamp = self.styled(self.elapsed(), "dim")
        symbol = self.styled(self.symbols.get(kind, self.symbols["active"]), color)
        main = self.styled(message, color) if kind != "info" else message
        suffix = f"  {self.styled(detail, 'dim')}" if detail else ""
        self.line(f"{stamp}  {symbol} {main}{suffix}")

    def live(self, text: str, *, style: str = "cyan") -> None:
        if not self.interactive:
            return
        with self._lock:
            rendered = f"{self.styled('●', style)} {text}"
            visible = f"● {text}"
            if len(visible) >= self.width:
                keep = max(1, self.width - 2)
                text = text[:keep] + "…"
                rendered = f"{self.styled('●', style)} {text}"
            self.stream.write("\r\033[2K" + rendered)
            self.stream.flush()
            self._live_text = text

    def clear_live(self) -> None:
        with self._lock:
            self._clear_live_locked()
            if self.interactive:
                self.stream.flush()

    def rows(
        self,
        title: str,
        rows: list[tuple[str, str] | tuple[str, str, str]],
    ) -> None:
        if not rows:
            return
        self.line(self.styled(title, "yellow"))
        label_width = max(len(row[0]) for row in rows)
        for row in rows:
            label, value = row[0], row[1]
            style = row[2] if len(row) > 2 else "reset"
            rendered = self.styled(value, style) if style != "reset" else value
            self.line(f"  {self.styled(label.ljust(label_width), 'dim')}  {rendered}")

    def divider(self) -> None:
        self.line(self.styled(self.symbols["divider"] * min(72, self.width), "dim"))
