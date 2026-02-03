# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Compact formatter implementation using terminal output."""

import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.formatter import Formatter

LINE_LENGTH = 100

class CompactFormatter(Formatter):
    def __init__(self) -> None:
        self.color = sys.stdout.isatty()
        self._console = Console() if self.color else None
        self._stderr_console = Console(stderr=True) if self.color else None

    def format_message(self, message: dict[str, Any]) -> str:
        if DEFAULT_CONFIG.agent.verbose:
            content = message.get("content", "").replace(chr(10), chr(92) + "n")
            return f'[{message.get("role", "unknown")}]: {content}'[:LINE_LENGTH] + " ..."
        return ""

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        if DEFAULT_CONFIG.agent.verbose:
            return "\n".join(self.format_message(m) for m in messages)
        return ""

    def print_message(self, message: dict[str, Any]) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            print(self.format_message(message))

    def print_messages(self, messages: list[dict[str, Any]]) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            print(self.format_messages(messages))

    def print_status(self, message: str) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            if self._console:
                self._console.print(message, style="green")
            else:
                print(message)

    def print_error(self, message: str) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            if self._stderr_console:
                self._stderr_console.print(message, style="red")
            else:
                print(message, file=sys.stderr)

    def print_warning(self, message: str) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            if self._console:
                self._console.print(message, style="yellow")
            else:
                print(message)

    def print_label_and_value(self, label: str, value: str) -> None:
        if DEFAULT_CONFIG.agent.verbose:
            md = Markdown(f"__**{label}**__: {value}")
            if self._console:
                self._console.print(md)
            else:
                print(md)
