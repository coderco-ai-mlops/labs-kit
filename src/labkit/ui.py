"""Terminal output in the brand colours. Plain text when colour is not supported."""

import os
import sys

CORAL = "\x1b[38;2;235;107;71m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"


def _colour_on() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("CI"):
        return False
    return sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _colour_on() else text


def coral(text: str) -> str:
    return paint(text, CORAL)


def bold(text: str) -> str:
    return paint(text, BOLD)


def dim(text: str) -> str:
    return paint(text, DIM)


def setup_output() -> None:
    # Windows consoles default to a legacy code page; this keeps printing from crashing on any character.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def heading(text: str) -> None:
    print()
    print(coral(bold(text)))
