"""Resolve ``pheromoned`` binary: dev checkout vs wheel-bundled (macOS arm64)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent


def is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )


def bundled_pheromoned_path() -> Path | None:
    """
    Path to a wheel-shipped ``pheromoned`` under ``native/macos_arm64/``, if present.
    CI copies the release binary here before ``maturin build``; git ignores the file.

    ``_PKG_ROOT`` is this module's package directory (``site-packages/hiveclaw_python`` when
    installed from a wheel), so the binary is discovered without a separate repo root.
    """
    if not is_macos_arm64():
        return None
    p = _PKG_ROOT / "native" / "macos_arm64" / "pheromoned"
    if p.is_file():
        return p
    return None


def package_plist_template_path() -> Path:
    return _PKG_ROOT / "data" / "com.hiveclaw.pheromoned.plist.in"


def render_plist_program(program: Path | str) -> str:
    """Same layout as ``crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in``."""
    body = package_plist_template_path().read_text(encoding="utf-8")
    return body.replace("@PROGRAM@", str(program))
