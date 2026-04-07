"""Resolve ``pheromoned`` binary: dev checkout vs wheel-bundled (macOS arm64)."""

from __future__ import annotations

import platform
import sys
from importlib import resources
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent

_PLIST_NAME = "com.hiveclaw.pheromoned.plist.in"


def _read_plist_template_text() -> str:
    """Load LaunchAgent template from package data (wheel-safe) or filesystem fallback."""
    try:
        tr = resources.files("hiveclaw_python").joinpath("data", _PLIST_NAME)
        return tr.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError, TypeError, ValueError, ModuleNotFoundError):
        pass
    legacy = _PKG_ROOT / "data" / _PLIST_NAME
    if legacy.is_file():
        return legacy.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Missing plist template: hiveclaw_python.data/{_PLIST_NAME} or {legacy}"
    )


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
    """Filesystem path to bundled template when present (e.g. editable install).

    Prefer :func:`render_plist_program`, which uses :mod:`importlib.resources` so the
    template loads from wheels without relying on a checkout layout.
    """
    return _PKG_ROOT / "data" / _PLIST_NAME


def render_plist_program(program: Path | str) -> str:
    """Render plist body; kept in sync with ``crates/hiveclaw-daemon/data/...plist.in`` (``make check-plist``)."""
    body = _read_plist_template_text()
    return body.replace("@PROGRAM@", str(program))
