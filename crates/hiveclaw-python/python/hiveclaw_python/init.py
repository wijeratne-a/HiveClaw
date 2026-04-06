"""One-shot setup: locate repo root, bootstrap ``pheromoned``, return a :class:`HiveClawManager`."""

from __future__ import annotations

import sys
from pathlib import Path

from .manager import HiveClawManager

_MARKERS = ("com.hiveclaw.pheromoned.plist.in", "Makefile")


def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from ``start`` (default: this package directory) for a directory
    containing ``Makefile`` and ``com.hiveclaw.pheromoned.plist.in``.
    """
    here = (start or Path(__file__).resolve().parent).resolve()
    for d in [here, *here.parents]:
        ok = all((d / m).is_file() for m in _MARKERS)
        if ok:
            return d
    raise RuntimeError(
        "Cannot auto-detect HiveClaw repo root (need Makefile + "
        "com.hiveclaw.pheromoned.plist.in). Pass repo_root= to init() explicitly."
    )


def init(
    repo_root: Path | str | None = None,
    *,
    python_exe: str | None = None,
    build_if_missing: bool = False,
    bootstrap_daemon: bool = True,
) -> HiveClawManager:
    """
    Create a :class:`HiveClawManager` and optionally bootstrap the LaunchAgent.

    - ``repo_root``: checkout root; default auto-detect from package location.
    - ``build_if_missing``: run ``cargo build --release -p hiveclaw-daemon`` if needed.
    - ``bootstrap_daemon``: install plist + ``launchctl bootstrap`` when not already
      running this binary.
    """
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    mgr = HiveClawManager(root, python_exe=python_exe or sys.executable)
    if bootstrap_daemon:
        mgr.bootstrap(build_if_missing=build_if_missing, skip_if_running=True)
    return mgr
