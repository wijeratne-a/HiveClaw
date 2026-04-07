"""One-shot setup: locate repo root, bootstrap ``pheromoned``, return a :class:`HiveClawManager`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .manager import HiveClawManager

_PLIST_REL = Path("crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in")


def resolve_manager_repo_root(repo_root: Path | str | None) -> Path:
    """
    Root used for ``make``/``scripts/`` paths and ``cargo`` builds.

    Order: explicit ``repo_root``, then :func:`find_repo_root`, then ``HIVECLAW_REPO_ROOT``,
    then the installed ``hiveclaw_python`` package directory (daemon may still work via a
    bundled ``pheromoned``; set ``HIVECLAW_REPO_ROOT`` for model paths when not in a checkout).
    """
    if repo_root is not None:
        return Path(repo_root).resolve()
    try:
        return find_repo_root()
    except RuntimeError:
        env = os.environ.get("HIVECLAW_REPO_ROOT", "").strip()
        if env:
            p = Path(env).resolve()
            if (p / "Makefile").is_file():
                return p
        return Path(__file__).resolve().parent


def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from ``start`` (default: this package directory) for a directory
    containing ``Makefile`` and ``crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in``.
    """
    here = (start or Path(__file__).resolve().parent).resolve()
    for d in [here, *here.parents]:
        if (d / "Makefile").is_file() and (d / _PLIST_REL).is_file():
            return d
    raise RuntimeError(
        "Cannot auto-detect HiveClaw repo root (need Makefile + "
        "crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in). "
        "Pass repo_root= to init() explicitly."
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

    - ``repo_root``: checkout root; default :func:`find_repo_root`, else ``HIVECLAW_REPO_ROOT``,
      else the ``hiveclaw_python`` install directory (bundled ``pheromoned`` only; set
      ``HIVECLAW_REPO_ROOT`` when resolving repo-relative model paths — see :func:`resolve_manager_repo_root`).
    - ``build_if_missing``: run ``cargo build --release -p hiveclaw-daemon`` if needed.
    - ``bootstrap_daemon``: install plist + ``launchctl bootstrap`` when not already
      running this binary.
    """
    root = resolve_manager_repo_root(repo_root)
    mgr = HiveClawManager(root, python_exe=python_exe or sys.executable)
    if bootstrap_daemon:
        mgr.bootstrap(build_if_missing=build_if_missing, skip_if_running=True)
    return mgr
