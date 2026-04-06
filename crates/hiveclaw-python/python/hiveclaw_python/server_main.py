"""Console entrypoint ``hiveclaw-server``: prefer packaged :mod:`hiveclaw_python.openai_server`."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _scripts_dir_with_server() -> Path | None:
    env = os.environ.get("HIVECLAW_REPO_ROOT", "").strip()
    if env:
        p = Path(env).resolve() / "scripts"
        if (p / "hiveclaw_server.py").is_file():
            return p
    try:
        from .init import find_repo_root

        root = find_repo_root()
    except RuntimeError:
        return None
    p = root / "scripts"
    if (p / "hiveclaw_server.py").is_file():
        return p
    return None


def main() -> None:
    try:
        from .openai_server import main as run_impl

        run_impl()
        return
    except ImportError:
        pass
    d = _scripts_dir_with_server()
    if d is None:
        print(
            "hiveclaw-server: could not load hiveclaw_python.openai_server "
            "or find scripts/hiveclaw_server.py.\n"
            "Run `make python` from a HiveClaw checkout or set HIVECLAW_REPO_ROOT.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    sys.path.insert(0, str(d))
    runpy.run_path(str(d / "hiveclaw_server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
