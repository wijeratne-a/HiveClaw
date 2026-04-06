"""Console entrypoint ``hiveclaw-server``: run ``scripts/hiveclaw_server.py`` from a checkout."""

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
    d = _scripts_dir_with_server()
    if d is None:
        print(
            "hiveclaw-server: could not find scripts/hiveclaw_server.py.\n"
            "Clone HiveClaw and set HIVECLAW_REPO_ROOT to the repository root, "
            "or install the package from a tree where find_repo_root() succeeds.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    sys.path.insert(0, str(d))
    runpy.run_path(str(d / "hiveclaw_server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
