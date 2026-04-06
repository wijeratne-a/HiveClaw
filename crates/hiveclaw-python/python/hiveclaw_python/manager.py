"""Optional helpers to build/load the pheromoned daemon from a repo checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class HiveClawManager:
    """Run ``cargo`` / ``make`` daemon targets the same way as the Makefile + doctor."""

    def __init__(
        self,
        repo_root: Path | str,
        *,
        python_exe: str | None = None,
        teardown: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.python_exe = python_exe or sys.executable
        self.teardown = bool(teardown)
        self._pheromoned = self.repo_root / "target" / "release" / "pheromoned"
        self._doctor = self.repo_root / "scripts" / "hiveclaw_doctor.py"

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = self.python_exe
        return env

    def build_daemon_release(self) -> None:
        subprocess.run(
            ["cargo", "build", "--release", "-p", "hiveclaw-daemon"],
            cwd=str(self.repo_root),
            check=True,
        )

    def daemon_load(self) -> None:
        subprocess.run(
            ["make", "daemon-load", f"PYTHON={self.python_exe}"],
            cwd=str(self.repo_root),
            env=self._env(),
            check=True,
        )

    def daemon_unload(self) -> None:
        subprocess.run(
            ["make", "daemon-unload"],
            cwd=str(self.repo_root),
            env=self._env(),
            check=True,
        )

    def doctor(self) -> None:
        if not self._doctor.is_file():
            raise FileNotFoundError(self._doctor)
        subprocess.run(
            [self.python_exe, str(self._doctor), str(self.repo_root), str(self._pheromoned)],
            cwd=str(self.repo_root),
            check=True,
        )

    def __enter__(self) -> HiveClawManager:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.teardown:
            try:
                self.daemon_unload()
            except subprocess.CalledProcessError:
                pass
