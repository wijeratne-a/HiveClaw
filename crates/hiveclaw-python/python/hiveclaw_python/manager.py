"""Optional helpers to build/load the pheromoned daemon from a repo checkout."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .daemon_paths import bundled_pheromoned_path, package_plist_template_path, render_plist_program


def _launchctl_print(svc: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["launchctl", "print", svc],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as e:
        return 1, str(e)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _parse_program(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"program\s*=\s*(.+?);?\s*$", line)
        if m:
            return m.group(1).strip().strip('"')
    return None


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
        self._pheromoned = self._resolve_pheromoned_binary()
        self._doctor = self.repo_root / "scripts" / "hiveclaw_doctor.py"
        self._plist_template = self.repo_root / "com.hiveclaw.pheromoned.plist.in"
        self._launch_agents = Path.home() / "Library" / "LaunchAgents"
        self._installed_plist = self._launch_agents / "com.hiveclaw.pheromoned.plist"

    def _resolve_pheromoned_binary(self) -> Path:
        dev = self.repo_root / "target" / "release" / "pheromoned"
        if dev.is_file():
            return dev
        bundled = bundled_pheromoned_path()
        if bundled is not None:
            return bundled
        return dev

    @property
    def gui_domain(self) -> str:
        return f"gui/{os.getuid()}"

    @property
    def launchd_service(self) -> str:
        return f"{self.gui_domain}/com.hiveclaw.pheromoned"

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = self.python_exe
        return env

    def build_daemon_release(self) -> None:
        if not (self.repo_root / "Cargo.toml").is_file():
            raise RuntimeError(
                "Cannot build pheromoned: no Cargo.toml at repo_root. "
                "Use a full HiveClaw checkout, pass repo_root= to HiveClawManager, "
                "or install a wheel that bundles native/macos_arm64/pheromoned."
            )
        subprocess.run(
            ["cargo", "build", "--release", "-p", "hiveclaw-daemon"],
            cwd=str(self.repo_root),
            check=True,
        )
        self._pheromoned = self._resolve_pheromoned_binary()

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

    def is_running(self) -> bool:
        rc, text = _launchctl_print(self.launchd_service)
        if rc != 0:
            return False
        return "state = running" in text

    def render_plist(self) -> str:
        if self._plist_template.is_file():
            template = self._plist_template.read_text(encoding="utf-8")
            return template.replace("@PROGRAM@", str(self._pheromoned))
        ptp = package_plist_template_path()
        if ptp.is_file():
            return render_plist_program(self._pheromoned)
        raise FileNotFoundError(
            f"Missing plist template: {self._plist_template} or {ptp}"
        )

    def bootstrap(
        self,
        *,
        build_if_missing: bool = True,
        skip_if_running: bool = True,
    ) -> None:
        """
        Build binary if needed, install LaunchAgent plist, ``launchctl bootstrap``.
        If ``skip_if_running`` and the service is already running with this binary, no-op.
        """
        if skip_if_running and self.is_running():
            rc, text = _launchctl_print(self.launchd_service)
            prog = _parse_program(text)
            if prog is not None and Path(prog).resolve() == self._pheromoned.resolve():
                return

        if build_if_missing and not self._pheromoned.is_file():
            try:
                self.build_daemon_release()
            except RuntimeError:
                self._pheromoned = self._resolve_pheromoned_binary()

        if not self._pheromoned.is_file():
            raise RuntimeError(
                f"Missing pheromoned binary: {self._pheromoned}. "
                "From a checkout: `cargo build --release -p hiveclaw-daemon`. "
                "For pip wheels on Apple Silicon, ensure the wheel includes "
                "hiveclaw_python/native/macos_arm64/pheromoned (see native/macos_arm64/README.txt)."
            )

        self._launch_agents.mkdir(parents=True, exist_ok=True)
        plist_body = self.render_plist()
        self._installed_plist.write_text(plist_body, encoding="utf-8")
        self._installed_plist.chmod(0o644)

        subprocess.run(
            ["launchctl", "bootout", self.gui_domain, "com.hiveclaw.pheromoned"],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["launchctl", "bootout", self.gui_domain, str(self._installed_plist)],
            capture_output=True,
            text=True,
        )
        time.sleep(0.5)

        r = subprocess.run(
            ["plutil", "-lint", str(self._installed_plist)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Invalid plist {self._installed_plist}: {r.stderr or r.stdout}"
            )

        br = subprocess.run(
            ["launchctl", "bootstrap", self.gui_domain, str(self._installed_plist)],
            capture_output=True,
            text=True,
        )
        if br.returncode == 0:
            return

        svc = self.launchd_service
        rc2, t2 = _launchctl_print(svc)
        pbin = str(self._pheromoned)
        if rc2 == 0 and "state = running" in t2 and pbin in t2:
            return

        raise RuntimeError(
            "launchctl bootstrap failed and com.hiveclaw.pheromoned is not running.\n"
            f"  Domain: {self.gui_domain}  plist: {self._installed_plist}\n"
            f"  stderr: {br.stderr or br.stdout}\n"
            "If you see EIO (error 5) from an IDE terminal, run bootstrap from Terminal.app "
            "(see scripts/README.md)."
        )

    def spawn_server(
        self,
        *,
        port: int = 8080,
        stigmergy: bool = True,
        continuous_batch: bool = True,
        env: dict[str, str] | None = None,
        timeout_s: float = 300.0,
    ) -> subprocess.Popen:
        """
        Start ``scripts/hiveclaw_server.py``; wait until ``/health`` responds.
        Caller must ``terminate()`` the process when done.
        """
        srv = self.repo_root / "scripts" / "hiveclaw_server.py"
        if not srv.is_file():
            raise FileNotFoundError(
                f"{srv} not found. The chat server ships in the HiveClaw repo under scripts/. "
                "Set repo_root to your checkout, or set HIVECLAW_REPO_ROOT and run "
                "`hiveclaw-server` / `python -m hiveclaw_python.server_main`."
            )

        child_env = os.environ.copy()
        child_env["HIVECLAW_REPO_ROOT"] = str(self.repo_root)
        if continuous_batch:
            child_env["HIVECLAW_CONTINUOUS_BATCH"] = "1"
            child_env["HIVECLAW_COMPILE_WARMUP"] = "1"
        if not stigmergy:
            child_env["HIVECLAW_STIGMERGY"] = "0"
        if env:
            child_env.update(env)

        cmd = [
            self.python_exe,
            str(srv),
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port)),
        ]
        if not stigmergy:
            cmd.append("--no-stigmergy")

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        base = f"http://127.0.0.1:{int(port)}"
        deadline = time.time() + float(timeout_s)
        try:
            import httpx
        except ImportError as e:
            proc.terminate()
            raise RuntimeError(
                "httpx required for spawn_server(); pip install -r scripts/requirements-server.txt"
            ) from e

        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"hiveclaw_server exited early (code={proc.returncode}); "
                    "check daemon, models, and SAE (see scripts/README.md)."
                )
            try:
                r = httpx.get(f"{base}/health", timeout=2.0)
                if r.status_code == 200:
                    return proc
            except Exception:
                pass
            time.sleep(0.5)

        proc.terminate()
        raise RuntimeError(f"Timeout waiting for server health at {base}")

    def __enter__(self) -> HiveClawManager:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.teardown:
            try:
                self.daemon_unload()
            except subprocess.CalledProcessError:
                pass
