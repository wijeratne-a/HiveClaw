#!/usr/bin/env python3
"""Verify launchd pheromoned matches this checkout and SlabClient can handshake."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


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


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: hiveclaw_doctor.py <repo_root> <expected_pheromoned_path>",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    pbin = Path(sys.argv[2]).resolve()
    uid = os.getuid()
    domain = f"gui/{uid}"
    svc = f"{domain}/com.hiveclaw.pheromoned"

    rc, text = _launchctl_print(svc)
    ok = True

    low = text.lower()
    if rc != 0 and ("could not find" in low or "not found" in low):
        print(f"[doctor] Service not loaded: {svc}", file=sys.stderr)
        print(
            "[doctor] From repo root: cargo build --release -p hiveclaw-daemon && make daemon-load",
            file=sys.stderr,
        )
        ok = False
    elif "state = running" not in text:
        print(
            f"[doctor] Job does not look running (no 'state = running' in launchctl print).",
            file=sys.stderr,
        )
        ok = False

    prog = _parse_program(text)
    if prog is None:
        print(
            "[doctor] Could not parse program= from launchctl output (see below).",
            file=sys.stderr,
        )
        ok = False
    else:
        loaded = Path(prog).resolve()
        if loaded != pbin.resolve():
            print("[doctor] Program path mismatch:", file=sys.stderr)
            print(f"  launchd:  {loaded}", file=sys.stderr)
            print(f"  expected: {pbin.resolve()}", file=sys.stderr)
            print(
                "[doctor] You may have moved the repo or an old plist. "
                "Run: make daemon-uninstall && make daemon-load",
                file=sys.stderr,
            )
            ok = False

    if not pbin.is_file():
        print(
            f"[doctor] Missing release binary: {pbin} (run: cargo build --release -p hiveclaw-daemon)",
            file=sys.stderr,
        )
        ok = False

    print("[doctor] launchctl print excerpt:")
    for line in text.splitlines()[:40]:
        print(f"  {line}")
    if len(text.splitlines()) > 40:
        print("  ...")

    try:
        import hiveclaw_python  # noqa: PLC0415 — only after venv / maturin

        hiveclaw_python.SlabClient()
        print("[doctor] SlabClient(): OK")
    except Exception as e:
        print(f"[doctor] SlabClient(): {e}", file=sys.stderr)
        ok = False

    if ok:
        print("[doctor] All checks passed.")
        return 0
    print("[doctor] One or more checks failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
