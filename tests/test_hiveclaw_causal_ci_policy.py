#!/usr/bin/env python3
"""CI policy guards for .github/workflows (no network). Does not change causal math."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_CAUSAL = _WORKFLOWS / "causal.yml"

_USES = re.compile(r"^\s+uses:\s+(\S+)\s*(?:#.*)?$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTINUE = re.compile(r"continue-on-error\s*:")
_CURL_PIPE = re.compile(r"curl\s+[^\n]*\|\s*(?:bash|sh)\b")
_ECHO_SECRET = re.compile(r"echo\s+.*\$\{\{\s*secrets\.")


def _workflow_files() -> list[Path]:
    return sorted(_WORKFLOWS.glob("*.yml")) + sorted(_WORKFLOWS.glob("*.yaml"))


def _strip_comment(line: str) -> str:
    if "#" in line:
        return line[: line.index("#")]
    return line


class TestGitHubActionsPolicy(unittest.TestCase):
    def test_every_uses_is_sha_pinned(self) -> None:
        found = 0
        for path in _workflow_files():
            for i, raw in enumerate(path.read_text().splitlines(), 1):
                m = _USES.match(raw)
                if not m:
                    continue
                ref = m.group(1)
                self.assertNotIn(
                    ".github/workflows/",
                    ref,
                    f"{path}:{i}: reusable workflow calls are not used; unexpected {ref}",
                )
                self.assertIn("@", ref, f"{path}:{i}: missing @ref: {ref}")
                _owner_repo, pin = ref.rsplit("@", 1)
                self.assertRegex(
                    pin,
                    _SHA,
                    f"{path}:{i}: action must be pinned to a 40-char SHA, got {ref}",
                )
                found += 1
        self.assertGreaterEqual(found, 2, "expected checkout + setup-python at minimum")

    def test_no_continue_on_error(self) -> None:
        for path in _workflow_files():
            text = path.read_text()
            self.assertIsNone(
                _CONTINUE.search(text),
                f"{path} sets continue-on-error (forbidden for causal/test jobs)",
            )

    def test_no_curl_pipe_shell(self) -> None:
        for path in _workflow_files():
            self.assertIsNone(
                _CURL_PIPE.search(path.read_text()),
                f"{path} downloads and pipes a remote script",
            )

    def test_no_echo_secrets(self) -> None:
        for path in _workflow_files():
            self.assertIsNone(
                _ECHO_SECRET.search(path.read_text()),
                f"{path} echoes a GitHub secret",
            )

    def test_causal_permissions_contents_read(self) -> None:
        text = _CAUSAL.read_text()
        self.assertRegex(
            text,
            r"(?m)^permissions:\n  contents: read\s*$",
            "causal.yml must declare top-level permissions: contents: read",
        )

    def test_causal_python_version_explicit(self) -> None:
        text = _CAUSAL.read_text()
        self.assertIn('python-version: "3.11"', text)

    def test_causal_invokes_make_test_causal_and_fails_closed(self) -> None:
        """Guard: the causal job must run make test-causal with failure propagation."""
        text = _CAUSAL.read_text()
        self.assertIn("make test-causal PYTHON=python3", text)
        # The Makefile target must still discover hiveclaw_causal tests.
        mk = (_REPO_ROOT / "Makefile").read_text()
        self.assertIn("test_hiveclaw_causal_*.py", mk)
        self.assertIn("unittest discover", mk)
        # The make step must be a real `run:` and not continue-on-error.
        lines = text.splitlines()
        found_run = False
        for i, line in enumerate(lines):
            if "make test-causal PYTHON=python3" in line:
                found_run = True
                window = "\n".join(lines[max(0, i - 6) : i + 3])
                self.assertIn("run:", window)
                self.assertNotIn("continue-on-error", window)
        self.assertTrue(found_run)

    def test_causal_checkout_does_not_persist_credentials(self) -> None:
        text = _CAUSAL.read_text()
        self.assertIn("persist-credentials: false", text)
        self.assertIn("fetch-depth: 1", text)

    def test_no_dependency_cache_without_lockfile(self) -> None:
        """Causal job must not restore an unkeyed pip cache (no lockfile for mypy)."""
        uncommented = "\n".join(
            _strip_comment(line) for line in _CAUSAL.read_text().splitlines()
        )
        self.assertNotRegex(uncommented, r"(?m)^\s+cache:\s+pip\s*$")


if __name__ == "__main__":
    unittest.main()
