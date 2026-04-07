#!/usr/bin/env python3
"""Unit tests for quality checks and QualityController logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quality_gate.quality_checks.common_checks import (  # noqa: E402
    ViolationSeverity,
    check_fence_extraction,
    extract_single_fence,
)
from quality_gate.quality_checks.python_checks import (  # noqa: E402
    check_py_compile,
    check_python_parse,
    check_security,
)
from quality_gate.quality_controller import (  # noqa: E402
    QualityController,
    QualityGateFailure,
    QualityProfile,
    format_repair_prompt,
    load_profile,
    verify_assistant_output,
)


class TestFence(unittest.TestCase):
    def test_no_fence(self) -> None:
        code, v = extract_single_fence("just prose")
        self.assertIsNone(code)
        self.assertTrue(any(x.rule_id == "FMT_NO_FENCE" for x in v))

    def test_multiple_fences(self) -> None:
        text = "```python\na=1\n```\n```python\nb=2\n```"
        code, v = extract_single_fence(text)
        self.assertIsNone(code)
        self.assertTrue(any(x.rule_id == "FMT_MULTIPLE_FENCES" for x in v))

    def test_empty_fence(self) -> None:
        code, v = extract_single_fence("```python\n```")
        self.assertIsNone(code)
        self.assertTrue(any(x.rule_id == "FMT_EMPTY_FENCE" for x in v))

    def test_valid_single_fence(self) -> None:
        code, v = extract_single_fence("```python\nx = 1\n```")
        self.assertEqual(code, "x = 1")
        self.assertEqual(v, [])

    def test_check_fence_extraction_disabled(self) -> None:
        c, v = check_fence_extraction("hello", fence_required=False)
        self.assertIsNone(c)
        self.assertEqual(v, [])


class TestSecurity(unittest.TestCase):
    def test_eval_detected(self) -> None:
        tree, _ = check_python_parse("def f():\n    eval('1')\n")
        assert tree is not None
        v = check_security(tree)
        self.assertTrue(any(x.rule_id == "SEC_EVAL_EXEC" for x in v))

    def test_exec_detected(self) -> None:
        tree, _ = check_python_parse("exec('pass')\n")
        assert tree is not None
        v = check_security(tree)
        self.assertTrue(any(x.rule_id == "SEC_EVAL_EXEC" for x in v))

    def test_shell_true(self) -> None:
        tree, _ = check_python_parse(
            "import subprocess\nsubprocess.run('x', shell=True)\n"
        )
        assert tree is not None
        v = check_security(tree)
        self.assertTrue(any(x.rule_id == "SEC_SUBPROCESS_SHELL_TRUE" for x in v))

    def test_clean_passes(self) -> None:
        tree, _ = check_python_parse("def f():\n    return 1\n")
        assert tree is not None
        self.assertEqual(check_security(tree), [])


class TestPyCompile(unittest.TestCase):
    def test_broken_syntax(self) -> None:
        v = check_py_compile("def oops(\n")
        self.assertTrue(any(x.rule_id == "TOOL_PY_COMPILE" for x in v))

    def test_valid(self) -> None:
        v = check_py_compile("def f():\n    return 42\n")
        self.assertEqual(v, [])


class TestRuffOptional(unittest.TestCase):
    def test_ruff_skips_gracefully(self) -> None:
        import shutil

        from quality_gate.quality_checks.python_checks import check_ruff

        if not shutil.which("ruff"):
            self.assertEqual(check_ruff("x=1\n"), [])
        else:
            v = check_ruff("import os\n")  # unused import often flagged
            self.assertIsInstance(v, list)


class TestVerifyAndProfile(unittest.TestCase):
    def _minimal_profile(self, **kwargs: object) -> QualityProfile:
        base = dict(
            artifact_type="python",
            hard_blockers=frozenset({"FMT_NO_FENCE", "SYN_PARSE_ERROR"}),
            warn_checks=frozenset(),
            max_retries=2,
            report_only=False,
            fence_required=True,
            ruff=False,
            py_compile=False,
            require_type_hints=False,
            require_docstrings=False,
            retry_on_warn=False,
        )
        base.update(kwargs)
        return QualityProfile(**base)  # type: ignore[arg-type]

    def test_accept_clean(self) -> None:
        p = self._minimal_profile(
            hard_blockers=frozenset(
                {
                    "FMT_NO_FENCE",
                    "FMT_MULTIPLE_FENCES",
                    "FMT_EMPTY_FENCE",
                    "SYN_PARSE_ERROR",
                    "SEC_EVAL_EXEC",
                    "SEC_SUBPROCESS_SHELL_TRUE",
                }
            )
        )
        text = "```python\ndef f() -> int:\n    return 1\n```"
        r = verify_assistant_output(text, p, report_only=False)
        self.assertEqual(r.decision, "ACCEPT")
        self.assertIsNotNone(r.extracted_code)

    def test_retry_on_blocker(self) -> None:
        p = self._minimal_profile()
        r = verify_assistant_output("no fence", p, report_only=False)
        self.assertEqual(r.decision, "RETRY")

    def test_report_only_accepts(self) -> None:
        p = self._minimal_profile()
        r = verify_assistant_output("no fence", p, report_only=True)
        self.assertEqual(r.decision, "ACCEPT")

    def test_load_profile_real_file(self) -> None:
        path = _REPO_ROOT / "quality_gate" / "quality_profiles" / "python_refactor.yaml"
        self.assertTrue(path.is_file(), "expected bundled profile")
        prof = load_profile(path)
        self.assertEqual(prof.artifact_type, "python")
        self.assertIn("FMT_NO_FENCE", prof.hard_blockers)

    def test_load_profile_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_profile(Path("/nonexistent/no.yaml"))

    def test_format_repair_dedupes(self) -> None:
        from quality_gate.quality_checks.common_checks import Violation

        v = [
            Violation("A", ViolationSeverity.critical, "m1"),
            Violation("A", ViolationSeverity.critical, "m2"),
            Violation("B", ViolationSeverity.warning, "m3"),
        ]
        s = format_repair_prompt(v)
        self.assertIn("[A]", s)
        self.assertIn("[B]", s)
        self.assertEqual(s.count("[A]"), 1)


class TestRunTurnRetries(unittest.TestCase):
    def test_exhaust_raises(self) -> None:
        path = _REPO_ROOT / "quality_gate" / "quality_profiles" / "python_refactor.yaml"
        qc = QualityController(path, report_only=False)
        qc.profile = replace_profile_retries(qc.profile, 0)

        outputs = ["bad"]

        def call_fn(_u: str) -> tuple[str, None]:
            return (outputs.pop(0), None)

        with self.assertRaises(QualityGateFailure):
            qc.run_turn(call_fn, "user", label="t", role_name="Architect")


def replace_profile_retries(p: QualityProfile, n: int) -> QualityProfile:
    return QualityProfile(
        artifact_type=p.artifact_type,
        hard_blockers=p.hard_blockers,
        warn_checks=p.warn_checks,
        max_retries=n,
        report_only=p.report_only,
        fence_required=p.fence_required,
        ruff=False,
        py_compile=False,
        require_type_hints=p.require_type_hints,
        require_docstrings=p.require_docstrings,
        retry_on_warn=p.retry_on_warn,
    )


if __name__ == "__main__":
    unittest.main()
