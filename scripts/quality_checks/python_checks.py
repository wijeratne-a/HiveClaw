"""Python-specific AST, security, compile, ruff, and style checks."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common_checks import Violation, ViolationSeverity


def check_python_parse(code: str) -> tuple[ast.AST | None, list[Violation]]:
    try:
        return ast.parse(code), []
    except SyntaxError as e:
        return None, [
            Violation(
                "SYN_PARSE_ERROR",
                ViolationSeverity.critical,
                f"SyntaxError: {e.msg} (line {e.lineno})",
                e.lineno,
            )
        ]


def check_security(tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                violations.append(
                    Violation(
                        "SEC_EVAL_EXEC",
                        ViolationSeverity.critical,
                        f"Forbidden call: {node.func.id}()",
                        getattr(node, "lineno", None),
                    )
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "eval",
                "exec",
            ):
                violations.append(
                    Violation(
                        "SEC_EVAL_EXEC_ATTR",
                        ViolationSeverity.critical,
                        f"Forbidden attribute call: .{node.func.attr}()",
                        getattr(node, "lineno", None),
                    )
                )
            for kw in node.keywords:
                if kw.arg == "shell":
                    use_shell = False
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        use_shell = True
                    elif isinstance(kw.value, ast.Name) and kw.value.id == "True":
                        use_shell = True
                    if use_shell:
                        fn = "call"
                        if isinstance(node.func, ast.Attribute):
                            fn = node.func.attr
                        elif isinstance(node.func, ast.Name):
                            fn = node.func.id
                        violations.append(
                            Violation(
                                "SEC_SUBPROCESS_SHELL_TRUE",
                                ViolationSeverity.critical,
                                f"Forbidden subprocess.{fn}(..., shell=True)",
                                getattr(node, "lineno", None),
                            )
                        )
            self.generic_visit(node)

    V().visit(tree)
    return violations


def _is_public_def(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return not node.name.startswith("_")


def _function_has_complete_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return False
    for arg in node.args.args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            return False
    for arg in getattr(node.args, "kwonlyargs", []) or []:
        if arg.annotation is None:
            return False
    for arg in getattr(node.args, "posonlyargs", []) or []:
        if arg.annotation is None:
            return False
    return True


def check_docstrings(tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public_def(node):
                continue
            if ast.get_docstring(node) is None:
                violations.append(
                    Violation(
                        "QUAL_MISSING_DOCSTRING",
                        ViolationSeverity.warning,
                        f"Public function {node.name!r} missing docstring",
                        node.lineno,
                    )
                )
    return violations


def check_type_hints(tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public_def(node):
                continue
            if not _function_has_complete_annotations(node):
                violations.append(
                    Violation(
                        "QUAL_INCOMPLETE_TYPE_HINTS",
                        ViolationSeverity.warning,
                        (
                            f"Public function {node.name!r} needs parameter and "
                            "return type hints"
                        ),
                        node.lineno,
                    )
                )
    return violations


def check_py_compile(code: str) -> list[Violation]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(tmp)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return []
        msg = (proc.stderr or proc.stdout or "").strip()[:500]
        return [
            Violation(
                "TOOL_PY_COMPILE",
                ViolationSeverity.critical,
                f"py_compile failed: {msg}",
            )
        ]
    finally:
        tmp.unlink(missing_ok=True)


def check_ruff(code: str) -> list[Violation]:
    exe = shutil.which("ruff")
    if not exe:
        return []
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            [exe, "check", str(tmp)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            return []
        err = ((proc.stdout or "") + (proc.stderr or "")).strip()[:800]
        return [
            Violation(
                "TOOL_RUFF",
                ViolationSeverity.warning,
                f"ruff check: {err}",
            )
        ]
    finally:
        tmp.unlink(missing_ok=True)
