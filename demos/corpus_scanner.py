#!/usr/bin/env python3
"""
Deterministic repo scanner for Repo Pulse demo.

Extracts compact Rust/Python findings from fixed target directories with
stable ordering and snippet caps so demo runtime remains predictable.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_FILES_DEFAULT = 60
MAX_SNIPPET_CHARS_DEFAULT = 300

RUST_TARGETS_DEFAULT = (
    "crates/hiveclaw-core",
    "crates/hiveclaw-daemon",
)
PY_TARGETS_DEFAULT = ("crates/hiveclaw-python/python/hiveclaw_python",)


@dataclass
class RustFinding:
    file: str
    line: int
    category: str
    snippet: str


@dataclass
class PyFinding:
    file: str
    line: int
    category: str
    snippet: str


@dataclass
class CorpusReport:
    rust_findings: list[RustFinding]
    py_findings: list[PyFinding]
    scanned_files: int
    max_files: int
    max_snippet_chars: int

    def to_json_dict(self) -> dict:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _collect_files(root: Path, rel_dirs: tuple[str, ...], suffix: str) -> list[Path]:
    files: list[Path] = []
    for rel_dir in rel_dirs:
        base = root / rel_dir
        if not base.exists():
            continue
        files.extend(sorted(base.rglob(f"*{suffix}")))
    # Stable order + dedupe while preserving order
    out: list[Path] = []
    seen: set[Path] = set()
    for f in sorted(files):
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _clip(s: str, max_chars: int) -> str:
    s = s.strip().replace("\t", "    ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _scan_rust_file(path: Path, root: Path, max_chars: int) -> list[RustFinding]:
    findings: list[RustFinding] = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(root))
    for i, line in enumerate(txt.splitlines(), start=1):
        stripped = line.strip()
        if ".unwrap()" in stripped:
            findings.append(
                RustFinding(rel, i, "unwrap_call", _clip(stripped, max_chars))
            )
        if ".expect(" in stripped:
            findings.append(
                RustFinding(rel, i, "expect_call", _clip(stripped, max_chars))
            )
        if re.search(r"\bunsafe\b", stripped):
            findings.append(
                RustFinding(rel, i, "unsafe_usage", _clip(stripped, max_chars))
            )
        if re.search(r"\bResult<", stripped) and "->" in stripped:
            findings.append(
                RustFinding(
                    rel,
                    i,
                    "result_signature",
                    _clip(stripped, max_chars),
                )
            )
    return findings


def _public_method(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")


def _missing_hints(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for a in list(node.args.args) + list(node.args.kwonlyargs):
        if a.arg in ("self", "cls"):
            continue
        if a.annotation is None:
            return True
    if node.args.vararg and node.args.vararg.annotation is None:
        return True
    if node.args.kwarg and node.args.kwarg.annotation is None:
        return True
    if node.returns is None:
        return True
    return False


def _scan_python_file(path: Path, root: Path, max_chars: int) -> list[PyFinding]:
    findings: list[PyFinding] = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(root))
    try:
        tree = ast.parse(txt)
    except SyntaxError as e:
        findings.append(
            PyFinding(rel, int(e.lineno or 1), "parse_error", _clip(str(e), max_chars))
        )
        return findings

    lines = txt.splitlines()
    for node in ast.walk(tree):
        if _public_method(node):
            line = int(getattr(node, "lineno", 1))
            if _missing_hints(node):
                snippet = lines[line - 1] if 0 < line <= len(lines) else f"def {node.name}(...)"
                findings.append(
                    PyFinding(
                        rel,
                        line,
                        "missing_type_hints",
                        _clip(snippet, max_chars),
                    )
                )
            ds = ast.get_docstring(node, clean=False)
            if ds is None or not ds.strip():
                snippet = lines[line - 1] if 0 < line <= len(lines) else f"def {node.name}(...)"
                findings.append(
                    PyFinding(
                        rel,
                        line,
                        "missing_docstring",
                        _clip(snippet, max_chars),
                    )
                )
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            line = int(getattr(node, "lineno", 1))
            snippet = lines[line - 1] if 0 < line <= len(lines) else "except:"
            findings.append(
                PyFinding(
                    rel,
                    line,
                    "bare_except",
                    _clip(snippet, max_chars),
                )
            )
    return findings


def build_corpus_report(
    *,
    max_files: int = MAX_FILES_DEFAULT,
    max_snippet_chars: int = MAX_SNIPPET_CHARS_DEFAULT,
    rust_targets: tuple[str, ...] = RUST_TARGETS_DEFAULT,
    py_targets: tuple[str, ...] = PY_TARGETS_DEFAULT,
) -> CorpusReport:
    root = _repo_root()
    rust_files = _collect_files(root, rust_targets, ".rs")
    py_files = _collect_files(root, py_targets, ".py")

    all_files: list[tuple[str, Path]] = [("rust", p) for p in rust_files] + [
        ("py", p) for p in py_files
    ]
    all_files = sorted(all_files, key=lambda t: str(t[1]))[: max(1, max_files)]

    rust_findings: list[RustFinding] = []
    py_findings: list[PyFinding] = []
    for kind, path in all_files:
        if kind == "rust":
            rust_findings.extend(_scan_rust_file(path, root, max_snippet_chars))
        else:
            py_findings.extend(_scan_python_file(path, root, max_snippet_chars))

    return CorpusReport(
        rust_findings=rust_findings,
        py_findings=py_findings,
        scanned_files=len(all_files),
        max_files=max_files,
        max_snippet_chars=max_snippet_chars,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic corpus scanner for Repo Pulse demo")
    p.add_argument("--max-files", type=int, default=MAX_FILES_DEFAULT)
    p.add_argument("--max-snippet-chars", type=int, default=MAX_SNIPPET_CHARS_DEFAULT)
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    rep = build_corpus_report(
        max_files=max(1, int(args.max_files)),
        max_snippet_chars=max(64, int(args.max_snippet_chars)),
    )
    payload = json.dumps(rep.to_json_dict(), indent=2)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
