"""
Shared Markdown validation for Repo Pulse HEALTH_REPORT outputs.
"""

from __future__ import annotations

import re

from demos.corpus_scanner import CorpusReport

_REQUIRED = (
    "# HiveClaw Repo Health Report",
    "## Rust Safety Issues",
    "## Python Type Safety Issues",
    "## Summary",
)

# Table rows like | ... | ... | with only ellipsis as cell content
_ELLIPSIS_ROW = re.compile(r"^\s*\|[^|\n]*\|\s*\.\.\.\s*\|", re.MULTILINE)


def validate_health_report_markdown(md: str) -> bool:
    if not all(s in md for s in _REQUIRED):
        return False
    if "|...|" in md:
        return False
    if _ELLIPSIS_ROW.search(md):
        return False
    return True


def format_allowed_paths_instructions(rep: CorpusReport) -> str:
    """Prompt fragment: corpus-only file paths for Architect grounding."""
    paths = sorted({f.file for f in rep.rust_findings} | {f.file for f in rep.py_findings})
    if not paths:
        return (
            "Allowed file paths: (scanner found no file paths). "
            "Do not invent generic filenames; use only paths implied by the findings JSON below.\n"
        )
    block = "\n".join(f"- {p}" for p in paths)
    return (
        "Allowed file paths — cite ONLY these paths in your tables; "
        "do not invent generic names (e.g. main.rs, util.py):\n"
        f"{block}\n"
    )
