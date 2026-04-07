#!/usr/bin/env python3
"""
Deprecated path: prefer ``hiveclaw-server`` or ``python -m hiveclaw_python.server_main``.
"""

from __future__ import annotations

import sys

from hiveclaw_python.openai_server import main

if __name__ == "__main__":
    print(
        "Note: use `hiveclaw-server` or `python -m hiveclaw_python.server_main` "
        "(scripts/hiveclaw_server.py is a compatibility shim).",
        file=sys.stderr,
    )
    main()
