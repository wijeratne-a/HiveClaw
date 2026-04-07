#!/usr/bin/env python3
"""Shim: use ``hiveclaw-overseer`` or :mod:`hiveclaw_python.cli.overseer`."""

from __future__ import annotations

from hiveclaw_python.cli.overseer import main

if __name__ == "__main__":
    main()
