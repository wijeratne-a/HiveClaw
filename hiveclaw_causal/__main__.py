"""python -m hiveclaw_causal [demo|store-status|verify-store|backup|restore|migrate|inspect]."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "demo",
        "store-status",
        "verify-store",
        "backup",
        "restore",
        "migrate",
        "inspect",
        "-h",
        "--help",
    }
    if not args or args[0] not in commands:
        from .demo_rewind import main as demo_main

        return demo_main(args)
    from .cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
