"""Console entrypoint ``hiveclaw-server``: :mod:`hiveclaw_python.openai_server`."""

from __future__ import annotations


def main() -> None:
    from .openai_server import main as run_impl

    run_impl()


if __name__ == "__main__":
    main()
