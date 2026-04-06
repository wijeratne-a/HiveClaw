"""FastAPI chat server (installed package)."""

from .openai_server import app, main

__all__ = ["app", "main"]
