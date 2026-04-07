# Demo assets

## `demo.gif`

The checked-in [`demo.gif`](demo.gif) is an **illustrative** animation (side-by-side JSON vs latent slab timing). To replace it with a screen recording of the real terminal UI:

1. macOS, Apple Silicon, 80-column terminal, dark background.
2. Install demo deps: `pip install -r requirements/requirements-server.txt`
3. Run: `python examples/hiveclaw_top.py --mock-only` (or `python examples/speed_test_viz.py --mock-only` — same script)
4. Record ~15s (e.g. [asciinema](https://asciinema.org/) then [agg](https://github.com/asciinema/agg), or QuickTime + `ffmpeg` to GIF).
5. Overwrite `docs/assets/demo.gif` (keep under ~5 MB for GitHub README load times).
