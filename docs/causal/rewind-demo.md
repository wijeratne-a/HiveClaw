# The Rewind — local demo

CPU-only causal runtime (`hiveclaw_causal/`). No GPU, daemon, or LLM.

From the repo root with `.venv` active:

```bash
source .venv/bin/activate
python -m hiveclaw_causal.demo_rewind
# equivalent: python -m hiveclaw_causal
```

Writes `output/rewind.sqlite` (gitignored if you add it; default path is under `output/`). Inspect one object:

```bash
python -m hiveclaw_causal.inspect --db output/rewind.sqlite --id claim-cache-regression
python -m hiveclaw_causal.inspect --db output/rewind.sqlite --id action-rollback-release
```

Tests:

```bash
python tests/test_hiveclaw_causal_engine.py
python tests/test_hiveclaw_causal_policy.py
python tests/test_hiveclaw_causal_rewind.py
```

Outage support percentage is `hiveclaw_causal.stats.outage_explains_pct` over generated failure timestamps — not a string constant. Architecture: `docs/adr/CAUSAL_RUNTIME_H5.md`.
