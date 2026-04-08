# Repo Pulse Demo

`demos/` contains the implementation for the "60-Second Repo Pulse" showcase:

- transcript-heavy baseline audit (JSON shared transcript growth),
- HiveClaw audit swarm (compact structured findings),
- live SAE feature proxy dashboard from slab latents,
- split-screen Rich TUI scoreboard.

## Prerequisites

From repo root:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make daemon-load
make doctor PYTHON="$PWD/.venv/bin/python3"
pip install -r requirements/requirements-server.txt
```

Start the server in another terminal:

```bash
HIVECLAW_TWO_AGENT=1 hiveclaw-server --host 127.0.0.1 --port 8080
```

## One-time calibration (optional but recommended)

Build/refresh feature labels for SAE dimensions:

```bash
python demos/scripts/label_sae_dims.py
```

This writes:

- `demos/data/feature_dictionary.json`

## Run the full demo

```bash
python demos/run_repo_pulse.py --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b --max-files 60
```

Outputs:

- `HEALTH_REPORT.md` (HiveClaw path)
- `HEALTH_REPORT_BASELINE.md` (baseline path)

## Run paths individually

HiveClaw path:

```bash
python demos/audit_swarm.py --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b
```

Baseline path:

```bash
python demos/baseline_audit.py --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b
```

Scanner only:

```bash
python demos/corpus_scanner.py --max-files 60
```

## Recording checklist

1. Run 3-5 dry runs and capture median wall times.
2. Keep `--max-files` fixed for reproducibility.
3. Use a clean terminal and stable viewport size.
4. Keep server logs visible in a side pane for credibility.
5. Verify `HEALTH_REPORT.md` exists before recording final take.

## Safe claim guide

Use:

- "Coordination context stays near-zero (no growing transcript passed between agents)."
- "SAE feature proxy visualizes active latent dimensions."
- "Local-only run path: no cloud API key required."

Avoid:

- "Zero tokens total."
- "Agent C decodes findings from epochs alone."
- "Feature labels are perfect semantic truth."
