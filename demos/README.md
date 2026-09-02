# Repo Pulse Demo

`demos/` contains the following:

- **Repo Pulse** — "60-Second Repo Pulse" showcase: transcript-heavy baseline vs HiveClaw audit swarm, SAE feature dashboard, Rich TUI scoreboard.
- **Consensus Showdown** — LangChain string committee vs HiveClaw latent committee (slab + steering); growing prompt context vs ~flat context; see below.

## Consensus Showdown (LangChain vs HiveClaw)

This demo runs the same multi-agent committee task as [`benchmarks/langchain_string_swarm.py`](../benchmarks/langchain_string_swarm.py) (LangChain prompt assembly + `mlx_lm` generation) and [`benchmarks/hiveclaw_consensus.py`](../benchmarks/hiveclaw_consensus.py) (slab read/write + `ActiveSteeringWrapper`) **sequentially**, with a **Rich** live panel and a final scoreboard plus per-round context bar charts.

**Does not** start `hiveclaw-server`; the HiveClaw path talks to **pheromoned** via `SlabClient` and loads **MLX** locally.

### Prerequisites

From repo root:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make daemon-load
make doctor PYTHON="$PWD/.venv/bin/python3"
pip install -r requirements/requirements-server.txt
pip install -r requirements/requirements-bench-langchain.txt
```

**SAE weights** (required for `hiveclaw_consensus`): `models/hiveclaw_sae_v1.safetensors` (see `training/harvester.py` and `training/train_sae.py` in the repo if you need to produce them).

### Run

```bash
python -m demos.consensus_showdown --rounds 10 --agents 5
```

Quick smoke (shorter):

```bash
python -m demos.consensus_showdown --rounds 3 --agents 3 --tokens-per-turn 12
```

HiveClaw phase only (skip LangChain — e.g. if LangChain is not installed yet):

```bash
python -m demos.consensus_showdown --no-langchain --rounds 5
```

Optional JSON summary file:

```bash
python -m demos.consensus_showdown --json-out consensus_showdown.json
```

### What to look for

- **LangChain row:** `total_coord_tokens` is large; **per-round max `ctx_tokens`** grows round-over-round (prior discussion appended to the prompt).
- **HiveClaw row:** `total_coord_tokens` is **0** (by benchmark definition: no growing transcript for coordination); **per-round context** stays ~flat.
- **Speedup** row: wall-clock ratio **LangChain wall_s / HiveClaw wall_s** (varies by machine; measure on your hardware).

### Safe claims

You **can** say this benchmark shows **string-passing coordination tax** (prompt tokens not attributable to the raw code body) vs **latent slab coordination** on the same MLX model and task, with metrics emitted by the existing benchmark scripts.

You **should not** claim "zero tokens for the whole model" — each agent still has a **fixed-size task prompt**; what goes away is the **growing committee transcript** in the Hive path. Do not quote a fixed speedup (e.g. 3.8×) without measuring on your run.

Equivalent non-TUI entry points: `python benchmarks/benchmark_external.py` or `python benchmarks/benchmark_consensus.py`.

---

## Prerequisites (Repo Pulse)

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

From the **repository root** (so package imports resolve):

```bash
python -m demos.run_repo_pulse --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b --max-files 60
```

Alternative if you prefer running the script path:

```bash
PYTHONPATH=. python demos/run_repo_pulse.py --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b --max-files 60
```

Outputs:

- `HEALTH_REPORT.md` (HiveClaw path)
- `HEALTH_REPORT_BASELINE.md` (baseline path)

### Flags

- **`--max-files`**: Cap for the deterministic corpus scanner. Use **`20`** (or similar) for quick dry runs; **`60`** for a fuller scan (more comparable across runs).
- **`--feature-slot`**: IOSurface slot index for the SAE feature dashboard (`read_slot_v5`). Default **`0`**. If the panel stays flat, align this with the slot your server/agents write (see server docs).

## Run paths individually

HiveClaw path:

```bash
python -m demos.audit_swarm --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b
```

Baseline path:

```bash
python -m demos.baseline_audit --base-url http://127.0.0.1:8080 --model hiveclaw-swarm-8b
```

Scanner only:

```bash
python -m demos.corpus_scanner --max-files 60
```

## Recording checklist

1. Run 3-5 dry runs and capture median wall times.
2. Keep `--max-files` fixed for reproducibility.
3. Use a clean terminal and stable viewport size.
4. Keep server logs visible in a side pane for credibility.
5. Verify `HEALTH_REPORT.md` exists before recording final take.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'demos'`**: Run from repo root using **`python -m demos.<script>`** or set **`PYTHONPATH`** to the repo root (see above).
- **TUI shows `0` elapsed briefly**: While a path is `running`, elapsed updates live; when both finish, the scoreboard shows **`wall_s`** for each side.
- **`report_items` is 0**: That field is the count of **parsed JSON issue objects** from Agents A/B. If the model wraps output in prose, the parser may see no rows; the Markdown report can still be written.

## Safe claim guide

Use:

- "Coordination context stays near-zero (no growing transcript passed between agents)."
- "SAE feature proxy visualizes active latent dimensions."
- "Local-only run path: no cloud API key required."

Avoid:

- "Zero tokens total."
- "Agent C decodes findings from epochs alone."
- "Feature labels are perfect semantic truth."
