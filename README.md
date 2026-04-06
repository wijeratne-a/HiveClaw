# HiveClaw

**VRAM-native multi-agent coordination for Apple Silicon** — a Rust daemon (`pheromoned`) exposes a Metal-backed IOSurface slab; Python MLX clients coordinate through **256-D bf16 latents** (SAE space) instead of growing JSON chat logs.

| Ironclad engine | macOS Metal | License |
|-----------------|-------------|---------|
| [Burn-in + zero `eager_fallback`](#ironclad-engine-proof) | IOSurface + XPC | AGPL-3.0 ([`LICENSE`](LICENSE)) |

## What it does

1. **Shared slab:** Thousands of slots in one IOSurface; agents **claim**, **read/write v5 latents**, and **release** with Mach-era semantics and torn-read protection.
2. **Steered generation:** Llama-class models run with an SAE tying **2048-D hidden ↔ 256-D slab** so the last layer can inject peer state without re-tokenizing megabytes of chat history.
3. **Continuous batching (Phase 7):** Optional compiled decode path; [`scripts/verify_burn_in.sh`](scripts/verify_burn_in.sh) gates **zero `eager_fallback`** under concurrent SSE load.

## Benchmark: coordination without a token tax

Multi-agent “committee” task (5 reviewers × 10 rounds) comparing:

| Path | Coordination tokens | Content tokens (typical) | Context growth | How to reproduce |
|------|----------------------|---------------------------|----------------|------------------|
| **String-passing baseline** | High (prompts include full prior discussion) | Same order | Grows every round | `python scripts/benchmark_consensus.py --no-hiveclaw` |
| **HiveClaw latent path** | **0** (no text passed between agents) | Same order | ~constant | `python scripts/benchmark_consensus.py` (needs daemon + SAE) |

Run [`scripts/benchmark_consensus.py`](scripts/benchmark_consensus.py) on your Mac and paste the printed table into docs or CI artifacts. Internal stigmergy A/B (server on vs off) remains in [`scripts/benchmark_stigmergy.py`](scripts/benchmark_stigmergy.py).

## Quick start (5-line style)

From the repo root with venv active, `make python`, `pip install -r scripts/requirements-server.txt`, models + SAE present:

```python
import hiveclaw_python as hc
swarm = hc.LocalSwarm(model="mlx-community/Llama-3.2-1B-Instruct-4bit")
swarm.add_agent(slot=1, goal="Say hello in one sentence.")
swarm.run()  # bootstraps daemon if needed, spawns hiveclaw_server, streams SSE
swarm.stop()
```

Lower-level slab-only example: `python examples/hello_swarm.py --slab-only`. Full stack demo: `python examples/hello_swarm.py`.

One-shot daemon bootstrap from Python:

```python
import hiveclaw_python as hc
m = hc.init(build_if_missing=True)   # install LaunchAgent + launchctl bootstrap
proc = m.spawn_server(port=8080)     # optional: background hiveclaw_server
# ... use httpx against http://127.0.0.1:8080 ...
proc.terminate()
```

## How it works (IOSurface / Metal / XPC)

- **`pheromoned`** (LaunchAgent) owns the Mach service **`com.hiveclaw.pheromoned`** and brokers **XPC** + **IOSurface** handoff to clients.
- The slab layout is versioned (v5: **640 B/slot**, **256×bf16** payload, epoch words for torn-read detection). Python uses **`hiveclaw_python.SlabClient`** (`read_slot_v5`, `write_slot_v5`, batched variants, `claim_task` / `release_task`).
- **Metal** paths in `hiveclaw_mlx` can accelerate batched slab traffic; default integration tests still validate CPU-side correctness.
- **No central JSON router** is required for peer state: the “sentinel” is the shared surface + SAE geometry, not a Redis topic.

See [`scripts/README.md`](scripts/README.md) for setup, `make daemon-load`, and troubleshooting (`launchctl` EIO from IDE terminals).

## Ironclad engine proof

The repo ships an exit-0 gate intended for a **local Apple Silicon Mac** with GUI `launchctl`, MLX weights, and the default SAE:

```bash
bash scripts/ci_ironclad_verify.sh
```

This runs **`make doctor`** (daemon path + `SlabClient` handshake) then **[`scripts/verify_burn_in.sh`](scripts/verify_burn_in.sh)** — SSE load, **`burn_in.py`** criteria (including HTTP 503 under overload), and **zero** `eager_fallback` JSON events on the server log. Phase 7 defaults compile the inner decode step when **`HIVECLAW_COMPILE_DECODE=1`** and **`HIVECLAW_COMPILE_WARMUP=1`**.

Optional GitHub Actions workflow: [`.github/workflows/ironclad-burn-in.yml`](.github/workflows/ironclad-burn-in.yml) (typically needs self-hosted macOS).

## Repository layout

- **`crates/hiveclaw-daemon`** — `pheromoned` binary and XPC surface broker.
- **`crates/hiveclaw-python`** — PyO3 + **`hiveclaw_python`** (SlabClient, `HiveClawManager`, `init`, `LocalSwarm`, `Swarm`).
- **`scripts/`** — MLX spikes, [`hiveclaw_server.py`](scripts/hiveclaw_server.py) (OpenAI-compatible API + continuous batching), benchmarks, burn-in.

## Licensing

HiveClaw is open-source under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [`LICENSE`](LICENSE).

Commercial use where AGPL is not feasible requires a **commercial license**. For inquiries, open a GitHub Discussion or an issue (e.g. label `commercial`), or contact the maintainers through the channel where you obtained the source.
