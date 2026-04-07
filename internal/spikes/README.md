# Research spikes (unsupported entrypoints)

These scripts are **experimental** demos and load tests. They are not part of the supported
user path documented in the root [README.md](../../README.md). Prefer
[examples/hello_swarm.py](../../examples/hello_swarm.py) and the OpenAI-compatible server for
day-to-day use.

| Script | Role |
|--------|------|
| `intelligence_spike.py` | Two-agent SAE + slab LLM handshake |
| `swarm_spike.py` | Synthetic slot contention / matmul under claim |
| `llm_swarm.py` | LLM + slab steering demo |
| `overseer_demo.py` | Overseer inhibit/reroute without LLM |

Run from repo root, venv active, daemon loaded:  
`python internal/spikes/<script>.py`
