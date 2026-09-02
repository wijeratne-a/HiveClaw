# exp-001 — targeted repair vs naive full rerun

**Date:** 2026-08-31  
**HEAD when measured:** local `main` after Part 1 (`b1b3a50`) plus this experiment’s code.  
**Command:** `python -m hiveclaw_causal.benchmark` (seed 42), twice.  
**Interpreter:** `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` 3.11.1.

## Hypothesis

After the provider-outage artifact is injected, **targeted repair** (reverse-dependency traversal + skip unrelated tasks) uses **fewer units of work** than a **naive full rerun** that re-evaluates every object, while reaching the **same final conclusion** (same `outage_explains_pct`, rollback blocked, follow-up investigation task present).

“Units of work” here are **deterministic evaluation steps** (`WorkCounter.inspect`: claim overlap checks, invalidation-graph visits, task inspections, policy checks) and **objects touched** (created, status-changed, or explicitly re-evaluated). There are **no LLM/tool calls** on this path.

## Baseline

`repair="naive"` in `RewindRuntime.ingest_artifact`: inspect and mark every existing artifact, observation, claim, action, and task; then apply the same overlap rule, create the verify task, run `outage_explains_pct`, and create the follow-up task. **Does not** use reverse-dependency lookup to skip a subtree.

## Method

1. Same generative fixture (`build_rewind_fixture(seed=42)`).
2. Same phase-1 ingest (`ingest_and_propose`).
3. Reset the work counter; time only the provider ingest + repair.
4. Targeted: `repair="targeted"` (existing cone invalidation + selective task schedule).
5. Naive: `repair="naive"` (full scan as above).
6. Conclusion checks: `support_pct`, rollback `blocked`, `task-followup-cache` exists.

Code: `hiveclaw_causal/benchmark.py`, `RewindRuntime._repair_targeted` / `_repair_naive`.

## Metrics (run 1 / run 2)

| metric | targeted | naive |
|--------|----------|-------|
| objects_before | 12 | 12 |
| objects_after | 19 | 19 |
| objects_touched | **9** | **19** |
| objects_untouched | **10** | **0** |
| eval_steps | **7** | **19** |
| wall_s (run 1) | 0.007147 | 0.006705 |
| wall_s (run 2) | 0.007333 | 0.006677 |
| support_pct | 92.0 | 92.0 |
| rollback_blocked | True | True |
| followup_present | True | True |

Targeted objects left untouched (10): `art-deploy`, `art-goal`, `art-health-note`, `art-incident-log`, `art-repo-checkout`, `obs-deploy-before-incident`, `obs-timeouts`, `obs-unrelated-health`, `task-provider-crosscheck`, `task-unrelated-docs`.

Naive left none untouched.

Wall-clock on this fixture: naive was **slightly faster** both runs (~0.6–0.7 ms). That difference is smaller than typical OS noise on a ~7 ms SQLite repair. **Do not treat wall-clock as a win for targeted repair at this scale.**

## Outcome

- **Correctness:** both paths agree (92.0% outage support, rollback blocked, follow-up task present).
- **Work:** targeted touched **9 of 19** objects vs naive **19 of 19**; **7 vs 19** eval steps. That is a real, small-N gap (12 skipped inspections / 10 skipped objects), not a latency story.
- **Fixture size:** 12 objects before injection. Too small to claim production cost savings or “revolutionary” efficiency. It is large enough to show the skip list is non-empty and that the naive path actually walks everything.

## Decision

**Keep** the targeted reverse-dependency engine as the default repair path (it is the product behavior and uses fewer eval steps).

**Revise** any claim that this slice is faster in wall-clock time — it was not, on this fixture.

**Do not revert.** The naive path stays as a comparison harness (`python -m hiveclaw_causal.benchmark`), not as the runtime default.

## Reproduction

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
python -m hiveclaw_causal.benchmark
python tests/test_hiveclaw_causal_benchmark.py
```

## Reconciliation note (2026-09-01)

Re-run on HEAD `ff4b146e65f10407d3b3552ce3a9a2328faf1afa` (`python -m hiveclaw_causal.benchmark`, seed 42, twice) produced targeted **eval_steps = 6** (naive still 19). Touched 9 vs 19, untouched 10 vs 0, support_pct 92.0, rollback blocked, follow-up present — unchanged. Wall-clock was ~2.8–2.9 ms targeted vs ~3.2–3.3 ms naive (not the ~7 ms in the table above).

Original numbers in this file reflect the codebase at time of writing (HEAD after `4fe3c6d` / Session 3, **prior to** cone-indexed tasks `f9874a6` and the topic-key claim index `4a47330`). Session 5 already recorded N=12 targeted eval_steps dropping 7 → 6; this file’s table was not backfilled. Structural conclusion (bounded vs linear at this N is a skip-list, not a latency win) is unchanged.
