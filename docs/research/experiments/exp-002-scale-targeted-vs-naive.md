# exp-002 — targeted vs naive at scaled fixture sizes

**Date:** 2026-09-01 (Session 5 re-run)  
**Earlier table:** Session 4, 2026-08-31, same scales without cone-indexed task lookup.  
**Command:** `measure_repair(..., extra_unrelated=U, extra_related_claims=R)` seed 42, two runs per scale.  
**Interpreter:** `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` 3.11.1.

## Hypothesis

At larger N, targeted reverse-dep repair still reaches the same conclusion as a naive full scan, uses fewer **eval steps** and **objects touched**, and **may** become faster in wall-clock once SQLite scan cost of the naive path exceeds the targeted cone.

Session 4 found targeted **eval_steps still grew with task count** because `_after_provider` inspected every task to decide skip vs schedule. Session 5 asks whether `reverse_deps` can avoid that inspection entirely.

## What Session 5 changed (not a new index)

Tasks were not in `reverse_deps`. `_put_task` now writes a `depends_on` edge from the task to `payload.target_id`. Repair then lists candidate tasks with `dependent_tasks(cone_id)` (SQL join on `reverse_deps`) instead of `objects_of(TASK)`.

Invalidation **does not** flip task rows to stale when a claim is challenged (`InvalidationEngine` skips `ObjectKind.TASK`). The edge exists for lookup, not for status fan-out. That is use of the existing index, not a redesign.

Unrelated tasks are **not listed** in `RevalidationReport.skipped`. Skipped now means “considered in the cone and not rerun” (e.g. `task-provider-crosscheck`), not “we walked the whole queue and wrote a skip reason.”

## Baseline (Session 4, inspect-all-tasks)

| scale | U,R | eval t/n | wall_s run1 t/n | wall_s run2 t/n |
|-------|-----|----------|-----------------|-----------------|
| ~12 | 0,0 | 7 / 19 | 0.007134 / 0.007175 | 0.006982 / 0.006677 |
| ~100 | 29,1 | 38 / 109 | 0.008567 / 0.009886 | 0.007753 / 0.010071 |
| ~500 | 162,2 | 173 / 511 | 0.012144 / 0.019063 | 0.013902 / 0.019323 |

Targeted eval_steps tracked task count (~N/3).

## Method

Same as Session 4, plus N=2000 (`U=662,R=2` → 2000 objects before injection). Time only provider ingest + repair. Two runs per scale.

## Metrics (Session 5, cone-indexed tasks)

Touched/eval/untouched identical across the two runs at each scale. Wall-clock varied.

| scale | U,R | before | after | touched t/n | untouched t | eval t/n | wall_s run1 t/n | wall_s run2 t/n | support_pct | rollback blocked |
|-------|-----|--------|-------|-------------|-------------|----------|-----------------|-----------------|-------------|------------------|
| ~12 | 0,0 | 12 | 19 | 9 / 19 | 10 | **6** / 19 | 0.003131 / 0.003376 | 0.003643 / 0.002916 | 92.0 / 92.0 | True |
| ~100 | 29,1 | 100 | 107 | 10 / 107 | 97 | **8** / 109 | 0.002610 / 0.004298 | 0.002392 / 0.004426 | 92.0 / 92.0 | True |
| ~500 | 162,2 | 500 | 507 | 11 / 507 | 496 | **10** / 511 | 0.003107 / 0.011973 | 0.002590 / 0.011888 | 92.0 / 92.0 | True |
| ~2000 | 662,2 | 2000 | 2007 | 11 / 2007 | 1996 | **10** / 2011 | 0.003178 / 0.041284 | 0.003048 / 0.045704 | 92.0 / 92.0 | True |

N=12 eval_steps dropped 7 → 6 because `task-unrelated-docs` is no longer inspected. Same conclusion (92.0%, rollback blocked, follow-up present).

## Outcome

- **Correctness:** both paths still agree at every scale.
- **Objects touched:** targeted stays 9–11; naive equals the full post-repair graph. Untouched: 10 → 97 → 496 → 1996.
- **Eval steps:** targeted is now **flat with N** for this fixture (6 → 8 → 10 → 10). The +2 from N=12 to N=100 is the extra related claim in the overlap scan, not the 29 extra tasks. Naive remains ≈ objects_after (19 / 109 / 511 / 2011).
- **Does the gap grow faster than linearly?** The **eval-step difference** naive − targeted is ≈ N (13, 101, 501, 2001). That is **linear in N**, which is the best this comparison can do once targeted is O(cone) and naive is O(N). It is not superlinear. The **ratio** naive/targeted grows ~linearly with N (about 3×, 14×, 51×, 201×).
- **Wall-clock:** targeted stays ~3 ms from N=12 through N=2000. Naive ~3 ms → ~4 ms → ~12 ms → ~41–46 ms. At N=500 the gap is ~9 ms (larger than Session 4’s 6–7 ms because targeted got cheaper, not because naive got worse). At N=2000 the gap is ~38–43 ms. Absolute times are still **tens of milliseconds** on this machine.
- **Remaining eval-step limiter (Session 5):** `apply_provider_overlap_rule` still inspected **every claim**. Extra unrelated units in the task-scale fixture are artifact/observation/task triples, not claims.

## Session 6 — unrelated claims (topic index, not observation cone)

**Date:** 2026-09-02  
**Structural note:** Claims cannot be found the same way tasks are. A new `provider_outage` observation has **no reverse_deps** to existing claims until the overlap rule fires a `contradicts` edge. `dependent_claims(obs.id)` would be empty. Provider-interested claims are instead `depends_on` a stable topic key `topic-provider-status` at create time; overlap uses `dependent_claims(topic)`. Same index, different key. Unrelated claims (no provider mention) are not indexed and not inspected.

**Command:** `measure_repair(..., extra_unrelated_claims=C)` seed 42.

| C | before | after | touched t/n | eval t/n | wall_s run1 t/n | wall_s run2 t/n | support_pct | rollback |
|---|--------|-------|-------------|----------|-----------------|-----------------|-------------|----------|
| 0 | 12 | 19 | 9 / 19 | **6** / 19 | 0.002965 / 0.003016 | 0.002960 / 0.002954 | 92.0 | True |
| 500 | 512 | 519 | 9 / 519 | **6** / 519 | 0.003266 / 0.010017 | 0.003017 / 0.009392 | 92.0 | True |
| 2000 | 2012 | 2019 | 9 / 2019 | **6** / 2019 | 0.003816 / 0.034676 | 0.053339 / 0.030007 | 92.0 | True |

Eval-steps and touched were identical across runs. Run 2 targeted wall at C=2000 (53 ms) is an outlier; three follow-up runs were 2.63 / 2.71 / 2.61 ms targeted vs 31.7 / 29.9 / 34.8 ms naive. Treat ~3 ms vs ~30–35 ms as the typical C=2000 pair.

Flat targeted eval_steps (6) vs linear naive (19 → 519 → 2019) holds on **unrelated claims**, not only tasks. This is a bounded-cost coordination property on this fixture, single-host SQLite, not a general production latency claim.

## Decision

**Keep** cone-indexed task lookup and the provider-topic claim index. Task-only O(N) was Session 5; claim-side O(N) is closed the same way on a **topic key**, not the observation cone.

**Keep** objects-touched and eval-steps as primary efficiency metrics. Wall-clock now tracks naive scan cost; targeted wall-clock is dominated by a small constant cone plus SQLite.

**Do not** claim superlinear eval-step savings, production latency wins, or that the property is “revolutionary.”

## Reproduction

```bash
python -m hiveclaw_causal.benchmark
python -m hiveclaw_causal.benchmark --extra-unrelated 29 --extra-related 1
python -m hiveclaw_causal.benchmark --extra-unrelated 162 --extra-related 2
python -m hiveclaw_causal.benchmark --extra-unrelated 662 --extra-related 2
python -m hiveclaw_causal.benchmark --extra-unrelated-claims 500
python -m hiveclaw_causal.benchmark --extra-unrelated-claims 2000
```
