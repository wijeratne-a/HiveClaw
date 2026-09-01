# exp-002 — targeted vs naive at scaled fixture sizes

**Date:** 2026-08-31  
**Prerequisite:** exp-001 at N=12; CI green at
https://github.com/wijeratne-a/HiveClaw/actions/runs/33478599893 (`368b330`, job ~12s).  
**Command:** `measure_repair(..., extra_unrelated=U, extra_related_claims=R)` seed 42, two runs per scale.  
**Interpreter:** `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` 3.11.1.

## Hypothesis

At larger N, targeted reverse-dep repair still reaches the same conclusion as a naive full scan, uses fewer **eval steps** and **objects touched**, and **may** become faster in wall-clock once SQLite scan cost of the naive path exceeds the targeted cone. If wall-clock stays tied even at ~500 objects, SQLite/Python overhead dominates and eval-step savings do not imply latency savings.

## Baseline

Same naive path as exp-001 (`repair="naive"`): inspect and touch every object, then apply overlap + verify + follow-up.

## Method

1. Base Rewind graph is 12 objects (`U=0,R=0`).
2. Each extra unrelated unit adds 1 artifact + 1 observation + 1 task **outside** the cache/provider cone (`U`).
3. Extra related claims (`R`) `depends_on` `claim-cache-regression` so they sit in the invalidation cone.
4. Scales: `(U,R) = (0,0) → 12`, `(29,1) → 100`, `(162,2) → 500` objects before injection.
5. Time only provider ingest + repair. Two runs per scale.

## Metrics

| scale | U,R | before | after | touched t/n | untouched t/n | eval t/n | wall_s run1 t/n | wall_s run2 t/n | support_pct | rollback blocked |
|-------|-----|--------|-------|-------------|---------------|----------|-----------------|-----------------|-------------|------------------|
| ~12 | 0,0 | 12 | 19 | 9 / 19 | 10 / 0 | 7 / 19 | 0.007134 / 0.007175 | 0.006982 / 0.006677 | 92.0 / 92.0 | True |
| ~100 | 29,1 | 100 | 107 | 10 / 107 | 97 / 0 | 38 / 109 | 0.008567 / 0.009886 | 0.007753 / 0.010071 | 92.0 / 92.0 | True |
| ~500 | 162,2 | 500 | 507 | 11 / 507 | 496 / 0 | 173 / 511 | 0.012144 / 0.019063 | 0.013902 / 0.019323 | 92.0 / 92.0 | True |

Touched/eval/untouched were identical across the two runs at each scale. Wall-clock varied.

## Outcome

- **Correctness:** both paths still agree at every scale (92.0%, rollback blocked, follow-up present).
- **Objects touched:** targeted stays ~9–11; naive equals the full post-repair graph (19 / 107 / 507). Untouched unrelated objects: 10 → 97 → 496.
- **Eval steps:** naive ≈ objects_after. Targeted grows with **task count** because `_after_provider` still **inspects every task** to decide skip vs schedule (7 → 38 → 173), even though it does not *touch* unrelated tasks.
- **Wall-clock:** at N=12, runs overlap (naive sometimes faster). At N=100, targeted was ~1.2–1.3× faster (8.6 vs 9.9 ms; 7.8 vs 10.1 ms). At N=500, targeted was ~1.4–1.6× faster (12.1 vs 19.1 ms; 13.9 vs 19.3 ms). Absolute times remain **tens of milliseconds**. This is a measurable gap, not a production SLA result.
- **Where cost lives:** naive wall-clock tracks scanning/touching every row. Targeted wall-clock still grows (task inspections + SQLite), just slower. SQLite overhead is visible; it does **not** fully hide the eval-step gap at N=500.

## Decision

**Keep** targeted repair as the default.

**Keep** reporting objects-touched and eval-steps as the primary efficiency metrics; treat wall-clock as secondary and scale-dependent.

**Revise** the exp-001 statement that “no wall-clock win” is the last word — it was true at N=12 and is **not** true at N=100 or N=500 on this machine.

**Do not revert.** Do not claim order-of-magnitude latency wins; the N=500 gap is ~6–7 ms.

## Reproduction

```bash
python -m hiveclaw_causal.benchmark
python -m hiveclaw_causal.benchmark --extra-unrelated 29 --extra-related 1
python -m hiveclaw_causal.benchmark --extra-unrelated 162 --extra-related 2
```
