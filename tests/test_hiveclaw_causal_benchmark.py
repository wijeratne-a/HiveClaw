#!/usr/bin/env python3
"""Targeted reverse-dep repair vs naive full re-evaluation (same fixture)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.benchmark import measure_repair  # noqa: E402
from hiveclaw_causal.types import ActionStatus  # noqa: E402


class TestTargetedVsNaive(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_same_conclusion_naive_does_more_work(self) -> None:
        _rt_t, targeted = measure_repair(
            self.dir / "t.sqlite", mode="targeted", seed=42
        )
        _rt_n, naive = measure_repair(self.dir / "n.sqlite", mode="naive", seed=42)
        self.assertAlmostEqual(targeted.support_pct, naive.support_pct, places=6)
        self.assertGreaterEqual(targeted.support_pct, 70.0)
        self.assertTrue(targeted.rollback_blocked)
        self.assertTrue(naive.rollback_blocked)
        self.assertEqual(targeted.rollback_status, ActionStatus.BLOCKED.value)
        self.assertEqual(naive.rollback_status, ActionStatus.BLOCKED.value)
        self.assertTrue(targeted.followup_present)
        self.assertTrue(naive.followup_present)
        self.assertGreaterEqual(naive.eval_steps, targeted.eval_steps)
        self.assertGreaterEqual(naive.objects_touched, targeted.objects_touched)
        self.assertGreater(targeted.objects_untouched, 0)
        self.assertGreaterEqual(naive.objects_touched, naive.objects_before)


if __name__ == "__main__":
    unittest.main()
