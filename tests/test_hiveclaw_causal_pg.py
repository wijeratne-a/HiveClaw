#!/usr/bin/env python3
"""Rewind against Postgres over TCP. Skips unless HIVECLAW_PG_DSN is set.

This is not discovered as a required CI service. Default `make test-causal`
skips the class. Session 7's runner sets the DSN and executes these tests.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.benchmark import measure_repair  # noqa: E402
from hiveclaw_causal.lease import (  # noqa: E402
    drain_pending_tasks,
    hold_lease_renew_until_fail,
    lease_one_and_die,
    mp_drain,
    work_slow_with_renew,
)
from hiveclaw_causal.netproxy import TcpProxy  # noqa: E402
from hiveclaw_causal.pg_store import (  # noqa: E402
    PgStore,
    locator_json,
    new_schema_name,
    require_psycopg,
)
from hiveclaw_causal.store import open_store  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ObjectKind,
    ObjectStatus,
    Provenance,
    Record,
    SourceRef,
    TaskStatus,
    TrustClass,
)
from hiveclaw_causal.util import content_hash  # noqa: E402

_DSN = os.environ.get("HIVECLAW_PG_DSN", "").strip()
_PG_REASON = "HIVECLAW_PG_DSN not set (Postgres networked tests are opt-in)"


def _pg_available() -> bool:
    if not _DSN:
        return False
    try:
        require_psycopg()
    except RuntimeError:
        return False
    return True


def _seed_pending_tasks(locator: str, n: int, start: int = 0) -> list[str]:
    store = open_store(locator)
    ids: list[str] = []
    ts = "2026-08-31T00:00:00Z"
    try:
        for i in range(start, start + n):
            tid = f"task-lease-{i:04d}"
            payload = {"kind": "synthetic", "i": i}
            rec = Record(
                id=tid,
                kind=ObjectKind.TASK,
                status=TaskStatus.PENDING.value,
                provenance=Provenance(
                    producer="test",
                    sources=(
                        SourceRef(
                            uri=f"test://task/{i}",
                            version="1",
                            content_hash=content_hash(payload),
                        ),
                    ),
                    timestamp=ts,
                    trust=TrustClass.DETERMINISTIC,
                ),
                payload=payload,
                evidence_ids=(),
                source_snapshot=content_hash(payload),
                created_at=ts,
                updated_at=ts,
            )
            store.put_object(rec, ts=ts, event_type="seed_task", reason="lease stress seed")
            ids.append(tid)
    finally:
        store.close()
    return ids


def _proxy_dsn(dsn: str, port: int) -> str:
    u = urlparse(dsn)
    user = u.username or "hiveclaw"
    password = u.password or "hiveclaw"
    path = u.path or "/hiveclaw"
    return f"postgresql://{user}:{password}@127.0.0.1:{port}{path}?sslmode=disable"


@unittest.skipUnless(_pg_available(), _PG_REASON)
class TestPostgresNetworkedRewind(unittest.TestCase):
    def setUp(self) -> None:
        self.dsn = _DSN
        self.schema = new_schema_name("t")
        self.store = PgStore(self.dsn, schema=self.schema)
        self.locator = locator_json(self.dsn, self.schema)

    def tearDown(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass

    def test_events_append_only(self) -> None:
        rec = Record(
            id="obj-seed",
            kind=ObjectKind.OBSERVATION,
            status=ObjectStatus.ACTIVE.value,
            provenance=Provenance(
                producer="test",
                sources=(
                    SourceRef(
                        uri="test://seed",
                        version="1",
                        content_hash=content_hash({"k": "v"}),
                    ),
                ),
                timestamp="2026-08-31T00:00:00Z",
                trust=TrustClass.DETERMINISTIC,
            ),
            payload={"k": "v"},
            evidence_ids=(),
            source_snapshot=content_hash("seed"),
            created_at="2026-08-31T00:00:00Z",
            updated_at="2026-08-31T00:00:00Z",
        )
        self.store.put_object(rec, ts="2026-08-31T00:00:00Z", event_type="seed", reason="seed")
        with self.assertRaises(Exception) as ctx:
            self.store._conn.execute("UPDATE events SET reason = 'mutated' WHERE seq = 1")
        self.assertIn("append-only", str(ctx.exception).lower())
        with self.assertRaises(Exception) as ctx2:
            self.store._conn.execute("DELETE FROM events WHERE seq = 1")
        self.assertIn("append-only", str(ctx2.exception).lower())

    def test_targeted_eval_steps_flat_at_n500(self) -> None:
        """Same fixture as exp-002 N=500 (U=162,R=2) over TCP Postgres."""
        t_store = PgStore(self.dsn, schema=new_schema_name("tn500"))
        n_store = PgStore(self.dsn, schema=new_schema_name("nn500"))
        try:
            _rt_t, targeted = measure_repair(
                "pg:tn500",
                mode="targeted",
                seed=42,
                extra_unrelated=162,
                extra_related_claims=2,
                store=t_store,
            )
            _rt_n, naive = measure_repair(
                "pg:nn500",
                mode="naive",
                seed=42,
                extra_unrelated=162,
                extra_related_claims=2,
                store=n_store,
            )
            self.assertEqual(targeted.objects_before, 500)
            # Exact SQLite integers (10/511) need not transfer; bounded vs linear must.
            self.assertLessEqual(targeted.eval_steps, 12)
            self.assertGreaterEqual(naive.eval_steps, 500)
            self.assertLess(targeted.eval_steps, naive.eval_steps)
            self.assertAlmostEqual(targeted.support_pct, 92.0, places=1)
            self.assertTrue(targeted.rollback_blocked)
            self.assertTrue(naive.rollback_blocked)
        finally:
            t_store.close()
            n_store.close()

    def test_targeted_eval_steps_flat_at_c500_claims(self) -> None:
        t_store = PgStore(self.dsn, schema=new_schema_name("tc500"))
        n_store = PgStore(self.dsn, schema=new_schema_name("nc500"))
        try:
            _rt_t, targeted = measure_repair(
                "pg:tc500",
                mode="targeted",
                seed=42,
                extra_unrelated_claims=500,
                store=t_store,
            )
            _rt_n, naive = measure_repair(
                "pg:nc500",
                mode="naive",
                seed=42,
                extra_unrelated_claims=500,
                store=n_store,
            )
            self.assertEqual(targeted.objects_before, 512)
            self.assertLessEqual(targeted.eval_steps, 8)
            self.assertGreaterEqual(naive.eval_steps, 500)
            self.assertLess(targeted.eval_steps, naive.eval_steps)
            self.assertAlmostEqual(targeted.support_pct, 92.0, places=1)
        finally:
            t_store.close()
            n_store.close()

    def test_more_workers_than_tasks_no_double_lease(self) -> None:
        double = 0
        dropped = 0
        ctx = multiprocessing.get_context("spawn")
        for trial in range(8):
            schema = new_schema_name(f"st{trial}")
            PgStore(self.dsn, schema=schema).close()
            loc = locator_json(self.dsn, schema)
            expected = _seed_pending_tasks(loc, 3)
            jobs = [(loc, f"worker-{i}", 0.004) for i in range(5)]
            with ctx.Pool(processes=5) as pool:
                results = list(pool.map(mp_drain, jobs))
            leased_all = [tid for _w, ids in results for tid in ids]
            counts = Counter(leased_all)
            if any(c != 1 for c in counts.values()):
                double += 1
            if sorted(leased_all) != sorted(expected):
                dropped += 1
            store = open_store(loc)
            try:
                tasks = {t.id: t for t in store.objects_of(ObjectKind.TASK)}
                for tid in expected:
                    self.assertEqual(tasks[tid].status, TaskStatus.DONE.value)
            finally:
                store.close()
        self.assertEqual(double, 0, "double-lease trials")
        self.assertEqual(dropped, 0, "dropped-task trials")

    def test_killed_worker_lease_is_reclaimed(self) -> None:
        expected = _seed_pending_tasks(self.locator, 1)
        tid = expected[0]
        ttl_s = 0.25
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=lease_one_and_die,
            args=((self.locator, "crasher", ttl_s),),
        )
        p.start()
        deadline = time.time() + 8.0
        saw = False
        while time.time() < deadline:
            rec = self.store.get(tid)
            if rec.status == TaskStatus.LEASED.value and rec.payload.get("lease_owner") == "crasher":
                saw = True
                break
            time.sleep(0.01)
        p.join(timeout=3)
        self.assertTrue(saw, "crasher never committed a lease")
        self.assertNotEqual(p.exitcode, 0)
        time.sleep(ttl_s + 0.15)
        recovered = drain_pending_tasks(
            self.locator, "survivor", pause_s=0.0, lease_ttl_s=30.0
        )
        self.assertEqual(recovered, [tid])
        rec = self.store.get(tid)
        self.assertEqual(rec.status, TaskStatus.DONE.value)
        self.assertEqual(rec.payload.get("completed_by"), "survivor")
        self.assertEqual(rec.payload.get("reclaimed_from"), "crasher")

    def test_slow_alive_worker_that_renews_is_not_reclaimed(self) -> None:
        expected = _seed_pending_tasks(self.locator, 1)
        tid = expected[0]
        ttl_s = 0.2
        work_s = 0.7
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=work_slow_with_renew,
            args=((self.locator, "slow", ttl_s, work_s, 0.05),),
        )
        p.start()
        deadline = time.time() + 8.0
        saw = False
        while time.time() < deadline:
            rec = self.store.get(tid)
            if rec.status == TaskStatus.LEASED.value and rec.payload.get("lease_owner") == "slow":
                saw = True
                break
            time.sleep(0.01)
        self.assertTrue(saw, "slow worker never leased")
        time.sleep(ttl_s + 0.2)
        stolen = self.store.try_lease_one_task("poacher", "2026-08-31T00:00:01Z", lease_ttl_s=30.0)
        self.assertIsNone(stolen, "slow-but-alive lease was reclaimed")
        p.join(timeout=5)
        self.assertEqual(p.exitcode, 0)
        rec = self.store.get(tid)
        self.assertEqual(rec.status, TaskStatus.DONE.value)
        self.assertEqual(rec.payload.get("completed_by"), "slow")
        self.assertIsNone(rec.payload.get("reclaimed_from"))

    def test_tcp_drop_mid_lease_is_reclaimed_without_process_death(self) -> None:
        """Network path fails; worker process stays up. TTL reclaim must still fire."""
        u = urlparse(self.dsn)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 5432)
        proxy = TcpProxy(host, port)
        listen = proxy.start()
        pdsn = _proxy_dsn(self.dsn, listen)
        schema = new_schema_name("drop")
        PgStore(pdsn, schema=schema).close()
        loc = locator_json(pdsn, schema)
        expected = _seed_pending_tasks(loc, 1)
        tid = expected[0]
        ttl_s = 0.35
        td = tempfile.TemporaryDirectory()
        ready = Path(td.name) / "ready"
        fail = Path(td.name) / "fail"
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=hold_lease_renew_until_fail,
            args=((loc, "cut", ttl_s, str(ready), str(fail)),),
        )
        p.start()
        deadline = time.time() + 8.0
        while time.time() < deadline and not ready.exists():
            time.sleep(0.01)
        self.assertTrue(ready.exists(), "worker never leased through proxy")
        self.assertTrue(p.is_alive(), "worker died before the drop — would be SIGKILL, not TCP fail")
        proxy.drop_all()
        deadline = time.time() + 8.0
        while time.time() < deadline and not fail.exists():
            time.sleep(0.02)
        self.assertTrue(fail.exists(), "renew after drop did not fail")
        self.assertTrue(p.is_alive(), "worker process died; this must stay a network-only failure")
        time.sleep(ttl_s + 0.2)
        # New TCP path, not the dropped sockets.
        survivor_loc = locator_json(self.dsn, schema)
        recovered = drain_pending_tasks(
            survivor_loc, "survivor", pause_s=0.0, lease_ttl_s=30.0
        )
        self.assertEqual(recovered, [tid])
        store = open_store(survivor_loc)
        try:
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.DONE.value)
            self.assertEqual(rec.payload.get("completed_by"), "survivor")
            self.assertEqual(rec.payload.get("reclaimed_from"), "cut")
        finally:
            store.close()
        self.assertTrue(p.is_alive(), "worker still up after reclaim")
        p.terminate()
        p.join(timeout=3)
        proxy.close()
        td.cleanup()

    def test_proxy_stall_longer_than_ttl_false_reclaims_live_worker(self) -> None:
        """Heartbeat delayed by a stall (not death) is treated as silence."""
        u = urlparse(self.dsn)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 5432)
        proxy = TcpProxy(host, port)
        listen = proxy.start()
        pdsn = _proxy_dsn(self.dsn, listen)
        schema = new_schema_name("stall")
        PgStore(pdsn, schema=schema).close()
        loc = locator_json(pdsn, schema)
        expected = _seed_pending_tasks(loc, 1)
        tid = expected[0]
        ttl_s = 0.2
        work_s = 1.2
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=work_slow_with_renew,
            args=((loc, "slow", ttl_s, work_s, 0.05),),
        )
        p.start()
        deadline = time.time() + 8.0
        saw = False
        watch = open_store(locator_json(self.dsn, schema))
        try:
            while time.time() < deadline:
                rec = watch.get(tid)
                if rec.status == TaskStatus.LEASED.value and rec.payload.get("lease_owner") == "slow":
                    saw = True
                    break
                time.sleep(0.01)
            self.assertTrue(saw, "slow worker never leased through proxy")
            proxy.stall(0.45)
            stolen = watch.try_lease_one_task("poacher", "2026-08-31T00:00:01Z", lease_ttl_s=30.0)
            self.assertIsNotNone(stolen, "expected false-reclaim after stall > TTL")
            assert stolen is not None
            self.assertEqual(stolen.id, tid)
            self.assertEqual(stolen.payload.get("reclaimed_from"), "slow")
        finally:
            watch.close()
        p.join(timeout=5)
        proxy.close()


if __name__ == "__main__":
    unittest.main()
