//! Phase C acceptance tests (claim CAS, inhibit ABI, stale-lock watchdog).
//!
//! These are implemented in `tests/ipc_test.rs` as `ipc_macos::phase_c_*` because every
//! test must share the single Mach service label `com.hiveclaw.pheromoned` and a process-wide
//! mutex. Running a second integration test binary would race on launchd bootstrap.
//!
//! Run: `cargo test -p hiveclaw-daemon --test ipc_test phase_c_ -- --test-threads=1`

#[test]
fn phase_c_suite_is_ipc_macos_module() {
    assert!(true);
}
