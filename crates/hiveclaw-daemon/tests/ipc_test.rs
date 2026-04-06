//! XPC + IOSurface integration tests (Phase 4 + Phase C v5).
//!
//! All tests share one `launchd` Mach registration (`com.hiveclaw.pheromoned`) and must not run in
//! parallel with each other (or with other test binaries that bootstrap the same label).

#[cfg(not(target_os = "macos"))]
#[test]
fn ipc_iosurface_skipped_on_non_macos() {}

#[cfg(target_os = "macos")]
mod ipc_macos {
    use half::bf16;
    use hiveclaw_backend_metal::MetalPheromoneBuffer;
    use hiveclaw_core::math::{
        layout_magic_version_u64, SlabLayout, DEFAULT_LATENT_ELEMS, N_SLOTS, OFF_G_VERSION_V5,
        OFF_S_CLAIM_FLAG, OFF_S_LAST_CLAIM_MACH, OFF_S_SLOT_STATE, SLOT_STATUS_CLAIMED,
        SLOT_STATUS_INHIBITED, SLAB_VERSION_V6, pack_slot_claimed, slot_status,
    };
    use hiveclaw_daemon::xpc::fetch_surface_v5;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::ptr;
    use std::sync::atomic::{AtomicU32, fence, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Serialize launchd bootstrap across all tests in this binary (same Mach service label).
    static LAUNCHD_TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn launchd_serial_lock() -> std::sync::MutexGuard<'static, ()> {
        LAUNCHD_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    const LABEL: &str = "com.hiveclaw.pheromoned";
    const POLL_MS: u64 = 100;
    const POLL_MAX: u32 = 200;

    struct LaunchdGuard {
        plist_path: PathBuf,
        domain: String,
    }

    impl LaunchdGuard {
        fn bootstrap(exe: &Path) -> Self {
            let uid = unsafe { libc::getuid() };
            let domain = format!("gui/{uid}");
            let dir = std::env::temp_dir().join(format!("hiveclaw_xpc_{}", std::process::id()));
            fs::create_dir_all(&dir).expect("temp dir");
            let plist_path = dir.join(format!("{LABEL}.plist"));

            let exe_str = exe.to_str().expect("exe path must be UTF-8 for plist");
            let escaped = exe_str.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");

            let plist_body = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{escaped}</string>
    </array>
    <key>MachServices</key>
    <dict>
        <key>{LABEL}</key>
        <true/>
    </dict>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"#
            );

            fs::write(&plist_path, plist_body.as_bytes()).expect("write plist");

            let plist_s = plist_path.to_str().expect("plist path utf-8");
            let _ = Command::new("launchctl")
                .args(["bootout", &domain, plist_s])
                .status();
            let _ = Command::new("launchctl")
                .args(["bootout", &domain, LABEL])
                .status();

            let st = Command::new("launchctl")
                .args(["bootstrap", &domain, plist_s])
                .status()
                .expect("launchctl bootstrap");
            assert!(st.success(), "launchctl bootstrap failed: {st:?}");

            Self { plist_path, domain }
        }
    }

    impl Drop for LaunchdGuard {
        fn drop(&mut self) {
            if let Some(plist_s) = self.plist_path.to_str() {
                let _ = Command::new("launchctl")
                    .args(["bootout", &self.domain, plist_s])
                    .status();
            }
            let _ = Command::new("launchctl")
                .args(["bootout", &self.domain, LABEL])
                .status();
            let _ = fs::remove_file(&self.plist_path);
            if let Some(dir) = self.plist_path.parent() {
                let _ = fs::remove_dir(dir);
            }
        }
    }

    fn default_slab_layout() -> SlabLayout {
        SlabLayout::try_from_latent_elems(DEFAULT_LATENT_ELEMS, N_SLOTS as u32).unwrap()
    }

    fn wait_for_xpc() -> u32 {
        let start = Instant::now();
        for _ in 0..POLL_MAX {
            if let Ok((id, _)) = fetch_surface_v5() {
                return id;
            }
            thread::sleep(Duration::from_millis(POLL_MS));
        }
        panic!(
            "timed out after {:?} waiting for XPC ({POLL_MAX}×{POLL_MS}ms)",
            start.elapsed()
        );
    }

    #[test]
    fn global_header_v6_validation() {
        let _serial = launchd_serial_lock();
        if std::env::var("HIVECLAW_SKIP_LAUNCHD_TEST").as_deref() == Ok("1") {
            eprintln!("SKIP ipc: HIVECLAW_SKIP_LAUNCHD_TEST=1");
            return;
        }

        let exe = PathBuf::from(env!("CARGO_BIN_EXE_pheromoned"));
        let _guard = LaunchdGuard::bootstrap(&exe);
        let id = wait_for_xpc();

        let slab = MetalPheromoneBuffer::from_surface_id(id);
        let base = slab.base_ptr();

        fence(Ordering::SeqCst);

        let magic = unsafe { ptr::read_unaligned(base as *const u64) };
        let ver = unsafe { ptr::read_unaligned(base.add(OFF_G_VERSION_V5) as *const u32) };
        assert_eq!(magic, layout_magic_version_u64());
        assert_eq!(ver, SLAB_VERSION_V6);
    }

    #[test]
    fn xpc_iosurface_bitwise_roundtrip() {
        let _serial = launchd_serial_lock();
        if std::env::var("HIVECLAW_SKIP_LAUNCHD_TEST").as_deref() == Ok("1") {
            eprintln!("SKIP ipc: HIVECLAW_SKIP_LAUNCHD_TEST=1 (launchctl unavailable in this environment)");
            return;
        }

        let exe = PathBuf::from(env!("CARGO_BIN_EXE_pheromoned"));
        let _guard = LaunchdGuard::bootstrap(&exe);
        let id = wait_for_xpc();

        let slab = MetalPheromoneBuffer::from_surface_id(id);
        let base = slab.base_ptr();

        fence(Ordering::SeqCst);

        let lay = default_slab_layout();
        let scent0 = lay.slot_payload(0);
        let a = bf16::from_f32(0.5);
        let b = bf16::from_f32(-0.5);
        unsafe {
            let dst = base.add(scent0) as *mut bf16;
            dst.write(a);
            dst.add(1).write(b);
        }
        fence(Ordering::SeqCst);
        let ga = unsafe { (base.add(scent0) as *const bf16).read() };
        let gb = unsafe {
            (base.add(scent0) as *const bf16)
                .add(1)
                .read()
        };
        assert_eq!(ga.to_bits(), a.to_bits());
        assert_eq!(gb.to_bits(), b.to_bits());
    }

    // ── Phase C ─────────────────────────────────────────────────────────────

    /// Mirrors `InhibitSlab::eval_cpu` (v5: full slot memset + INHIBITED, no separate watchdog u32).
    unsafe fn cpu_inhibit_slot(base: *mut u8, slot: usize, lay: &SlabLayout) {
        assert!(slot < N_SLOTS);
        let b = base as usize;
        let sb = lay.slot_base(slot);
        ptr::write_bytes((b + sb) as *mut u8, 0, lay.stride as usize);
        let st = (b + sb + OFF_S_SLOT_STATE) as *mut AtomicU32;
        (*st).store(SLOT_STATUS_INHIBITED, Ordering::Release);
    }

    #[test]
    fn phase_c_claim_flag_atomic_single_winner() {
        let _serial = launchd_serial_lock();
        if std::env::var("HIVECLAW_SKIP_LAUNCHD_TEST").as_deref() == Ok("1") {
            eprintln!("SKIP phase_c: HIVECLAW_SKIP_LAUNCHD_TEST=1");
            return;
        }

        let exe = PathBuf::from(env!("CARGO_BIN_EXE_pheromoned"));
        let _guard = LaunchdGuard::bootstrap(&exe);
        let id = wait_for_xpc();

        let slab = MetalPheromoneBuffer::from_surface_id(id);
        let base = slab.base_ptr() as usize;
        let slot = 6usize;
        let lay = default_slab_layout();
        let claim_off = base + lay.slot_base(slot) + OFF_S_CLAIM_FLAG;

        let wins = Arc::new(AtomicU32::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(3));

        let mut handles = Vec::new();
        for tid in 0..2 {
            let w = Arc::clone(&wins);
            let b = Arc::clone(&barrier);
            let owner = (tid as u32) + 1;
            handles.push(thread::spawn(move || {
                b.wait();
                let p = claim_off as *mut AtomicU32;
                let desired = pack_slot_claimed(owner);
                unsafe {
                    if (*p)
                        .compare_exchange(0, desired, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        w.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }));
        }

        barrier.wait();
        for h in handles {
            h.join().expect("join");
        }

        assert_eq!(
            wins.load(Ordering::SeqCst),
            1,
            "exactly one contender should win the claim CAS"
        );
        let held = unsafe { (*(claim_off as *const AtomicU32)).load(Ordering::Acquire) };
        assert_eq!(slot_status(held), SLOT_STATUS_CLAIMED);
    }

    #[test]
    fn phase_c_inhibit_correctness_cpu_mirror() {
        let _serial = launchd_serial_lock();
        if std::env::var("HIVECLAW_SKIP_LAUNCHD_TEST").as_deref() == Ok("1") {
            eprintln!("SKIP phase_c: HIVECLAW_SKIP_LAUNCHD_TEST=1");
            return;
        }

        let exe = PathBuf::from(env!("CARGO_BIN_EXE_pheromoned"));
        let _guard = LaunchdGuard::bootstrap(&exe);
        let id = wait_for_xpc();

        let slab = MetalPheromoneBuffer::from_surface_id(id);
        let base = slab.base_ptr();
        let slot = 7usize;
        let lay = default_slab_layout();

        unsafe {
            let sb = lay.slot_base(slot);
            let claim = (base.add(sb + OFF_S_CLAIM_FLAG)) as *mut AtomicU32;
            (*claim).store(pack_slot_claimed(1), Ordering::Release);
            let payload = (base.add(lay.slot_payload(slot))) as *mut bf16;
            payload.write(bf16::from_f32(1.25));

            cpu_inhibit_slot(base, slot, &lay);

            let w = (*claim).load(Ordering::Acquire);
            assert_eq!(slot_status(w), SLOT_STATUS_INHIBITED);
            assert_eq!(payload.read().to_bits(), bf16::from_f32(0.0).to_bits());
        }
    }

    #[test]
    fn phase_c_stale_lock_watchdog_eviction() {
        let _serial = launchd_serial_lock();
        if std::env::var("HIVECLAW_SKIP_LAUNCHD_TEST").as_deref() == Ok("1") {
            eprintln!("SKIP phase_c: HIVECLAW_SKIP_LAUNCHD_TEST=1");
            return;
        }

        let exe = PathBuf::from(env!("CARGO_BIN_EXE_pheromoned"));
        let _guard = LaunchdGuard::bootstrap(&exe);
        let id = wait_for_xpc();

        let slab = MetalPheromoneBuffer::from_surface_id(id);
        let base = slab.base_ptr();
        let slot = 9usize;
        let lay = default_slab_layout();
        let hdr = base as usize + lay.slot_base(slot);

        unsafe {
            let claim_p = (hdr + OFF_S_CLAIM_FLAG) as *mut AtomicU32;
            (*claim_p).store(pack_slot_claimed(42), Ordering::Release);
            ((hdr + OFF_S_LAST_CLAIM_MACH) as *mut u64).write_unaligned(0);
        }

        thread::sleep(Duration::from_millis(500));

        let claim = unsafe {
            let claim_p = (hdr + OFF_S_CLAIM_FLAG) as *const AtomicU32;
            (*claim_p).load(Ordering::Acquire)
        };
        assert_eq!(claim, 0, "daemon should evict stale held slot (Mach time)");
    }
}
