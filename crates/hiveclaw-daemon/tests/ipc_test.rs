//! XPC + IOSurface integration test (Phase 4).

#[cfg(not(target_os = "macos"))]
#[test]
fn ipc_iosurface_skipped_on_non_macos() {}

#[cfg(target_os = "macos")]
mod ipc_macos {
    use half::bf16;
    use hiveclaw_backend_metal::MetalPheromoneBuffer;
    use hiveclaw_core::math::{SLOT0_SCALAR_BYTE_OFFSET, SLOT0_SCENT_BYTE_OFFSET};
    use hiveclaw_daemon::xpc::fetch_surface_id;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::ptr;
    use std::sync::atomic::{fence, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

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
            // Prefer bootout by plist path (matches `bootstrap` target on recent macOS).
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

    fn wait_for_xpc() -> u32 {
        let start = Instant::now();
        for _ in 0..POLL_MAX {
            if let Ok(id) = fetch_surface_id() {
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
    fn xpc_iosurface_bitwise_roundtrip() {
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

        let zeros: [f32; 4] = unsafe {
            let p = base.add(SLOT0_SCALAR_BYTE_OFFSET) as *const f32;
            [p.read(), p.add(1).read(), p.add(2).read(), p.add(3).read()]
        };
        assert_eq!(zeros.map(f32::to_bits), [0f32; 4].map(f32::to_bits));

        let scalars = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            ptr::copy_nonoverlapping(
                scalars.as_ptr(),
                base.add(SLOT0_SCALAR_BYTE_OFFSET) as *mut f32,
                4,
            );
        }
        fence(Ordering::SeqCst);
        let read_back: [f32; 4] = unsafe {
            let p = base.add(SLOT0_SCALAR_BYTE_OFFSET) as *const f32;
            [p.read(), p.add(1).read(), p.add(2).read(), p.add(3).read()]
        };
        assert_eq!(read_back[0].to_bits(), scalars[0].to_bits());
        assert_eq!(read_back[1].to_bits(), scalars[1].to_bits());
        assert_eq!(read_back[2].to_bits(), scalars[2].to_bits());
        assert_eq!(read_back[3].to_bits(), scalars[3].to_bits());

        let a = bf16::from_f32(0.5);
        let b = bf16::from_f32(-0.5);
        unsafe {
            let dst = base.add(SLOT0_SCENT_BYTE_OFFSET) as *mut bf16;
            dst.write(a);
            dst.add(1).write(b);
        }
        fence(Ordering::SeqCst);
        let ga = unsafe { (base.add(SLOT0_SCENT_BYTE_OFFSET) as *const bf16).read() };
        let gb = unsafe {
            (base.add(SLOT0_SCENT_BYTE_OFFSET) as *const bf16)
                .add(1)
                .read()
        };
        assert_eq!(ga.to_bits(), a.to_bits());
        assert_eq!(gb.to_bits(), b.to_bits());
    }
}
