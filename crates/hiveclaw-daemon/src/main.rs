//! `pheromoned` — IOSurface slab + XPC Mach service (Phase 4).

#[cfg(target_os = "macos")]
fn main() {
    hiveclaw_daemon::xpc::run_pheromoned_daemon();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("pheromoned is only supported on macOS");
    std::process::exit(1);
}
