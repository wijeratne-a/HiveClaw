//! `pheromoned` — IOSurface slab + XPC Mach service (Phase 4).

#[cfg(target_os = "macos")]
fn main() {
    use clap::Parser;
    use hiveclaw_core::math::{SlabLayout, DEFAULT_LATENT_ELEMS, N_SLOTS};

    #[derive(Parser, Debug)]
    #[command(name = "pheromoned")]
    struct Args {
        /// bf16 latent width D per slot (stride = 128 + 2*D).
        #[arg(long, default_value_t = DEFAULT_LATENT_ELEMS)]
        latent_dim: u32,
    }

    let args = Args::parse();
    let layout = match SlabLayout::try_from_latent_elems(args.latent_dim, N_SLOTS as u32) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Invalid slab layout: {e}");
            std::process::exit(1);
        }
    };
    hiveclaw_daemon::xpc::run_pheromoned_daemon(layout);
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("pheromoned is only supported on macOS");
    std::process::exit(1);
}
