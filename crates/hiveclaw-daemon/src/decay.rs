//! Background CPU decay loop for Phase C slab slots (daemon only).

use hiveclaw_core::math::{
    OFF_G_DECAY_RATE, OFF_G_ZETA_T, OFF_S_LAST_CLAIM_MACH, OFF_S_SLOT_STATE, SlabLayout,
    SLOT_STATUS_CLAIMED, SLOT_STATUS_FREE, STALE_LOCK_MS, slot_status,
};
use half::bf16;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

fn stale_threshold_mach_ticks() -> u64 {
    static THRESH: OnceLock<u64> = OnceLock::new();
    *THRESH.get_or_init(|| {
        #[cfg(target_os = "macos")]
        unsafe {
            let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
            if mach_timebase_info(&mut info) != 0 || info.numer == 0 || info.denom == 0 {
                return 0;
            }
            let ns = STALE_LOCK_MS.saturating_mul(1_000_000);
            ns.saturating_mul(info.denom as u64) / info.numer as u64
        }
        #[cfg(not(target_os = "macos"))]
        0u64
    })
}

/// Start a detached thread that periodically advances `global_zeta_t`, decays unclaimed slot
/// payloads, and force-evicts stale held slots (Mach time).
pub fn start_decay_loop(base: *mut u8, tick_ms: u64, layout: SlabLayout) {
    let base_usize = base as usize;
    let layout = Arc::new(layout);
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(tick_ms));
        unsafe {
            decay_tick(base_usize as *mut u8, &layout);
        }
    });
}

unsafe fn decay_tick(base: *mut u8, layout: &SlabLayout) {
    let zeta_ptr = (base as usize + OFF_G_ZETA_T) as *mut f32;
    let zeta = zeta_ptr.read_volatile();
    let decay_rate = (base as usize + OFF_G_DECAY_RATE) as *const f32;
    let alpha = decay_rate.read_volatile().clamp(0.0, 1.0);
    let new_zeta = zeta + (1.0 - alpha).ln();
    zeta_ptr.write_volatile(new_zeta);

    let scale = (new_zeta - zeta).exp();

    let threshold = stale_threshold_mach_ticks();
    let now = mach_now();
    let latent = layout.latent_elems as usize;
    let n = layout.n_slots as usize;

    for i in 0..n {
        let hdr_base = base as usize + layout.slot_base(i);
        let state_ptr = (hdr_base + OFF_S_SLOT_STATE) as *const AtomicU32;
        let word = (*state_ptr).load(Ordering::Acquire);
        let st = slot_status(word);

        if st == SLOT_STATUS_CLAIMED {
            let mach_ptr = (hdr_base + OFF_S_LAST_CLAIM_MACH) as *const u64;
            let last = mach_ptr.read_unaligned();
            if threshold > 0 && now >= last && now - last >= threshold {
                force_evict_slot(base, layout, i);
            }
            continue;
        }

        if st == SLOT_STATUS_FREE {
            decay_slot_payload(base, layout.slot_payload(i), latent, scale);
        }
    }
}

#[cfg(target_os = "macos")]
fn mach_now() -> u64 {
    unsafe { mach_absolute_time() }
}

#[cfg(not(target_os = "macos"))]
fn mach_now() -> u64 {
    0
}

unsafe fn force_evict_slot(base: *mut u8, layout: &SlabLayout, slot_index: usize) {
    let stride = layout.stride as usize;
    let hdr = base.add(layout.slot_base(slot_index));
    std::ptr::write_bytes(hdr, 0, stride);
}

unsafe fn decay_slot_payload(base: *mut u8, payload_off: usize, latent: usize, scale: f32) {
    let payload = (base as usize + payload_off) as *mut bf16;
    for j in 0..latent {
        let v = payload.add(j).read();
        let f = v.to_f32() * scale;
        payload.add(j).write(bf16::from_f32(f));
    }
}
