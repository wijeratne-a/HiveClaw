//! Background CPU decay loop for Phase C slab slots (daemon only).

use hiveclaw_core::math::{
    N_SLOTS, OFF_G_DECAY_RATE, OFF_G_ZETA_T, OFF_S_CLAIM_FLAG, OFF_S_LAST_WRITE_CLK,
    OFF_S_LAST_INHIBIT_CLK, OFF_S_WATCHDOG_FLAGS, SCENT_ELEMS, STALE_LOCK_ZETA_DELTA, slot_base,
    slot_payload,
};
use half::bf16;
use std::sync::atomic::{AtomicU32, Ordering};

/// Start a detached thread that periodically advances `global_zeta_t`, decays unclaimed slot
/// payloads, and force-evicts stale locks.
pub fn start_decay_loop(base: *mut u8, tick_ms: u64) {
    let base_usize = base as usize;
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(tick_ms));

        // SAFETY: `base_usize` is the leaked IOSurface mapping for the daemon lifetime.
        unsafe {
            decay_tick(base_usize as *mut u8);
        }
    });
}

unsafe fn decay_tick(base: *mut u8) {
    let zeta_ptr = (base as usize + OFF_G_ZETA_T) as *mut f32;
    let zeta = zeta_ptr.read_volatile();
    let decay_rate = (base as usize + OFF_G_DECAY_RATE) as *const f32;
    let alpha = decay_rate.read_volatile().clamp(0.0, 1.0);
    let new_zeta = zeta + (1.0 - alpha).ln();
    zeta_ptr.write_volatile(new_zeta);

    let scale = (new_zeta - zeta).exp();

    for i in 0..N_SLOTS {
        let hdr_base = base as usize + slot_base(i);
        let claim_ptr = (hdr_base + OFF_S_CLAIM_FLAG) as *const AtomicU32;
        if (*claim_ptr).load(Ordering::Acquire) == 1 {
            let clk_ptr = (hdr_base + OFF_S_LAST_WRITE_CLK) as *const f32;
            let clk = clk_ptr.read_volatile();
            // ζ-time since last write: |ζ_now − ζ_at_write| (both in global ζ coordinates).
            if (new_zeta - clk).abs() >= STALE_LOCK_ZETA_DELTA {
                force_evict(base, i, new_zeta);
            }
            continue;
        }

        decay_slot_payload(base, i, scale);
    }
}

unsafe fn force_evict(base: *mut u8, slot_index: usize, zeta: f32) {
    let hdr_base = base as usize + slot_base(slot_index);
    let claim_ptr = (hdr_base + OFF_S_CLAIM_FLAG) as *mut AtomicU32;
    (*claim_ptr).store(0, Ordering::Release);

    let watchdog = (hdr_base + OFF_S_WATCHDOG_FLAGS) as *mut u32;
    watchdog.write_volatile(watchdog.read_volatile() | 0x1);

    let inh = (hdr_base + OFF_S_LAST_INHIBIT_CLK) as *mut f32;
    inh.write_volatile(zeta);

    let payload = (base as usize + slot_payload(slot_index)) as *mut bf16;
    for j in 0..SCENT_ELEMS {
        payload.add(j).write(bf16::from_f32(0.0));
    }
}

unsafe fn decay_slot_payload(base: *mut u8, slot_index: usize, scale: f32) {
    let payload = (base as usize + slot_payload(slot_index)) as *mut bf16;
    for j in 0..SCENT_ELEMS {
        let v = payload.add(j).read();
        let f = v.to_f32() * scale;
        payload.add(j).write(bf16::from_f32(f));
    }
}
