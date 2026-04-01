//! Shared layout constants.

/// IOSurface-backed slab size (bytes) for the Phase 2 POC.
pub const SLAB_SIZE: usize = 4_718_720;

/// Slot 0: scalar probe region (f32×4) — byte offset from slab base.
pub const SLOT0_SCALAR_BYTE_OFFSET: usize = 256;

/// Slot 0: scent vector (bf16×`SLOT0_SCENT_ELEMS`) — byte offset from slab base.
pub const SLOT0_SCENT_BYTE_OFFSET: usize = 384;

/// Number of bf16 elements in slot 0 scent (kept in sync with `SCENT_ELEMS`).
pub const SLOT0_SCENT_ELEMS: usize = 2048;

// ── Phase C layout (concurrent swarm / fluid dynamics) ─────────────────────

/// Magic `'HCLW'` in little-endian `u32`.
pub const SLAB_MAGIC: u32 = 0x4843_4C57;
pub const SLAB_VERSION_C: u32 = 3;
pub const GLOBAL_HDR_BYTES: usize = 128;
pub const SLOT_HDR_BYTES: usize = 64;
pub const SCENT_ELEMS: usize = 2048;
pub const SCENT_BYTES: usize = SCENT_ELEMS * 2; // bf16
pub const SLOT_STRIDE: usize = SLOT_HDR_BYTES + SCENT_BYTES; // 4160
pub const N_SLOTS: usize = 32;
pub const PHASE_C_BYTES: usize = GLOBAL_HDR_BYTES + N_SLOTS * SLOT_STRIDE; // 133_248

/// Byte offsets within GlobalHeader (all u32/f32 = 4B each).
pub const OFF_G_MAGIC: usize = 0;
pub const OFF_G_VERSION: usize = 4;
pub const OFF_G_N_SLOTS: usize = 8;
pub const OFF_G_SLOT_STRIDE: usize = 12;
pub const OFF_G_ZETA_T: usize = 16; // f32 global_zeta_t
pub const OFF_G_DECAY_RATE: usize = 20; // f32 decay_rate

/// Byte offsets within a SlotHeader (relative to slot base).
pub const OFF_S_CLAIM_FLAG: usize = 0; // atomic_uint / u32
pub const OFF_S_OWNER_ID: usize = 4;
pub const OFF_S_LAST_WRITE_CLK: usize = 8; // float
pub const OFF_S_WATCHDOG_FLAGS: usize = 12; // uint32 (bit0 = forced eviction)
pub const OFF_S_LAST_INHIBIT_CLK: usize = 16; // float

/// Watchdog: if `|global_zeta_t - last_write_clock| > this`, force-evict a held slot.
pub const STALE_LOCK_ZETA_DELTA: f32 = 0.35;

#[inline]
pub const fn slot_base(i: usize) -> usize {
    GLOBAL_HDR_BYTES + i * SLOT_STRIDE
}

#[inline]
pub const fn slot_payload(i: usize) -> usize {
    slot_base(i) + SLOT_HDR_BYTES
}
