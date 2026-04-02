//! Shared layout constants (slab v4).

/// IOSurface-backed slab size (bytes) for the Phase 2 POC.
pub const SLAB_SIZE: usize = 4_718_720;

/// Slot 0: scalar probe region (f32×4) — byte offset from slab base.
pub const SLOT0_SCALAR_BYTE_OFFSET: usize = 256;

/// Slot 0: scent vector (bf16×`SLOT0_SCENT_ELEMS`) — byte offset from slab base.
pub const SLOT0_SCENT_BYTE_OFFSET: usize = 384;

/// Number of bf16 elements in slot 0 scent (kept in sync with `SCENT_ELEMS`).
pub const SLOT0_SCENT_ELEMS: usize = 2048;

// ── Phase C slab v4 (4224-byte slots, dual epochs) ─────────────────────────

/// Magic `'HCLW'` in little-endian `u32`.
pub const SLAB_MAGIC: u32 = 0x4843_4C57;
pub const SLAB_VERSION_V4: u32 = 4;
/// Deprecated alias (v3 removed); use `SLAB_VERSION_V4`.
pub const SLAB_VERSION_C: u32 = SLAB_VERSION_V4;

pub const GLOBAL_HDR_BYTES: usize = 128;
pub const SLOT_HDR_BYTES: usize = 64;
pub const SCENT_ELEMS: usize = 2048;
pub const SCENT_BYTES: usize = SCENT_ELEMS * 2; // bf16
/// Footer: `back_epoch` + 60 bytes reserved (64-byte tail).
pub const SLOT_FOOTER_BYTES: usize = 64;
pub const SLOT_STRIDE: usize = SLOT_HDR_BYTES + SCENT_BYTES + SLOT_FOOTER_BYTES; // 4224
pub const N_SLOTS: usize = 32;
pub const PHASE_C_BYTES: usize = GLOBAL_HDR_BYTES + N_SLOTS * SLOT_STRIDE;

/// Stale held-slot eviction threshold (wall milliseconds), converted to Mach ticks in daemon.
pub const STALE_LOCK_MS: u64 = 500;

/// Packed XPC reply: `(magic as u64) << 32 | version`.
#[inline]
pub const fn layout_magic_version_u64() -> u64 {
    ((SLAB_MAGIC as u64) << 32) | (SLAB_VERSION_V4 as u64)
}

/// Byte offsets within GlobalHeader.
pub const OFF_G_MAGIC: usize = 0;
pub const OFF_G_VERSION: usize = 4;
pub const OFF_G_N_SLOTS: usize = 8;
pub const OFF_G_SLOT_STRIDE: usize = 12;
pub const OFF_G_ZETA_T: usize = 16;
pub const OFF_G_DECAY_RATE: usize = 20;

/// Slot header v4 (relative to slot base).
pub const OFF_S_SLOT_STATE: usize = 0;
pub const OFF_S_LAST_CLAIM_MACH: usize = 4;
pub const OFF_S_FRONT_EPOCH: usize = 12;
pub const OFF_S_WATCHDOG_FLAGS: usize = 16;

/// `back_epoch` u32 at end of scent payload (slot-relative).
pub const OFF_SLOT_BACK_EPOCH: usize = SLOT_HDR_BYTES + SCENT_BYTES; // 4160

pub const SLOT_STATUS_MASK: u32 = 0x3;
pub const SLOT_STATUS_FREE: u32 = 0;
pub const SLOT_STATUS_CLAIMED: u32 = 1;
pub const SLOT_STATUS_INHIBITED: u32 = 2;
pub const SLOT_STATUS_FAULT: u32 = 3;
pub const SLOT_OWNER_SHIFT: u32 = 16;

#[inline]
pub const fn pack_slot_claimed(owner_id16: u32) -> u32 {
    SLOT_STATUS_CLAIMED | ((owner_id16 & 0xFFFF) << SLOT_OWNER_SHIFT)
}

#[inline]
pub const fn slot_status(word: u32) -> u32 {
    word & SLOT_STATUS_MASK
}

#[inline]
pub const fn slot_owner_id16(word: u32) -> u32 {
    (word >> SLOT_OWNER_SHIFT) & 0xFFFF
}

// Legacy v3 names — map to v4 fields where tests still import symbols.
pub const OFF_S_CLAIM_FLAG: usize = OFF_S_SLOT_STATE;
pub const OFF_S_OWNER_ID: usize = OFF_S_SLOT_STATE;
pub const OFF_S_LAST_WRITE_CLK: usize = OFF_S_LAST_CLAIM_MACH;
pub const OFF_S_LAST_INHIBIT_CLK: usize = OFF_S_LAST_CLAIM_MACH;

/// Deprecated: ζ-time stale delta; v4 uses Mach time in daemon.
pub const STALE_LOCK_ZETA_DELTA: f32 = 0.35;

#[inline]
pub const fn slot_base(i: usize) -> usize {
    GLOBAL_HDR_BYTES + i * SLOT_STRIDE
}

#[inline]
pub const fn slot_payload(i: usize) -> usize {
    slot_base(i) + SLOT_HDR_BYTES
}

// Compile-time contract: keep Rust layout identical to hiveclaw_layout_v4.h
const _: () = assert!(SLOT_STRIDE == 4224);
const _: () = assert!(OFF_SLOT_BACK_EPOCH == 4160);
const _: () = assert!(SLOT_HDR_BYTES == 64);
const _: () = assert!(SCENT_BYTES == 4096);
const _: () = assert!(SLOT_FOOTER_BYTES == 64);
const _: () = assert!(PHASE_C_BYTES == GLOBAL_HDR_BYTES + N_SLOTS * SLOT_STRIDE);
