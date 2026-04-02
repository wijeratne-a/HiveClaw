//! Shared layout constants (slab v5).

/// IOSurface-backed slab size (bytes) for the Phase 2 POC.
pub const SLAB_SIZE: usize = 4_718_720;

// ── Slab v5 (4096-byte global header, 640-byte slots, 256-D latent) ────────

/// Magic `'HCLW'` in little-endian `u32` (low 32 bits of packed magic).
pub const SLAB_MAGIC: u32 = 0x4843_4C57;
pub const SLAB_VERSION_V5: u32 = 5;

pub const GLOBAL_HDR_BYTES: usize = 4096;
pub const SLOT_HDR_BYTES: usize = 64;
pub const SCENT_ELEMS: usize = 256;
pub const SCENT_BYTES: usize = SCENT_ELEMS * 2; // bf16
pub const SLOT_FOOTER_BYTES: usize = 64;
pub const SLOT_STRIDE: usize = SLOT_HDR_BYTES + SCENT_BYTES + SLOT_FOOTER_BYTES; // 640
pub const N_SLOTS: usize = 4096;
pub const PHASE_C_BYTES: usize = GLOBAL_HDR_BYTES + N_SLOTS * SLOT_STRIDE;

/// Stale held-slot eviction threshold (wall milliseconds), converted to Mach ticks in daemon.
pub const STALE_LOCK_MS: u64 = 500;

/// Global header: packed `u64` magic+version for XPC (`0x48434C5700000005`).
#[inline]
pub const fn layout_magic_version_u64() -> u64 {
    ((SLAB_MAGIC as u64) << 32) | (SLAB_VERSION_V5 as u64)
}

// Global header field offsets (first 64 bytes used for contract fields)
pub const OFF_G_MAGIC_V5: usize = 0; // u64
pub const OFF_G_VERSION_V5: usize = 8; // u32
pub const OFF_G_N_SLOTS_V5: usize = 12; // u32
pub const OFF_G_STRIDE_V5: usize = 16; // u32
/// Daemon decay: ζ clock (reserved region, not part of client contract).
pub const OFF_G_ZETA_T: usize = 32;
pub const OFF_G_DECAY_RATE: usize = 36;

/// Slot header v5 (relative to slot base).
pub const OFF_S_SLOT_STATE: usize = 0;
pub const OFF_S_LAST_CLAIM_MACH: usize = 4;
pub const OFF_S_FRONT_EPOCH: usize = 12;

/// `back_epoch` u32 immediately after 512-byte scent payload (slot-relative).
pub const OFF_SLOT_BACK_EPOCH: usize = SLOT_HDR_BYTES + SCENT_BYTES; // 576

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

/// Alias for claim CAS offset (same as `OFF_S_SLOT_STATE`).
pub const OFF_S_CLAIM_FLAG: usize = OFF_S_SLOT_STATE;

#[inline]
pub const fn slot_base(i: usize) -> usize {
    GLOBAL_HDR_BYTES + i * SLOT_STRIDE
}

#[inline]
pub const fn slot_payload(i: usize) -> usize {
    slot_base(i) + SLOT_HDR_BYTES
}

// Compile-time contract: keep Rust layout identical to hiveclaw_layout_v5.h
const _: () = assert!(PHASE_C_BYTES <= SLAB_SIZE);
const _: () = assert!(GLOBAL_HDR_BYTES + 4095 * SLOT_STRIDE + SLOT_STRIDE <= SLAB_SIZE);
const _: () = assert!(OFF_SLOT_BACK_EPOCH == 576);
const _: () = assert!(SLOT_STRIDE == 640);
const _: () = assert!(SCENT_BYTES == 512);
const _: () = assert!(SLOT_HDR_BYTES == 64);
const _: () = assert!(SLOT_FOOTER_BYTES == 64);
const _: () = assert!(PHASE_C_BYTES == GLOBAL_HDR_BYTES + N_SLOTS * SLOT_STRIDE);
