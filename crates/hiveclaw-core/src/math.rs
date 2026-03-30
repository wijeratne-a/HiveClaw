//! Shared layout constants.

/// IOSurface-backed slab size (bytes) for the Phase 2 POC.
pub const SLAB_SIZE: usize = 4_718_720;

/// Slot 0: scalar probe region (f32×4) — byte offset from slab base.
pub const SLOT0_SCALAR_BYTE_OFFSET: usize = 256;

/// Slot 0: scent vector (bf16×1024) — byte offset from slab base.
pub const SLOT0_SCENT_BYTE_OFFSET: usize = 384;

/// Number of bf16 elements in slot 0 scent.
pub const SLOT0_SCENT_ELEMS: usize = 1024;
