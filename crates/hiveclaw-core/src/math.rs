//! Shared slab layout v6: runtime latent width in global header; stride derived.

use core::fmt;

/// Upper bound for a single IOSurface allocation (sanity check).
pub const SLAB_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Default IOSurface size when using legacy fixed layout (256-D latent); kept for docs/tests.
pub const SLAB_SIZE: usize = 4_718_720;

pub const SLAB_MAGIC: u32 = 0x4843_4C57;
pub const SLAB_VERSION_V6: u32 = 6;

pub const GLOBAL_HDR_BYTES: usize = 4096;
pub const SLOT_HDR_BYTES: usize = 64;
pub const SLOT_FOOTER_BYTES: usize = 64;
pub const N_SLOTS: usize = 4096;

/// Default latent width (bf16 elems per slot payload) matching historical v5 SAEs.
pub const DEFAULT_LATENT_ELEMS: u32 = 256;

/// Stale held-slot eviction threshold (wall milliseconds).
pub const STALE_LOCK_MS: u64 = 500;

#[derive(Clone, Debug)]
pub struct SlabLayout {
    pub latent_elems: u32,
    pub stride: u32,
    pub n_slots: u32,
    pub phase_c_bytes: usize,
    pub iosurface_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    LatentElemsOutOfRange { got: u32 },
    BadSlabVersion { got: u32 },
    StrideOverflow,
    PhaseCOverflow,
    ExceedsMaxSlab { need: usize, max: usize },
    StrideLatentMismatch { stride: u32, latent_elems: u32 },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::LatentElemsOutOfRange { got } => {
                write!(f, "latent_elems out of range: {got}")
            }
            LayoutError::BadSlabVersion { got } => write!(
                f,
                "slab global header version mismatch: got {got}, expected {}",
                SLAB_VERSION_V6
            ),
            LayoutError::StrideOverflow => write!(f, "slot stride overflow"),
            LayoutError::PhaseCOverflow => write!(f, "phase_c byte size overflow"),
            LayoutError::ExceedsMaxSlab { need, max } => {
                write!(f, "slab need {need} bytes exceeds max {max}")
            }
            LayoutError::StrideLatentMismatch { stride, latent_elems } => write!(
                f,
                "stride {stride} != 128 + 2*{latent_elems}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

#[inline]
pub fn slot_stride_u32(latent_elems: u32) -> Option<u32> {
    let body = latent_elems.checked_mul(2)?;
    let add = (SLOT_HDR_BYTES as u32).checked_add(body)?;
    add.checked_add(SLOT_FOOTER_BYTES as u32)
}

#[inline]
pub fn off_slot_back_epoch(latent_elems: u32) -> u32 {
    SLOT_HDR_BYTES as u32 + latent_elems.saturating_mul(2)
}

#[inline]
pub fn page_align_4096(n: usize) -> usize {
    const PAGE: usize = 4096;
    n.checked_add(PAGE - 1).map(|x| x / PAGE * PAGE).unwrap_or(usize::MAX)
}

impl SlabLayout {
    /// Build layout for `n_slots` (normally 4096) and `latent_elems` bf16 payload width.
    pub fn try_from_latent_elems(latent_elems: u32, n_slots: u32) -> Result<Self, LayoutError> {
        if latent_elems == 0 || latent_elems > 1_000_000 {
            return Err(LayoutError::LatentElemsOutOfRange { got: latent_elems });
        }
        let stride = slot_stride_u32(latent_elems).ok_or(LayoutError::StrideOverflow)?;
        let n = n_slots as usize;
        let body = n
            .checked_mul(stride as usize)
            .ok_or(LayoutError::PhaseCOverflow)?;
        let phase_c_bytes = GLOBAL_HDR_BYTES
            .checked_add(body)
            .ok_or(LayoutError::PhaseCOverflow)?;
        let iosurface_bytes = page_align_4096(phase_c_bytes);
        if iosurface_bytes > SLAB_MAX_BYTES {
            return Err(LayoutError::ExceedsMaxSlab {
                need: iosurface_bytes,
                max: SLAB_MAX_BYTES,
            });
        }
        Ok(Self {
            latent_elems,
            stride,
            n_slots,
            phase_c_bytes,
            iosurface_bytes,
        })
    }

    #[inline]
    pub fn slot_base(&self, i: usize) -> usize {
        GLOBAL_HDR_BYTES + i * self.stride as usize
    }

    #[inline]
    pub fn slot_payload(&self, i: usize) -> usize {
        self.slot_base(i) + SLOT_HDR_BYTES
    }
}

/// Global header: packed `u64` magic+version for XPC.
#[inline]
pub const fn layout_magic_version_u64() -> u64 {
    ((SLAB_MAGIC as u64) << 32) | (SLAB_VERSION_V6 as u64)
}

pub const OFF_G_MAGIC_V6: usize = 0;
pub const OFF_G_VERSION_V6: usize = 8;
pub const OFF_G_N_SLOTS_V6: usize = 12;
pub const OFF_G_STRIDE_V6: usize = 16;
pub const OFF_G_LATENT_ELEMS: usize = 20;
pub const OFF_G_ZETA_T: usize = 32;
pub const OFF_G_DECAY_RATE: usize = 36;

// Aliases for older symbol names in crates
pub const OFF_G_MAGIC_V5: usize = OFF_G_MAGIC_V6;
pub const OFF_G_VERSION_V5: usize = OFF_G_VERSION_V6;
pub const OFF_G_N_SLOTS_V5: usize = OFF_G_N_SLOTS_V6;
pub const OFF_G_STRIDE_V5: usize = OFF_G_STRIDE_V6;

pub const OFF_S_SLOT_STATE: usize = 0;
pub const OFF_S_LAST_CLAIM_MACH: usize = 4;
pub const OFF_S_FRONT_EPOCH: usize = 12;
pub const OFF_S_CLAIM_FLAG: usize = OFF_S_SLOT_STATE;

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

/// Read stride/latent from a mapped v6 header and verify consistency.
pub unsafe fn read_layout_from_header(base: *const u8) -> Result<SlabLayout, LayoutError> {
    let ver = base.add(OFF_G_VERSION_V6).cast::<u32>().read_unaligned();
    if ver != SLAB_VERSION_V6 {
        return Err(LayoutError::BadSlabVersion { got: ver });
    }
    let n_slots = base.add(OFF_G_N_SLOTS_V6).cast::<u32>().read_unaligned();
    let stride = base.add(OFF_G_STRIDE_V6).cast::<u32>().read_unaligned();
    let latent = base.add(OFF_G_LATENT_ELEMS).cast::<u32>().read_unaligned();
    let expected = slot_stride_u32(latent).ok_or(LayoutError::StrideOverflow)?;
    if stride != expected {
        return Err(LayoutError::StrideLatentMismatch { stride, latent_elems: latent });
    }
    let lay = SlabLayout::try_from_latent_elems(latent, n_slots)?;
    if lay.stride != stride {
        return Err(LayoutError::StrideLatentMismatch { stride, latent_elems: latent });
    }
    Ok(lay)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_fits_legacy_slab_cap() {
        let lay = SlabLayout::try_from_latent_elems(DEFAULT_LATENT_ELEMS, N_SLOTS as u32).unwrap();
        assert!(lay.iosurface_bytes <= SLAB_SIZE);
        assert_eq!(lay.stride, 640);
    }
}
