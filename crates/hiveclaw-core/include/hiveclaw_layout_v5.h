/**
 * HiveClaw IOSurface Phase-C slab layout v6 (C++/MSL single source of truth).
 * Latent width is runtime (global header); Rust mirrors in hiveclaw-core::math.
 */
#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint64_t HCLW_MAGIC_V6 = 0x48434C5700000006ULL;
constexpr uint32_t HCLW_VERSION_V6 = 6u;

constexpr size_t HCLW_GLOBAL_HDR = 4096;
constexpr size_t HCLW_SLOT_HDR = 64;
constexpr size_t HCLW_SLOT_FOOTER = 64;
constexpr size_t HCLW_N_SLOTS = 4096;

// Global header field offsets (contract; first 64 bytes)
constexpr size_t OFF_G_MAGIC_V6 = 0;    // u64 packed magic+version
constexpr size_t OFF_G_VERSION_V6 = 8;  // u32 (redundant; also in magic word)
constexpr size_t OFF_G_N_SLOTS_V6 = 12; // u32
constexpr size_t OFF_G_STRIDE_V6 = 16;  // u32 bytes per slot
constexpr size_t OFF_G_LATENT_ELEMS = 20; // u32 bf16 latent width D
constexpr size_t OFF_G_ZETA_T = 32;     // f32 (daemon decay)
constexpr size_t OFF_G_DECAY_RATE = 36; // f32

// Slot header (relative to slot base)
constexpr size_t OFF_S_SLOT_STATE = 0;
constexpr size_t OFF_S_LAST_CLAIM_MACH = 4;
constexpr size_t OFF_S_FRONT_EPOCH = 12;

constexpr uint32_t HCLW_SLOT_STATUS_MASK = 0x3u;
constexpr uint32_t HCLW_SLOT_STATUS_FREE = 0u;
constexpr uint32_t HCLW_SLOT_STATUS_CLAIMED = 1u;
constexpr uint32_t HCLW_SLOT_STATUS_INHIBITED = 2u;
constexpr uint32_t HCLW_SLOT_STATUS_FAULT = 3u;
constexpr uint32_t HCLW_SLOT_OWNER_SHIFT = 16u;

inline uint32_t hclw_slot_stride_bytes(uint32_t latent_elems) {
    return static_cast<uint32_t>(HCLW_SLOT_HDR) + latent_elems * 2u +
           static_cast<uint32_t>(HCLW_SLOT_FOOTER);
}

inline uint32_t hclw_off_slot_back_epoch(uint32_t latent_elems) {
    return static_cast<uint32_t>(HCLW_SLOT_HDR) + latent_elems * 2u;
}

inline uint32_t hclw_pack_claimed(uint32_t owner_id16) {
    return HCLW_SLOT_STATUS_CLAIMED | ((owner_id16 & 0xFFFFu) << HCLW_SLOT_OWNER_SHIFT);
}

inline uint32_t hclw_slot_status(uint32_t word) { return word & HCLW_SLOT_STATUS_MASK; }

inline uint32_t hclw_slot_owner16(uint32_t word) {
    return (word >> HCLW_SLOT_OWNER_SHIFT) & 0xFFFFu;
}

inline size_t hclw_slot_base(size_t i, uint32_t stride) {
    return HCLW_GLOBAL_HDR + i * static_cast<size_t>(stride);
}

inline size_t hclw_slot_payload(size_t i, uint32_t stride) {
    return hclw_slot_base(i, stride) + HCLW_SLOT_HDR;
}

// Legacy names used by older includes (map to v6 offsets)
constexpr size_t OFF_G_MAGIC_V5 = OFF_G_MAGIC_V6;
constexpr size_t OFF_G_VERSION_V5 = OFF_G_VERSION_V6;
constexpr size_t OFF_G_N_SLOTS_V5 = OFF_G_N_SLOTS_V6;
constexpr size_t OFF_G_STRIDE_V5 = OFF_G_STRIDE_V6;

static_assert(HCLW_N_SLOTS == 4096);
static_assert(HCLW_GLOBAL_HDR == 4096);
static_assert(HCLW_SLOT_FOOTER == 64);
static_assert(HCLW_SLOT_HDR == 64);
