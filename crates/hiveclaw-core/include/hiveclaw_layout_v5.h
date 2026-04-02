/**
 * HiveClaw IOSurface Phase-C slab layout v5 (single source of truth for C++/MSL).
 * Rust mirrors these constants in hiveclaw-core::math.
 */
#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint64_t HCLW_MAGIC_V5 = 0x48434C5700000005ULL;
constexpr uint32_t HCLW_VERSION_V5 = 5u;

constexpr size_t HCLW_GLOBAL_HDR = 4096;
constexpr size_t HCLW_SLOT_HDR = 64;
constexpr size_t HCLW_SCENT_ELEMS = 256;
constexpr size_t HCLW_SCENT_BYTES = HCLW_SCENT_ELEMS * 2u; // 512
constexpr size_t HCLW_SLOT_FOOTER = 64;
constexpr size_t HCLW_SLOT_STRIDE =
    HCLW_SLOT_HDR + HCLW_SCENT_BYTES + HCLW_SLOT_FOOTER; // 640
constexpr size_t HCLW_N_SLOTS = 4096;

constexpr size_t HCLW_OFF_SLOT_BACK_EPOCH = HCLW_SLOT_HDR + HCLW_SCENT_BYTES; // 576

// Global header field offsets
constexpr size_t OFF_G_MAGIC_V5 = 0;   // u64
constexpr size_t OFF_G_VERSION_V5 = 8;   // u32
constexpr size_t OFF_G_N_SLOTS_V5 = 12;  // u32
constexpr size_t OFF_G_STRIDE_V5 = 16;   // u32
constexpr size_t OFF_G_ZETA_T = 32;      // f32 (daemon decay)
constexpr size_t OFF_G_DECAY_RATE = 36;  // f32

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

inline uint32_t hclw_pack_claimed(uint32_t owner_id16) {
    return HCLW_SLOT_STATUS_CLAIMED | ((owner_id16 & 0xFFFFu) << HCLW_SLOT_OWNER_SHIFT);
}

inline uint32_t hclw_slot_status(uint32_t word) { return word & HCLW_SLOT_STATUS_MASK; }

inline uint32_t hclw_slot_owner16(uint32_t word) {
    return (word >> HCLW_SLOT_OWNER_SHIFT) & 0xFFFFu;
}

inline size_t hclw_slot_base(size_t i) {
    return HCLW_GLOBAL_HDR + i * HCLW_SLOT_STRIDE;
}

inline size_t hclw_slot_payload(size_t i) {
    return hclw_slot_base(i) + HCLW_SLOT_HDR;
}

static_assert(HCLW_SLOT_STRIDE == 640);
static_assert(HCLW_OFF_SLOT_BACK_EPOCH == 576);
static_assert(HCLW_SCENT_BYTES == 512);
static_assert(HCLW_N_SLOTS == 4096);
static_assert(HCLW_GLOBAL_HDR == 4096);
static_assert(HCLW_SCENT_ELEMS == 256);
static_assert(HCLW_SLOT_FOOTER == 64);
static_assert(HCLW_SLOT_HDR == 64);
