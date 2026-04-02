/**
 * HiveClaw IOSurface Phase-C slab layout v4 (single source of truth for C++/MSL).
 * Rust mirrors these constants in hiveclaw-core::math.
 */
#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t HCLW_MAGIC = 0x48434C57u;
constexpr uint32_t HCLW_VERSION_V4 = 4u;

constexpr size_t HCLW_GLOBAL_HDR = 128;
constexpr size_t HCLW_SLOT_HDR = 64;
constexpr size_t HCLW_SCENT_ELEMS = 2048;
constexpr size_t HCLW_SCENT_BYTES = HCLW_SCENT_ELEMS * 2u;
/// Footer: back_epoch (4) + 60 bytes reserved (64-byte tail, cache-friendly).
constexpr size_t HCLW_SLOT_FOOTER = 64;
constexpr size_t HCLW_SLOT_STRIDE =
    HCLW_SLOT_HDR + HCLW_SCENT_BYTES + HCLW_SLOT_FOOTER; // 4224
constexpr size_t HCLW_N_SLOTS = 32;

constexpr size_t HCLW_OFF_SLOT_BACK_EPOCH = HCLW_SLOT_HDR + HCLW_SCENT_BYTES; // 4160 rel slot

// Global header (same offsets as v3 for magic/version/n_slots/stride/zeta/decay)
constexpr size_t OFF_G_MAGIC = 0;
constexpr size_t OFF_G_VERSION = 4;
constexpr size_t OFF_G_N_SLOTS = 8;
constexpr size_t OFF_G_SLOT_STRIDE = 12;
constexpr size_t OFF_G_ZETA_T = 16;
constexpr size_t OFF_G_DECAY_RATE = 20;

// Slot header v4 (relative to slot base)
constexpr size_t OFF_S_SLOT_STATE = 0;       // u32 bitfield
constexpr size_t OFF_S_LAST_CLAIM_MACH = 4;  // u64 mach_absolute_time ticks
constexpr size_t OFF_S_FRONT_EPOCH = 12;     // u32
constexpr size_t OFF_S_WATCHDOG_FLAGS = 16;  // u32 (bit0 = forced eviction)
// bytes 20..63 reserved (zero)

// slot_state encoding
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
