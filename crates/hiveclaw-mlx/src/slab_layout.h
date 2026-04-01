#pragma once

#include <cstddef>
#include <cstdint>

constexpr uint32_t HCLW_MAGIC = 0x48434C57u;
constexpr uint32_t HCLW_VERSION_C = 3u;
constexpr size_t HCLW_GLOBAL_HDR = 128;
constexpr size_t HCLW_SLOT_HDR = 64;
constexpr size_t HCLW_SCENT_ELEMS = 2048;
constexpr size_t HCLW_SCENT_BYTES = HCLW_SCENT_ELEMS * 2; // bf16
constexpr size_t HCLW_SLOT_STRIDE = HCLW_SLOT_HDR + HCLW_SCENT_BYTES; // 4160
constexpr size_t HCLW_N_SLOTS = 32;

// Global header field offsets
constexpr size_t OFF_G_MAGIC = 0;
constexpr size_t OFF_G_VERSION = 4;
constexpr size_t OFF_G_N_SLOTS = 8;
constexpr size_t OFF_G_SLOT_STRIDE = 12;
constexpr size_t OFF_G_ZETA_T = 16; // float
constexpr size_t OFF_G_DECAY_RATE = 20; // float

// Slot header field offsets (relative to slot base)
constexpr size_t OFF_S_CLAIM_FLAG = 0;
constexpr size_t OFF_S_OWNER_ID = 4;
constexpr size_t OFF_S_LAST_WRITE_CLK = 8;
constexpr size_t OFF_S_WATCHDOG_FLAGS = 12;
constexpr size_t OFF_S_LAST_INHIBIT_CLK = 16;

inline size_t slot_base(size_t i) {
    return HCLW_GLOBAL_HDR + i * HCLW_SLOT_STRIDE;
}

inline size_t slot_payload(size_t i) {
    return slot_base(i) + HCLW_SLOT_HDR;
}
