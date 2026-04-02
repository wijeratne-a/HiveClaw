#pragma once

#include "../../hiveclaw-core/include/hiveclaw_layout_v4.h"

#include <cstddef>

// Aliases for existing sources (v4 packs owner into slot_state).
constexpr size_t OFF_S_CLAIM_FLAG = OFF_S_SLOT_STATE;
constexpr size_t HCLW_VERSION_C = HCLW_VERSION_V4;

inline size_t slot_base(size_t i) { return hclw_slot_base(i); }

inline size_t slot_payload(size_t i) { return hclw_slot_payload(i); }
