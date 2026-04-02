#pragma once

#include "../../hiveclaw-core/include/hiveclaw_layout_v5.h"

#include <cstddef>

constexpr size_t OFF_S_CLAIM_FLAG = OFF_S_SLOT_STATE;

inline size_t slot_base(size_t i) { return hclw_slot_base(i); }

inline size_t slot_payload(size_t i) { return hclw_slot_payload(i); }
