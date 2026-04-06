#include "slab_bridge.h"
#include "slab_primitives.h"
#include "slab_layout.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mach/mach_time.h>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

using mlx::core::Shape;
using mlx::core::array;
using mlx::core::bfloat16;
using mlx::core::default_stream;
using mlx::core::depends;
using mlx::core::Device;
using mlx::core::Primitive;
using mlx::core::Stream;
using mlx::core::zeros_like;

namespace {

// Python batched slots: int32 -1 → bit pattern 0xFFFFFFFF (dummy row; no IOSurface access).
static constexpr uint32_t HIVECLAW_SENTINEL_SLOT = 0xFFFFFFFFu;

struct SlabParsedHeader {
    uint32_t latent_elems;
    uint32_t stride;
    uint32_t n_slots;
    size_t slab_bytes;
};

static SlabParsedHeader parse_slab_header(MTL::Buffer* buf) {
    void* c = buf->contents();
    const uint8_t* p = static_cast<const uint8_t*>(c);
    uint32_t ver = *reinterpret_cast<const uint32_t*>(p + OFF_G_VERSION_V6);
    if (ver != HCLW_VERSION_V6) {
        throw std::runtime_error("SlabHandle: IOSurface global header must be slab v6");
    }
    uint32_t latent = *reinterpret_cast<const uint32_t*>(p + OFF_G_LATENT_ELEMS);
    uint32_t stride = *reinterpret_cast<const uint32_t*>(p + OFF_G_STRIDE_V6);
    uint32_t n = *reinterpret_cast<const uint32_t*>(p + OFF_G_N_SLOTS_V6);
    uint32_t exp = hclw_slot_stride_bytes(latent);
    if (stride != exp) {
        throw std::runtime_error("SlabHandle: stride/latent_elems mismatch in global header");
    }
    size_t len = static_cast<size_t>(buf->length());
    return {latent, stride, n, len};
}

static const char* COPY_BF16_MSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void copy_bf16(
    const device uint16_t* src [[buffer(0)]],
    device uint16_t* dst [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  dst[index] = src[index];
}

)";

// v6: SlabParams via buffer(4); 32-wide threadgroup; latent_elems arbitrary.
static const char* READ_V5_MSL = R"(
#include <metal_stdlib>
using namespace metal;

struct SlabParams {
  uint global_hdr;
  uint stride;
  uint slot_hdr;
  uint latent_elems;
  uint back_epoch_off;
};

kernel void read_slab_v5(
    const device uint8_t* slab [[buffer(0)]],
    device uint16_t* h_out [[buffer(1)]],
    device uint8_t* status [[buffer(2)]],
    constant uint32_t& slot_index [[buffer(3)]],
    constant SlabParams& P [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
  constant uint OFF_S_FRONT_EPOCH = 12u;
  uint slot_base = P.global_hdr + slot_index * P.stride;
  threadgroup uint front_ep;
  threadgroup uint back_ep;
  threadgroup uint torn;

  if (tid == 0u) {
    front_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + OFF_S_FRONT_EPOCH);
    torn = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint payload_base = slot_base + P.slot_hdr;
  uint L = P.latent_elems;
  for (uint iter = 0u; iter < (L + 31u) / 32u; ++iter) {
    uint i = tid + iter * 32u;
    if (i < L) {
      uint16_t v = *reinterpret_cast<const device uint16_t*>(slab + payload_base + i * 2u);
      h_out[i] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    back_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + P.back_epoch_off);
    if (front_ep != back_ep) {
      torn = 1u;
      status[0] = 1u;
    } else {
      status[0] = 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (torn != 0u) {
    uint L2 = P.latent_elems;
    for (uint iter = 0u; iter < (L2 + 31u) / 32u; ++iter) {
      uint i = tid + iter * 32u;
      if (i < L2) {
        h_out[i] = 0u;
      }
    }
  }
}
)";

// Batched v6 read: params buffer(4); row offset b * latent_elems.
static const char* READ_V5_BATCHED_MSL = R"(
#include <metal_stdlib>
using namespace metal;

struct SlabParams {
  uint global_hdr;
  uint stride;
  uint slot_hdr;
  uint latent_elems;
  uint back_epoch_off;
};

kernel void read_slab_v5_batched(
    const device uint8_t* slab [[buffer(0)]],
    device uint16_t* h_out [[buffer(1)]],
    device uint8_t* status_out [[buffer(2)]],
    const device uint32_t* slot_indices [[buffer(3)]],
    constant SlabParams& P [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constant uint OFF_S_FRONT_EPOCH = 12u;
  uint b = tgp.z;
  uint slot_index = slot_indices[b];
  uint L = P.latent_elems;
  device uint16_t* row = h_out + (b * L);

  threadgroup uint is_sentinel;
  if (tid == 0u) {
    is_sentinel = (slot_index == 0xFFFFFFFFu) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (is_sentinel != 0u) {
    for (uint iter = 0u; iter < (L + 31u) / 32u; ++iter) {
      uint i = tid + iter * 32u;
      if (i < L) {
        row[i] = 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      status_out[b] = 0u;
    }
    return;
  }

  uint slot_base = P.global_hdr + slot_index * P.stride;
  threadgroup uint front_ep;
  threadgroup uint back_ep;
  threadgroup uint torn;

  if (tid == 0u) {
    front_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + OFF_S_FRONT_EPOCH);
    torn = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint payload_base = slot_base + P.slot_hdr;
  for (uint iter = 0u; iter < (L + 31u) / 32u; ++iter) {
    uint i = tid + iter * 32u;
    if (i < L) {
      uint16_t v = *reinterpret_cast<const device uint16_t*>(slab + payload_base + i * 2u);
      row[i] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    back_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + P.back_epoch_off);
    if (front_ep != back_ep) {
      torn = 1u;
      status_out[b] = 1u;
    } else {
      status_out[b] = 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (torn != 0u) {
    for (uint iter = 0u; iter < (L + 31u) / 32u; ++iter) {
      uint i = tid + iter * 32u;
      if (i < L) {
        row[i] = 0u;
      }
    }
  }
}
)";

// Batched v6 write: SlabParams buffer(4); row offset b * latent_elems.
static const char* WRITE_V5_BATCHED_MSL = R"(
#include <metal_stdlib>
using namespace metal;

struct SlabParams {
  uint global_hdr;
  uint stride;
  uint slot_hdr;
  uint latent_elems;
  uint back_epoch_off;
};

kernel void write_slab_v5_batched(
    device uint8_t* slab [[buffer(0)]],
    const device uint16_t* h_in [[buffer(1)]],
    device uint8_t* status_out [[buffer(2)]],
    const device uint32_t* slot_indices [[buffer(3)]],
    constant SlabParams& P [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constant uint OFF_S_CLAIM_FLAG = 0u;
  constant uint OFF_S_FRONT_EPOCH = 12u;
  constant uint HCLW_SLOT_STATUS_MASK = 3u;
  constant uint HCLW_SLOT_STATUS_CLAIMED = 1u;

  uint b = tgp.z;
  uint slot_index = slot_indices[b];
  uint L = P.latent_elems;

  threadgroup uint is_sentinel;
  if (tid == 0u) {
    is_sentinel = (slot_index == 0xFFFFFFFFu) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint slot_base = 0u;
  device uint8_t* sh = slab;
  if (is_sentinel == 0u) {
    slot_base = P.global_hdr + slot_index * P.stride;
    sh = slab + slot_base;
  }

  threadgroup uint epoch_val;
  threadgroup uint lost;

  if (tid == 0u) {
    lost = 0u;
    if (is_sentinel != 0u) {
      status_out[b] = 0u;
    } else {
      uint32_t word = *reinterpret_cast<const device uint32_t*>(sh + OFF_S_CLAIM_FLAG);
      if ((word & HCLW_SLOT_STATUS_MASK) != HCLW_SLOT_STATUS_CLAIMED) {
        status_out[b] = 2u;
        lost = 1u;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (is_sentinel != 0u) {
    return;
  }
  if (lost != 0u) {
    return;
  }

  if (tid == 0u) {
    uint32_t fe = *reinterpret_cast<const device uint32_t*>(sh + OFF_S_FRONT_EPOCH);
    epoch_val = fe + 1u;
    *reinterpret_cast<device uint32_t*>(sh + OFF_S_FRONT_EPOCH) = epoch_val;
  }
  threadgroup_barrier(mem_flags::mem_device);

  const device uint16_t* row = h_in + (b * L);
  uint payload_base = slot_base + P.slot_hdr;
  for (uint iter = 0u; iter < (L + 31u) / 32u; ++iter) {
    uint i = tid + iter * 32u;
    if (i < L) {
      *reinterpret_cast<device uint16_t*>(slab + payload_base + i * 2u) = row[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  if (tid == 0u) {
    *reinterpret_cast<device uint32_t*>(sh + P.back_epoch_off) = epoch_val;
    status_out[b] = 0u;
  }
}
)";

static bool hiveclaw_telemetry_enabled() {
    const char* e = std::getenv("HIVECLAW_TELEMETRY");
    if (e == nullptr) {
        return true;
    }
    return std::string(e) != "0";
}

/// Opt-in: batched read uses Metal + shared staging buffer (see ReadSlabBatchedOp::eval_gpu).
static bool hiveclaw_gpu_batch_read_enabled() {
    const char* e = std::getenv("HIVECLAW_GPU_BATCH_READ");
    return e != nullptr && std::string(e) == "1";
}

/// Opt-in: batched write dispatches WRITE_V5_BATCHED_MSL (IOSurface + status_out device buffer).
static bool hiveclaw_gpu_batch_write_enabled() {
    const char* e = std::getenv("HIVECLAW_GPU_BATCH_WRITE");
    return e != nullptr && std::string(e) == "1";
}

static void emit_read_v5_telemetry(uint8_t st, uint32_t slot_id) {
    if (!hiveclaw_telemetry_enabled() || st == 0) {
        return;
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    std::ostringstream oss;
    oss << "{\"event\":\"torn_epoch_skip\",\"slot_id\":" << slot_id << ",\"ts_ns\":" << ns
        << "}\n";
    std::cerr << oss.str();
}

static void emit_torn_batch_telemetry(
    const uint8_t* status,
    uint32_t B,
    const std::vector<uint32_t>& slot_indices) {
    if (!hiveclaw_telemetry_enabled()) {
        return;
    }
    std::vector<uint32_t> torn;
    torn.reserve(B);
    for (uint32_t i = 0; i < B; ++i) {
        if (status[i] == 1 && slot_indices[i] != HIVECLAW_SENTINEL_SLOT) {
            torn.push_back(slot_indices[i]);
        }
    }
    if (torn.empty()) {
        return;
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    std::ostringstream oss;
    oss << "{\"event\":\"torn_epoch_skip_batch\",\"slots\":[";
    for (size_t j = 0; j < torn.size(); ++j) {
        if (j > 0) {
            oss << ',';
        }
        oss << torn[j];
    }
    oss << "],\"ts_ns\":" << ns << "}\n";
    std::cerr << oss.str();
}

static std::vector<uint32_t> validate_slot_indices_batched(const std::vector<uint32_t>& v,
                                                             uint32_t n_slots) {
    if (v.empty()) {
        throw std::invalid_argument("batched slab: slot_indices must be non-empty");
    }
    std::set<uint32_t> seen;
    for (uint32_t x : v) {
        if (x == HIVECLAW_SENTINEL_SLOT) {
            continue;
        }
        if (x >= n_slots) {
            throw std::invalid_argument("batched slab: slot_index out of range");
        }
        if (!seen.insert(x).second) {
            throw std::invalid_argument("batched slab: duplicate real slot_index");
        }
    }
    return v;
}

static void validate_dep_rank3(const std::optional<array>& dep, uint32_t B, int latent_d) {
    if (!dep) {
        return;
    }
    if (dep->ndim() != 3) {
        throw std::invalid_argument("batched slab depends: expected rank-3 tensor");
    }
    const auto& sh = dep->shape();
    if (sh[0] != static_cast<int>(B) || sh[1] != 1) {
        throw std::invalid_argument("batched slab depends: expected shape [B,1,D]");
    }
    const int d = sh[2];
    if (d != 2048 && d != latent_d) {
        throw std::invalid_argument(
            "batched slab depends: last dim must be 2048 or latent_dim");
    }
}

static void validate_latents_batched_shape(const array& latents, uint32_t B, uint32_t latent_d) {
    if (latents.dtype() != bfloat16) {
        throw std::invalid_argument("write_slots_v5: latents must be bfloat16");
    }
    const auto& sh = latents.shape();
    if (sh.size() != 3 || sh[0] != static_cast<int>(B) || sh[1] != 1 ||
        sh[2] != static_cast<int>(latent_d)) {
        throw std::invalid_argument(
            "write_slots_v5: latents must be [B,1,latent_dim] bfloat16");
    }
}

static void validate_write_dep_latent(const std::optional<array>& dep, uint32_t B, uint32_t latent_d) {
    if (!dep) {
        return;
    }
    validate_dep_rank3(dep, B, static_cast<int>(latent_d));
    if (dep->shape()[2] != static_cast<int>(latent_d)) {
        throw std::invalid_argument(
            "write_slots_v5 depends: expected shape [B,1,latent_dim]");
    }
}

static void validate_write_v5_stamped(const array& in, uint32_t latent_d) {
    if (in.dtype() != bfloat16) {
        throw std::invalid_argument("WriteSlab v5: input must be bfloat16");
    }
    const auto& sh = in.shape();
    if (sh.size() != 3 || sh[0] != 1 || sh[1] != 1 || sh[2] != static_cast<int>(latent_d)) {
        throw std::invalid_argument("WriteSlab v5: input must be [1,1,latent_dim] bfloat16");
    }
    const size_t need = static_cast<size_t>(latent_d) * sizeof(uint16_t);
    if (in.nbytes() != need) {
        throw std::invalid_argument("WriteSlab v5: latent byte size mismatch");
    }
}

} // namespace

SlabHandle::SlabHandle(uint32_t surface_id) : surface_id_(surface_id) {
    auto* mtl_dev = mlx::core::metal::device(Device::gpu).mtl_device();
    void* p = create_slab_buffer(static_cast<void*>(mtl_dev), surface_id);
    slab_buf_ = static_cast<MTL::Buffer*>(p);
    if (!slab_buf_) {
        throw std::runtime_error("create_slab_buffer failed (IOSurfaceLookup or Metal buffer)");
    }
    SlabParsedHeader ph = parse_slab_header(slab_buf_);
    latent_elems_ = ph.latent_elems;
    stride_ = ph.stride;
    n_slots_ = ph.n_slots;
    back_epoch_off_ = hclw_off_slot_back_epoch(latent_elems_);
    slab_bytes_ = ph.slab_bytes;
}

SlabHandle::~SlabHandle() {
    if (slab_buf_) {
        release_slab_buffer(static_cast<void*>(slab_buf_));
        slab_buf_ = nullptr;
    }
}

array SlabHandle::write_slot(uint32_t slot_index,
                             array scent_c,
                             std::optional<array> dep) {
    if (slot_index >= n_slots_) {
        throw std::runtime_error("write_slot: slot_index out of range");
    }
    const size_t byte_offset = hclw_slot_payload(slot_index, stride_);
    Stream s = default_stream(Device::cpu);
    const size_t nbytes = scent_c.nbytes();
    auto prim = std::make_shared<WriteSlab>(
        slab_buf_,
        byte_offset,
        nbytes,
        s,
        slot_index,
        back_epoch_off_,
        latent_elems_);
    std::vector<array> inputs = {std::move(scent_c)};
    array out(inputs[0].shape(), inputs[0].dtype(), std::static_pointer_cast<Primitive>(prim), inputs);
    if (dep) {
        return depends({out}, {*dep})[0];
    }
    return out;
}

array SlabHandle::write_slot_v5(uint32_t slot_index,
                                array latent,
                                std::optional<array> dep) {
    validate_write_v5_stamped(latent, latent_elems_);
    return write_slot(slot_index, std::move(latent), std::move(dep));
}

array SlabHandle::write(size_t byte_offset,
                        array scent_c,
                        std::optional<array> dep) {
    Stream s = default_stream(Device::gpu);
    const size_t nbytes = scent_c.nbytes();
    auto prim = std::make_shared<WriteSlab>(
        slab_buf_, byte_offset, nbytes, s, 0xFFFFFFFFu, 0u, 0u);
    std::vector<array> inputs = {std::move(scent_c)};
    array out(inputs[0].shape(), inputs[0].dtype(), std::static_pointer_cast<Primitive>(prim), inputs);
    if (dep) {
        return depends({out}, {*dep})[0];
    }
    return out;
}

array SlabHandle::read_slot(uint32_t slot_index,
                            Shape shape,
                            std::optional<array> dep) {
    if (slot_index >= n_slots_) {
        throw std::runtime_error("read_slot: slot_index out of range");
    }
    return read(hclw_slot_payload(slot_index, stride_), std::move(shape), std::move(dep));
}

array SlabHandle::read_slot_v5(uint32_t slot_index, std::optional<array> dep) {
    if (slot_index >= n_slots_) {
        throw std::runtime_error("read_slot_v5: slot_index out of range");
    }
    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    auto prim = std::make_shared<ReadSlab>(
        slab_buf_,
        hclw_slot_payload(slot_index, stride_),
        Shape{1, 1, static_cast<mlx::core::ShapeElem>(latent_elems_)},
        s,
        slot_index,
        back_epoch_off_,
        latent_elems_);
    std::vector<array> inputs;
    if (dep) {
        inputs.push_back(*dep);
    }
    return array(
        Shape{1, 1, static_cast<mlx::core::ShapeElem>(latent_elems_)},
        bfloat16,
        std::static_pointer_cast<Primitive>(prim),
        inputs);
}

std::pair<array, array> SlabHandle::read_slots_v5(
    std::vector<uint32_t> slot_indices,
    std::optional<array> dep) {
    slot_indices = validate_slot_indices_batched(slot_indices, n_slots_);
    const uint32_t B = static_cast<uint32_t>(slot_indices.size());
    validate_dep_rank3(dep, B, static_cast<int>(latent_elems_));

    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    MTL::Device* mtl_dev = mlx::core::metal::device(s.device).mtl_device();
    auto status_ctx = std::make_shared<BatchStatusBuffer>(mtl_dev, B);
    auto prim = std::make_shared<ReadSlabBatchedOp>(
        slab_buf_,
        std::move(slot_indices),
        status_ctx,
        s,
        latent_elems_,
        stride_,
        back_epoch_off_);
    std::vector<array> inputs;
    if (dep) {
        inputs.push_back(*dep);
    }
    array data(
        Shape{static_cast<int>(B), 1, static_cast<int>(latent_elems_)},
        bfloat16,
        std::static_pointer_cast<Primitive>(prim),
        inputs);
    auto prim_s = std::make_shared<CopyBatchStatusOp>(status_ctx, s);
    array status(
        Shape{static_cast<int>(B)},
        mlx::core::uint8,
        std::static_pointer_cast<Primitive>(prim_s),
        {data});
    return {data, status};
}

std::pair<array, array> SlabHandle::write_slots_v5(
    std::vector<uint32_t> slot_indices,
    array latents,
    std::optional<array> dep) {
    slot_indices = validate_slot_indices_batched(slot_indices, n_slots_);
    const uint32_t B = static_cast<uint32_t>(slot_indices.size());
    validate_latents_batched_shape(latents, B, latent_elems_);
    validate_write_dep_latent(dep, B, latent_elems_);

    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    MTL::Device* mtl_dev = mlx::core::metal::device(s.device).mtl_device();
    auto status_ctx = std::make_shared<BatchStatusBuffer>(mtl_dev, B);
    auto prim = std::make_shared<WriteSlabBatchedOp>(
        slab_buf_,
        std::move(slot_indices),
        status_ctx,
        s,
        latent_elems_,
        stride_,
        back_epoch_off_);
    std::vector<array> inputs = {std::move(latents)};
    if (dep) {
        inputs.push_back(*dep);
    }
    array out(
        inputs[0].shape(),
        inputs[0].dtype(),
        std::static_pointer_cast<Primitive>(prim),
        inputs);
    auto prim_s = std::make_shared<CopyBatchStatusOp>(status_ctx, s);
    array status(
        Shape{static_cast<int>(B)},
        mlx::core::uint8,
        std::static_pointer_cast<Primitive>(prim_s),
        {out});
    return {out, status};
}

array SlabHandle::read(size_t byte_offset,
                       Shape shape,
                       std::optional<array> dep) {
    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    Shape out_shape = shape;
    auto prim = std::make_shared<ReadSlab>(slab_buf_, byte_offset, std::move(shape), s);
    std::vector<array> inputs;
    if (dep) {
        inputs.push_back(*dep);
    }
    return array(
        out_shape,
        bfloat16,
        std::static_pointer_cast<Primitive>(prim),
        inputs);
}

WriteSlab::WriteSlab(MTL::Buffer* slab,
                     size_t byte_offset,
                     size_t num_bytes,
                     Stream s,
                     uint32_t stamp_slot_index,
                     uint32_t v5_back_epoch_off,
                     uint32_t v5_latent_elems)
    : UnaryPrimitive(s),
      slab_buf_(slab),
      byte_offset_(byte_offset),
      num_bytes_(num_bytes),
      stamp_slot_index_(stamp_slot_index),
      v5_back_epoch_off_(v5_back_epoch_off),
      v5_latent_elems_(v5_latent_elems) {}

void WriteSlab::eval_cpu(const std::vector<array>& inputs, array& out) {
    auto& in = inputs[0];
    if (stamp_slot_index_ != 0xFFFFFFFFu) {
        validate_write_v5_stamped(in, v5_latent_elems_);
    } else if (in.nbytes() != num_bytes_) {
        throw std::runtime_error("WriteSlab: input size mismatch");
    }
    void* dst = static_cast<char*>(slab_buf_->contents()) + byte_offset_;
    if (stamp_slot_index_ != 0xFFFFFFFFu) {
        char* sh =
            static_cast<char*>(slab_buf_->contents()) + byte_offset_ - HCLW_SLOT_HDR;
        auto* fe = reinterpret_cast<std::atomic<uint32_t>*>(sh + OFF_S_FRONT_EPOCH);
        uint32_t e = fe->load(std::memory_order_relaxed) + 1u;
        fe->store(e, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(dst, in.data<const uint8_t>(), num_bytes_);
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<uint32_t*>(sh + static_cast<size_t>(v5_back_epoch_off_)) = e;
        *reinterpret_cast<uint64_t*>(sh + OFF_S_LAST_CLAIM_MACH) = mach_absolute_time();
    } else {
        std::memcpy(dst, in.data<const uint8_t>(), num_bytes_);
    }
    out.copy_shared_buffer(in);
}

void WriteSlab::eval_gpu(const std::vector<array>& inputs, array& out) {
    if (stamp_slot_index_ != 0xFFFFFFFFu) {
        eval_cpu(inputs, out);
        return;
    }
    auto& in = inputs[0];
    auto& d = mlx::core::metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    auto* lib = d.get_library("hiveclaw_copy_bf16", [] { return std::string(COPY_BF16_MSL); });

    size_t n = in.size();
    size_t tgp = std::min(n, static_cast<size_t>(256));

    auto* kernel = d.get_kernel("copy_bf16", lib);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(in, 0);
    enc.set_buffer(slab_buf_, 1, static_cast<int64_t>(byte_offset_));

    enc.dispatch_threads(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                         MTL::Size(static_cast<NS::UInteger>(tgp), 1, 1));

    out.copy_shared_buffer(in);
}

std::vector<array> WriteSlab::jvp(const std::vector<array>& primals,
                                  const std::vector<array>& tangents,
                                  const std::vector<int>& argnums) {
    (void)primals;
    std::vector<array> o;
    for (auto a : argnums) {
        (void)a;
        o.push_back(zeros_like(tangents[0], stream()));
    }
    return o;
}

std::vector<array> WriteSlab::vjp(const std::vector<array>& primals,
                                  const std::vector<array>& cotangents,
                                  const std::vector<int>& argnums,
                                  const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (auto a : argnums) {
        (void)a;
        vjps.push_back(zeros_like(primals[0], stream()));
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> WriteSlab::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[WriteSlab] vmap not implemented.");
}

std::vector<Shape> WriteSlab::output_shapes(const std::vector<array>& inputs) {
    return {inputs[0].shape()};
}

ReadSlab::ReadSlab(MTL::Buffer* slab,
                   size_t byte_offset,
                   Shape shape,
                   Stream s,
                   std::optional<uint32_t> v5_slot_for_epoch,
                   uint32_t v5_back_epoch_off,
                   uint32_t v5_latent_elems)
    : Primitive(s),
      slab_buf_(slab),
      byte_offset_(byte_offset),
      shape_(std::move(shape)),
      v5_slot_for_epoch_(v5_slot_for_epoch),
      v5_back_epoch_off_(v5_back_epoch_off),
      v5_latent_elems_(v5_latent_elems) {
    if (v5_slot_for_epoch_) {
        MTL::Device* dev = mlx::core::metal::device(s.device).mtl_device();
        status_buf_ = dev->newBuffer(1, MTL::ResourceStorageModeShared);
        if (!status_buf_) {
            throw std::runtime_error("ReadSlab v5: status buffer allocation failed");
        }
    }
}

ReadSlab::~ReadSlab() {
    if (status_buf_ != nullptr) {
        status_buf_->release();
        status_buf_ = nullptr;
    }
}

std::vector<Shape> ReadSlab::output_shapes(const std::vector<array>& inputs) {
    (void)inputs;
    return {shape_};
}

void ReadSlab::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    (void)inputs;
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    uint16_t* h_dst = out.data<uint16_t>();
    char* slab_base = static_cast<char*>(slab_buf_->contents());

    if (!v5_slot_for_epoch_) {
        void* src = slab_base + byte_offset_;
        std::memcpy(h_dst, src, out.nbytes());
        return;
    }

    uint32_t si = *v5_slot_for_epoch_;
    char* hdr = slab_base + byte_offset_ - HCLW_SLOT_HDR;
    const size_t scent_b = static_cast<size_t>(v5_latent_elems_) * 2u;

    auto load_u32 = [hdr](size_t off) -> uint32_t {
        return reinterpret_cast<std::atomic<uint32_t>*>(hdr + off)->load(
            std::memory_order_acquire);
    };

    const uint32_t fe = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be_pre = load_u32(static_cast<size_t>(v5_back_epoch_off_));
    std::memcpy(h_dst, hdr + HCLW_SLOT_HDR, scent_b);
    const uint32_t fe2 = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be2 = load_u32(static_cast<size_t>(v5_back_epoch_off_));
    if (fe != be_pre || fe2 != fe || be2 != be_pre || fe2 != be2) {
        std::memset(h_dst, 0, scent_b);
        emit_read_v5_telemetry(1, si);
    }
}

void ReadSlab::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    (void)inputs;
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    if (!v5_slot_for_epoch_) {
        auto& d = mlx::core::metal::device(stream().device);
        auto& enc = d.get_command_encoder(stream().index);

        auto* lib = d.get_library("hiveclaw_copy_bf16", [] { return std::string(COPY_BF16_MSL); });
        auto* kernel = d.get_kernel("copy_bf16", lib);
        enc.set_compute_pipeline_state(kernel);

        enc.set_buffer(slab_buf_, 0, static_cast<int64_t>(byte_offset_));
        enc.set_output_array(out, 1);

        size_t n = out.size();
        size_t tgp = std::min(n, static_cast<size_t>(256));
        enc.dispatch_threads(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                             MTL::Size(static_cast<NS::UInteger>(tgp), 1, 1));
        return;
    }

    *static_cast<uint8_t*>(status_buf_->contents()) = 0;

    auto& d = mlx::core::metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    auto* lib = d.get_library("hiveclaw_read_slab_v5", [] { return std::string(READ_V5_MSL); });
    auto* kernel = d.get_kernel("read_slab_v5", lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_output_array(out, 1);
    enc.set_buffer(status_buf_, 2, 0);
    uint32_t si = *v5_slot_for_epoch_;
    enc.set_bytes(si, 3);

    struct SlabParamsEnc {
        uint32_t global_hdr;
        uint32_t stride;
        uint32_t slot_hdr;
        uint32_t latent_elems;
        uint32_t back_epoch_off;
    };
    const uint8_t* gp = static_cast<const uint8_t*>(slab_buf_->contents());
    SlabParamsEnc P{
        static_cast<uint32_t>(HCLW_GLOBAL_HDR),
        *reinterpret_cast<const uint32_t*>(gp + OFF_G_STRIDE_V6),
        static_cast<uint32_t>(HCLW_SLOT_HDR),
        v5_latent_elems_,
        v5_back_epoch_off_};
    enc.set_bytes(&P, sizeof(P), 4);

    enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));

    MTL::CommandBuffer* cb = d.get_command_buffer(stream().index);
    if (cb != nullptr) {
        status_buf_->retain();
        MTL::Buffer* sbuf = status_buf_;
        const uint32_t slot_id = si;
        const bool tel = hiveclaw_telemetry_enabled();
        cb->addCompletedHandler([sbuf, slot_id, tel](MTL::CommandBuffer* /*cmd*/) {
            if (tel) {
                uint8_t st = *static_cast<uint8_t*>(sbuf->contents());
                emit_read_v5_telemetry(st, slot_id);
            }
            sbuf->release();
        });
    }
}

std::vector<array> ReadSlab::jvp(const std::vector<array>& primals,
                                 const std::vector<array>& tangents,
                                 const std::vector<int>& argnums) {
    (void)tangents;
    std::vector<array> o;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            o.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return o;
}

std::vector<array> ReadSlab::vjp(const std::vector<array>& primals,
                                 const std::vector<array>& cotangents,
                                 const std::vector<int>& argnums,
                                 const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            vjps.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> ReadSlab::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[ReadSlab] vmap not implemented.");
}

// ── Batch status buffer (shared Metal memory, B bytes) ─────────────────────

BatchStatusBuffer::BatchStatusBuffer(MTL::Device* dev, uint32_t b) : B_(b) {
    if (dev == nullptr || b == 0) {
        throw std::runtime_error("BatchStatusBuffer: invalid args");
    }
    buf_ = dev->newBuffer(
        static_cast<NS::UInteger>(b) * sizeof(uint8_t),
        MTL::ResourceStorageModeShared);
    if (buf_ == nullptr) {
        throw std::runtime_error("BatchStatusBuffer: allocation failed");
    }
    std::memset(buf_->contents(), 0, static_cast<size_t>(b));
}

BatchStatusBuffer::~BatchStatusBuffer() {
    if (buf_ != nullptr) {
        buf_->release();
        buf_ = nullptr;
    }
}

// ── ReadSlabBatchedOp ───────────────────────────────────────────────────────

ReadSlabBatchedOp::ReadSlabBatchedOp(
    MTL::Buffer* slab,
    std::vector<uint32_t> slot_indices,
    std::shared_ptr<BatchStatusBuffer> status_ctx,
    Stream s,
    uint32_t latent_elems,
    uint32_t stride,
    uint32_t back_epoch_off)
    : Primitive(s),
      slab_buf_(slab),
      slot_indices_(std::move(slot_indices)),
      status_ctx_(std::move(status_ctx)),
      latent_elems_(latent_elems),
      stride_(stride),
      back_epoch_off_(back_epoch_off) {}

ReadSlabBatchedOp::~ReadSlabBatchedOp() = default;

std::vector<Shape> ReadSlabBatchedOp::output_shapes(
    const std::vector<array>& inputs) {
    (void)inputs;
    const auto B = static_cast<mlx::core::ShapeElem>(slot_indices_.size());
    return {Shape{B, 1, static_cast<mlx::core::ShapeElem>(latent_elems_)}};
}

void ReadSlabBatchedOp::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
    (void)inputs;
    auto& out = outputs[0];
    const uint32_t B = static_cast<uint32_t>(slot_indices_.size());
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    uint16_t* h_dst = out.data<uint16_t>();
    uint8_t* st = static_cast<uint8_t*>(status_ctx_->buf()->contents());
    char* slab_base = static_cast<char*>(slab_buf_->contents());

    const size_t scent_b = static_cast<size_t>(latent_elems_) * 2u;
    for (uint32_t b = 0; b < B; ++b) {
        uint32_t si = slot_indices_[b];
        uint16_t* row = h_dst + b * latent_elems_;
        if (si == HIVECLAW_SENTINEL_SLOT) {
            std::memset(row, 0, scent_b);
            st[b] = 0;
            continue;
        }
        size_t sb = hclw_slot_base(si, stride_);
        char* hdr = slab_base + sb;

        auto load_u32 = [hdr](size_t off) -> uint32_t {
            return reinterpret_cast<std::atomic<uint32_t>*>(hdr + off)->load(
                std::memory_order_acquire);
        };

        const uint32_t fe = load_u32(OFF_S_FRONT_EPOCH);
        const uint32_t be_pre = load_u32(static_cast<size_t>(back_epoch_off_));
        std::memcpy(row, hdr + HCLW_SLOT_HDR, scent_b);
        const uint32_t fe2 = load_u32(OFF_S_FRONT_EPOCH);
        const uint32_t be2 = load_u32(static_cast<size_t>(back_epoch_off_));
        if (fe != be_pre || fe2 != fe || be2 != be_pre || fe2 != be2) {
            std::memset(row, 0, scent_b);
            st[b] = 1;
        } else {
            st[b] = 0;
        }
    }
    emit_torn_batch_telemetry(st, B, slot_indices_);
}

void ReadSlabBatchedOp::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
    if (!hiveclaw_gpu_batch_read_enabled()) {
        eval_cpu(inputs, outputs);
        return;
    }
    (void)inputs;
    auto& out = outputs[0];
    const uint32_t B = static_cast<uint32_t>(slot_indices_.size());

    auto& d = mlx::core::metal::device(stream().device);
    MTL::Device* mtl_dev = d.mtl_device();
    if (mtl_dev == nullptr) {
        eval_cpu(inputs, outputs);
        return;
    }

    const size_t idx_nbytes = slot_indices_.size() * sizeof(uint32_t);
    MTL::Buffer* idx_buf =
        mtl_dev->newBuffer(static_cast<NS::UInteger>(idx_nbytes), MTL::ResourceStorageModeShared);
    if (idx_buf == nullptr) {
        eval_cpu(inputs, outputs);
        return;
    }
    std::memcpy(idx_buf->contents(), slot_indices_.data(), idx_nbytes);

    const size_t staging_nbytes =
        static_cast<size_t>(B) * static_cast<size_t>(latent_elems_) * sizeof(uint16_t);
    MTL::Buffer* staging =
        mtl_dev->newBuffer(static_cast<NS::UInteger>(staging_nbytes), MTL::ResourceStorageModeShared);
    if (staging == nullptr) {
        idx_buf->release();
        eval_cpu(inputs, outputs);
        return;
    }
    std::memset(staging->contents(), 0, staging_nbytes);

    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    auto& enc = d.get_command_encoder(stream().index);
    auto* lib = d.get_library(
        "hiveclaw_read_slab_v5_batched",
        [] { return std::string(READ_V5_BATCHED_MSL); });
    auto* kernel = d.get_kernel("read_slab_v5_batched", lib);
    enc.set_compute_pipeline_state(kernel);

    struct SlabParamsEnc {
        uint32_t global_hdr;
        uint32_t stride;
        uint32_t slot_hdr;
        uint32_t latent_elems;
        uint32_t back_epoch_off;
    };
    const uint8_t* gp = static_cast<const uint8_t*>(slab_buf_->contents());
    SlabParamsEnc P{
        static_cast<uint32_t>(HCLW_GLOBAL_HDR),
        *reinterpret_cast<const uint32_t*>(gp + OFF_G_STRIDE_V6),
        static_cast<uint32_t>(HCLW_SLOT_HDR),
        latent_elems_,
        back_epoch_off_};
    enc.set_bytes(&P, sizeof(P), 4);

    // Shared staging: avoids binding MLX output arrays directly with IOSurface-backed slab
    // (kIOGPUCommandBufferCallbackErrorInvalidResource on some driver paths).
    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_buffer(staging, 1, 0);
    enc.set_buffer(status_ctx_->buf(), 2, 0);
    enc.set_buffer(idx_buf, 3, 0);

    enc.dispatch_threadgroups(
        MTL::Size(1, 1, static_cast<NS::UInteger>(B)),
        MTL::Size(32, 1, 1));

    MTL::CommandBuffer* cb = d.get_command_buffer(stream().index);
    if (cb == nullptr) {
        staging->release();
        idx_buf->release();
        eval_cpu(inputs, outputs);
        return;
    }
    cb->commit();
    cb->waitUntilCompleted();

    std::memcpy(out.data<uint16_t>(), staging->contents(), staging_nbytes);

    uint8_t* st = static_cast<uint8_t*>(status_ctx_->buf()->contents());
    emit_torn_batch_telemetry(st, B, slot_indices_);

    staging->release();
    idx_buf->release();
}

std::vector<array> ReadSlabBatchedOp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
    (void)tangents;
    std::vector<array> o;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            o.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return o;
}

std::vector<array> ReadSlabBatchedOp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            vjps.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> ReadSlabBatchedOp::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[ReadSlabBatchedOp] vmap not implemented.");
}

// ── CopyBatchStatusOp ──────────────────────────────────────────────────────

CopyBatchStatusOp::CopyBatchStatusOp(
    std::shared_ptr<BatchStatusBuffer> status_ctx,
    Stream s)
    : UnaryPrimitive(s), status_ctx_(std::move(status_ctx)) {}

std::vector<Shape> CopyBatchStatusOp::output_shapes(
    const std::vector<array>& inputs) {
    (void)inputs;
    return {Shape{static_cast<mlx::core::ShapeElem>(status_ctx_->batch_size())}};
}

void CopyBatchStatusOp::eval_cpu(const std::vector<array>& inputs, array& out) {
    (void)inputs;
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    std::memcpy(
        out.data<uint8_t>(),
        status_ctx_->buf()->contents(),
        static_cast<size_t>(status_ctx_->batch_size()));
}

void CopyBatchStatusOp::eval_gpu(const std::vector<array>& inputs, array& out) {
    eval_cpu(inputs, out);
}

std::vector<array> CopyBatchStatusOp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
    (void)primals;
    std::vector<array> o;
    for (auto a : argnums) {
        (void)a;
        o.push_back(zeros_like(tangents[0], stream()));
    }
    return o;
}

std::vector<array> CopyBatchStatusOp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (auto a : argnums) {
        (void)a;
        vjps.push_back(zeros_like(primals[0], stream()));
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> CopyBatchStatusOp::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[CopyBatchStatusOp] vmap not implemented.");
}

// ── WriteSlabBatchedOp ─────────────────────────────────────────────────────

WriteSlabBatchedOp::WriteSlabBatchedOp(
    MTL::Buffer* slab,
    std::vector<uint32_t> slot_indices,
    std::shared_ptr<BatchStatusBuffer> status_ctx,
    Stream s,
    uint32_t latent_elems,
    uint32_t stride,
    uint32_t back_epoch_off)
    : Primitive(s),
      slab_buf_(slab),
      slot_indices_(std::move(slot_indices)),
      status_ctx_(std::move(status_ctx)),
      latent_elems_(latent_elems),
      stride_(stride),
      back_epoch_off_(back_epoch_off) {}

WriteSlabBatchedOp::~WriteSlabBatchedOp() = default;

std::vector<Shape> WriteSlabBatchedOp::output_shapes(
    const std::vector<array>& inputs) {
    return {inputs[0].shape()};
}

void WriteSlabBatchedOp::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
    auto& in = inputs[0];
    auto& out = outputs[0];
    const uint32_t B = static_cast<uint32_t>(slot_indices_.size());
    uint8_t* st = static_cast<uint8_t*>(status_ctx_->buf()->contents());
    const uint16_t* src = in.data<uint16_t>();
    char* slab_base = static_cast<char*>(slab_buf_->contents());
    const size_t scent_b = static_cast<size_t>(latent_elems_) * 2u;

    for (uint32_t b = 0; b < B; ++b) {
        uint32_t si = slot_indices_[b];
        if (si == HIVECLAW_SENTINEL_SLOT) {
            st[b] = 0;
            continue;
        }
        size_t sb = hclw_slot_base(si, stride_);
        char* sh = slab_base + sb;
        auto* claim = reinterpret_cast<std::atomic<uint32_t>*>(sh + OFF_S_CLAIM_FLAG);
        const uint32_t w = claim->load(std::memory_order_relaxed);
        if (hclw_slot_status(w) != HCLW_SLOT_STATUS_CLAIMED) {
            st[b] = 2;
            continue;
        }
        st[b] = 0;
        void* dst = sh + HCLW_SLOT_HDR;
        const uint16_t* row = src + b * latent_elems_;
        auto* fe = reinterpret_cast<std::atomic<uint32_t>*>(sh + OFF_S_FRONT_EPOCH);
        uint32_t e = fe->load(std::memory_order_relaxed) + 1u;
        fe->store(e, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(dst, row, scent_b);
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<uint32_t*>(sh + static_cast<size_t>(back_epoch_off_)) = e;
        *reinterpret_cast<uint64_t*>(sh + OFF_S_LAST_CLAIM_MACH) = mach_absolute_time();
    }
    out.copy_shared_buffer(in);
}

void WriteSlabBatchedOp::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
    if (!hiveclaw_gpu_batch_write_enabled()) {
        eval_cpu(inputs, outputs);
        return;
    }
    auto& in = inputs[0];
    auto& out = outputs[0];
    const uint32_t B = static_cast<uint32_t>(slot_indices_.size());

    auto& d = mlx::core::metal::device(stream().device);
    MTL::Device* mtl_dev = d.mtl_device();
    if (mtl_dev == nullptr) {
        eval_cpu(inputs, outputs);
        return;
    }

    const size_t idx_nbytes = slot_indices_.size() * sizeof(uint32_t);
    MTL::Buffer* idx_buf =
        mtl_dev->newBuffer(static_cast<NS::UInteger>(idx_nbytes), MTL::ResourceStorageModeShared);
    if (idx_buf == nullptr) {
        eval_cpu(inputs, outputs);
        return;
    }
    std::memcpy(idx_buf->contents(), slot_indices_.data(), idx_nbytes);

    auto& enc = d.get_command_encoder(stream().index);
    auto* lib = d.get_library(
        "hiveclaw_write_slab_v5_batched",
        [] { return std::string(WRITE_V5_BATCHED_MSL); });
    auto* kernel = d.get_kernel("write_slab_v5_batched", lib);
    enc.set_compute_pipeline_state(kernel);

    struct SlabParamsEnc {
        uint32_t global_hdr;
        uint32_t stride;
        uint32_t slot_hdr;
        uint32_t latent_elems;
        uint32_t back_epoch_off;
    };
    const uint8_t* gp = static_cast<const uint8_t*>(slab_buf_->contents());
    SlabParamsEnc P{
        static_cast<uint32_t>(HCLW_GLOBAL_HDR),
        *reinterpret_cast<const uint32_t*>(gp + OFF_G_STRIDE_V6),
        static_cast<uint32_t>(HCLW_SLOT_HDR),
        latent_elems_,
        back_epoch_off_};
    enc.set_bytes(&P, sizeof(P), 4);

    // status_out is device uint8_t* (read-write in MSL), matching Ironclad input_rw_status intent
    // at the native layer (no const on status buffer).
    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_input_array(in, 1);
    enc.set_buffer(status_ctx_->buf(), 2, 0);
    enc.set_buffer(idx_buf, 3, 0);

    enc.dispatch_threadgroups(
        MTL::Size(1, 1, static_cast<NS::UInteger>(B)),
        MTL::Size(32, 1, 1));

    MTL::CommandBuffer* cb = d.get_command_buffer(stream().index);
    if (cb == nullptr) {
        idx_buf->release();
        eval_cpu(inputs, outputs);
        return;
    }
    cb->commit();
    cb->waitUntilCompleted();

    out.copy_shared_buffer(in);
    idx_buf->release();
}

std::vector<array> WriteSlabBatchedOp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
    (void)tangents;
    std::vector<array> o;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            o.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return o;
}

std::vector<array> WriteSlabBatchedOp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            vjps.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> WriteSlabBatchedOp::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[WriteSlabBatchedOp] vmap not implemented.");
}
