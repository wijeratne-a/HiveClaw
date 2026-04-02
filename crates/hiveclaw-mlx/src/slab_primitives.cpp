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
#include <sstream>
#include <stdexcept>
#include <string>

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

// PR2: fused epoch + L2 clamp + steering (256 threads × 8 elems = 2048).
static const char* FUSED_STEER_MSL = R"(
#include <metal_stdlib>
using namespace metal;

constant uint HCLW_GLOBAL_HDR = 128u;
constant uint HCLW_SLOT_STRIDE = 4224u;
constant uint HCLW_SLOT_HDR = 64u;
constant uint HCLW_SCENT_ELEMS = 2048u;
constant uint OFF_S_FRONT_EPOCH = 12u;
constant uint HCLW_OFF_SLOT_BACK_EPOCH = 4160u;

kernel void fused_steer_bf16(
    const device uint8_t* slab [[buffer(0)]],
    const device uint16_t* h_step [[buffer(1)]],
    device uint16_t* h_out [[buffer(2)]],
    device uint8_t* status [[buffer(3)]],
    constant uint32_t& slot_index [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint slot_base = HCLW_GLOBAL_HDR + slot_index * HCLW_SLOT_STRIDE;

  threadgroup uint front_ep;
  threadgroup uint back_ep;
  threadgroup float norm2_partial[256];
  threadgroup float scale_tg;

  if (tid == 0) {
    front_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + OFF_S_FRONT_EPOCH);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float scent_vals[8];
  float h_vals[8];
  uint payload_base = slot_base + HCLW_SLOT_HDR;
  for (uint k = 0; k < 8u; k++) {
    uint i = tid * 8u + k;
    uint16_t s16 = *reinterpret_cast<const device uint16_t*>(slab + payload_base + i * 2u);
    scent_vals[k] = as_type<float>((uint32_t)s16 << 16);
    uint16_t h16 = h_step[i];
    h_vals[k] = as_type<float>((uint32_t)h16 << 16);
  }

  if (tid == 0) {
    back_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + HCLW_OFF_SLOT_BACK_EPOCH);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (front_ep != back_ep) {
    if (tid == 0) {
      status[0] = 1u;
    }
    for (uint k = 0; k < 8u; k++) {
      uint i = tid * 8u + k;
      h_out[i] = h_step[i];
    }
    return;
  }

  float local_norm2 = 0.f;
  for (uint k = 0; k < 8u; k++) {
    float c = alpha * scent_vals[k];
    local_norm2 += c * c;
  }
  norm2_partial[tid] = local_norm2;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      norm2_partial[tid] += norm2_partial[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    float norm = sqrt(norm2_partial[0]);
    float sc = 1.f;
    if (norm > 2.f) {
      sc = 2.f / norm;
      status[0] = 2u;
    }
    scale_tg = sc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sc = scale_tg;

  for (uint k = 0; k < 8u; k++) {
    uint i = tid * 8u + k;
    float c = alpha * scent_vals[k];
    float result = h_vals[k] + c * sc;
    uint32_t u = as_type<uint32_t>(result);
    h_out[i] = (uint16_t)(u >> 16);
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

static void emit_fused_telemetry(uint8_t st, uint32_t slot_id) {
    if (!hiveclaw_telemetry_enabled() || st == 0) {
        return;
    }
    const char* ev = (st == 1) ? "torn_epoch_skip" : "poison_clamp";
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    std::ostringstream oss;
    oss << "{\"event\":\"" << ev << "\",\"slot_id\":" << slot_id << ",\"ts_ns\":" << ns
        << "}\n";
    std::cerr << oss.str();
}

static float bf16_u16_to_f32(uint16_t u) {
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_bf16_u16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

static void validate_fused_h_step(const array& h) {
    if (h.dtype() != bfloat16) {
        throw std::invalid_argument("fused_steer: h_step must be bfloat16");
    }
    const auto& sh = h.shape();
    if (sh.size() != 3 || sh[0] != 1 || sh[1] != 1 || sh[2] != static_cast<int>(HCLW_SCENT_ELEMS)) {
        throw std::invalid_argument("fused_steer: h_step shape must be [1,1,2048]");
    }
}

} // namespace

SlabHandle::SlabHandle(uint32_t surface_id) : surface_id_(surface_id) {
    auto* mtl_dev = mlx::core::metal::device(Device::gpu).mtl_device();
    void* p = create_slab_buffer(static_cast<void*>(mtl_dev), surface_id, HIVECLAW_SLAB_SIZE);
    slab_buf_ = static_cast<MTL::Buffer*>(p);
    if (!slab_buf_) {
        throw std::runtime_error("create_slab_buffer failed (IOSurfaceLookup or Metal buffer)");
    }
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
    if (slot_index >= HCLW_N_SLOTS) {
        throw std::runtime_error("write_slot: slot_index out of range");
    }
    const size_t byte_offset = slot_payload(slot_index);
    Stream s = default_stream(Device::cpu);
    const size_t nbytes = scent_c.nbytes();
    auto prim = std::make_shared<WriteSlab>(
        slab_buf_, byte_offset, nbytes, s, slot_index);
    std::vector<array> inputs = {std::move(scent_c)};
    array out(inputs[0].shape(), inputs[0].dtype(), std::static_pointer_cast<Primitive>(prim), inputs);
    if (dep) {
        return depends({out}, {*dep})[0];
    }
    return out;
}

array SlabHandle::write(size_t byte_offset,
                        array scent_c,
                        std::optional<array> dep) {
    Stream s = default_stream(Device::gpu);
    const size_t nbytes = scent_c.nbytes();
    auto prim = std::make_shared<WriteSlab>(slab_buf_, byte_offset, nbytes, s);
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
    if (slot_index >= HCLW_N_SLOTS) {
        throw std::runtime_error("read_slot: slot_index out of range");
    }
    return read(slot_payload(slot_index), std::move(shape), std::move(dep));
}

array SlabHandle::read(size_t byte_offset,
                       Shape shape,
                       std::optional<array> dep) {
    // Avoid passing Python mlx arrays through nanobind as C++ `array` for a stream hint
    // (that path can throw std::bad_cast when MLX/nanobind versions diverge). Use the
    // dependency's stream when present, else the default GPU stream.
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

array SlabHandle::fused_steer(uint32_t slot_index,
                              array h_step,
                              float alpha,
                              std::optional<array> dep) {
    if (slot_index >= HCLW_N_SLOTS) {
        throw std::runtime_error("fused_steer: slot_index out of range");
    }
    validate_fused_h_step(h_step);
    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    auto prim = std::make_shared<FusedSteerScent>(slab_buf_, slot_index, alpha, s);
    std::vector<array> inputs = {std::move(h_step)};
    if (dep) {
        inputs.push_back(*dep);
    }
    array out(
        inputs[0].shape(),
        bfloat16,
        std::static_pointer_cast<Primitive>(prim),
        inputs);
    return out;
}

WriteSlab::WriteSlab(MTL::Buffer* slab,
                     size_t byte_offset,
                     size_t num_bytes,
                     Stream s,
                     uint32_t stamp_slot_index)
    : UnaryPrimitive(s),
      slab_buf_(slab),
      byte_offset_(byte_offset),
      num_bytes_(num_bytes),
      stamp_slot_index_(stamp_slot_index) {}

void WriteSlab::eval_cpu(const std::vector<array>& inputs, array& out) {
    auto& in = inputs[0];
    if (in.nbytes() != num_bytes_) {
        throw std::runtime_error("WriteSlab: input size mismatch");
    }
    void* dst = static_cast<char*>(slab_buf_->contents()) + byte_offset_;
    if (stamp_slot_index_ != 0xFFFFFFFFu) {
        char* base = static_cast<char*>(slab_buf_->contents());
        size_t sb = slot_base(stamp_slot_index_);
        char* sh = base + sb;
        auto* fe = reinterpret_cast<std::atomic<uint32_t>*>(sh + OFF_S_FRONT_EPOCH);
        uint32_t e = fe->load(std::memory_order_relaxed) + 1u;
        fe->store(e, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(dst, in.data<const uint8_t>(), num_bytes_);
        std::atomic_thread_fence(std::memory_order_release);
        *reinterpret_cast<uint32_t*>(sh + static_cast<size_t>(HCLW_OFF_SLOT_BACK_EPOCH)) = e;
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
    std::vector<array> out;
    for (auto a : argnums) {
        (void)a;
        out.push_back(zeros_like(tangents[0], stream()));
    }
    return out;
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
                   Stream s)
    : Primitive(s), slab_buf_(slab), byte_offset_(byte_offset), shape_(std::move(shape)) {}

std::vector<Shape> ReadSlab::output_shapes(const std::vector<array>& inputs) {
    (void)inputs;
    return {shape_};
}

void ReadSlab::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    (void)inputs;
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
    void* src = static_cast<char*>(slab_buf_->contents()) + byte_offset_;
    std::memcpy(out.data<uint8_t>(), src, out.nbytes());
}

void ReadSlab::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
    (void)inputs;
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

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
}

std::vector<array> ReadSlab::jvp(const std::vector<array>& primals,
                                 const std::vector<array>& tangents,
                                 const std::vector<int>& argnums) {
    (void)tangents;
    std::vector<array> out;
    for (auto a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            out.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return out;
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

// --- FusedSteerScent (PR2) -------------------------------------------------

FusedSteerScent::FusedSteerScent(MTL::Buffer* slab,
                                 uint32_t slot_index,
                                 float alpha,
                                 Stream s)
    : Primitive(s),
      slab_buf_(slab),
      slot_index_(slot_index),
      alpha_(alpha) {
    // Must match the MLX stream's GPU; slab_buf_->device() can disagree with IOSurface wiring.
    MTL::Device* dev = mlx::core::metal::device(s.device).mtl_device();
    status_buf_ = dev->newBuffer(1, MTL::ResourceStorageModeShared);
    if (!status_buf_) {
        throw std::runtime_error("FusedSteerScent: status buffer allocation failed");
    }
}

FusedSteerScent::~FusedSteerScent() {
    if (status_buf_) {
        status_buf_->release();
        status_buf_ = nullptr;
    }
}

std::vector<Shape> FusedSteerScent::output_shapes(const std::vector<array>& inputs) {
    return {inputs[0].shape()};
}

void FusedSteerScent::eval_cpu(const std::vector<array>& inputs,
                               std::vector<array>& outputs) {
    auto& h_in = inputs[0];
    validate_fused_h_step(h_in);
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    char* slab_base = static_cast<char*>(slab_buf_->contents());
    const size_t sb = slot_base(slot_index_);
    char* hdr = slab_base + sb;

    auto load_u32 = [hdr](size_t off) -> uint32_t {
        return reinterpret_cast<std::atomic<uint32_t>*>(hdr + off)->load(
            std::memory_order_acquire);
    };

    const uint32_t fe = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be_pre = load_u32(HCLW_OFF_SLOT_BACK_EPOCH);

    const uint16_t* h_src = h_in.data<uint16_t>();
    uint16_t* h_dst = out.data<uint16_t>();

    if (fe != be_pre) {
        std::memcpy(h_dst, h_src, HCLW_SCENT_ELEMS * sizeof(uint16_t));
        emit_fused_telemetry(1, slot_index_);
        return;
    }

    const char* payload = hdr + HCLW_SLOT_HDR;
    float norm2 = 0.f;
    std::vector<float> contrib(HCLW_SCENT_ELEMS);
    for (size_t i = 0; i < HCLW_SCENT_ELEMS; i++) {
        uint16_t s16;
        std::memcpy(&s16, payload + i * 2, sizeof(s16));
        float scent = bf16_u16_to_f32(s16);
        float c = alpha_ * scent;
        contrib[i] = c;
        norm2 += c * c;
    }

    const uint32_t fe2 = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be2 = load_u32(HCLW_OFF_SLOT_BACK_EPOCH);
    if (fe2 != fe || be2 != be_pre || fe2 != be2) {
        std::memcpy(h_dst, h_src, HCLW_SCENT_ELEMS * sizeof(uint16_t));
        emit_fused_telemetry(1, slot_index_);
        return;
    }

    float norm = std::sqrt(norm2);
    float scale = 1.f;
    uint8_t st = 0;
    if (norm > 2.f) {
        scale = 2.f / norm;
        st = 2;
    }
    for (size_t i = 0; i < HCLW_SCENT_ELEMS; i++) {
        float hv = bf16_u16_to_f32(h_src[i]);
        float result = hv + contrib[i] * scale;
        h_dst[i] = f32_to_bf16_u16(result);
    }
    emit_fused_telemetry(st, slot_index_);
}

void FusedSteerScent::eval_gpu(const std::vector<array>& inputs,
                               std::vector<array>& outputs) {
    // Emergency rollback: HIVECLAW_FUSED_GPU=0 forces CPU path (no Metal dispatch).
    const char* fg = std::getenv("HIVECLAW_FUSED_GPU");
    if (fg != nullptr && std::string(fg) == "0") {
        eval_cpu(inputs, outputs);
        return;
    }

    auto& h_in = inputs[0];
    validate_fused_h_step(h_in);
    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    *static_cast<uint8_t*>(status_buf_->contents()) = 0;

    auto& d = mlx::core::metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    auto* lib = d.get_library("hiveclaw_fused_steer_bf16", [] {
        return std::string(FUSED_STEER_MSL);
    });
    auto* kernel = d.get_kernel("fused_steer_bf16", lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_input_array(h_in, 1);
    enc.set_output_array(out, 2);
    enc.set_buffer(status_buf_, 3, 0);
    enc.set_bytes(slot_index_, 4);
    enc.set_bytes(alpha_, 5);

    // MSL uses thread_index_in_threadgroup 0..255, 8 elems/thread, 256-lane reduction.
    enc.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));

    // Keep status_buf_ alive until the command buffer completes: ~FusedSteerScent may run
    // while GPU work is still in flight; an extra retain here is released in the handler.
    MTL::CommandBuffer* cb = d.get_command_buffer(stream().index);
    if (cb != nullptr) {
        status_buf_->retain();
        MTL::Buffer* sbuf = status_buf_;
        const uint32_t si = slot_index_;
        const bool tel = hiveclaw_telemetry_enabled();
        cb->addCompletedHandler([sbuf, si, tel](MTL::CommandBuffer* /*cmd*/) {
            if (tel) {
                uint8_t st = *static_cast<uint8_t*>(sbuf->contents());
                emit_fused_telemetry(st, si);
            }
            sbuf->release();
        });
    }
}

std::vector<array> FusedSteerScent::jvp(const std::vector<array>& primals,
                                        const std::vector<array>& tangents,
                                        const std::vector<int>& argnums) {
    std::vector<array> out;
    for (size_t i = 0; i < argnums.size(); i++) {
        int a = argnums[i];
        if (a == 0) {
            out.push_back(tangents[i]);
        } else {
            out.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return out;
}

std::vector<array> FusedSteerScent::vjp(const std::vector<array>& primals,
                                        const std::vector<array>& cotangents,
                                        const std::vector<int>& argnums,
                                        const std::vector<array>& /*outputs*/) {
    std::vector<array> vjps;
    for (int a : argnums) {
        if (a == 0) {
            vjps.push_back(cotangents[0]);
        } else {
            vjps.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> FusedSteerScent::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[FusedSteerScent] vmap not implemented.");
}
