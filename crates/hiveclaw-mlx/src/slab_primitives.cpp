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

// v5: 32 threads × 8 uint16 = 256 latent elems; torn → zeros + status byte.
static const char* READ_V5_MSL = R"(
#include <metal_stdlib>
using namespace metal;

constant uint HCLW_GLOBAL_HDR = 4096u;
constant uint HCLW_SLOT_STRIDE = 640u;
constant uint HCLW_SLOT_HDR = 64u;
constant uint OFF_S_FRONT_EPOCH = 12u;
constant uint HCLW_OFF_SLOT_BACK_EPOCH = 576u;

kernel void read_slab_v5(
    const device uint8_t* slab [[buffer(0)]],
    device uint16_t* h_out [[buffer(1)]],
    device uint8_t* status [[buffer(2)]],
    constant uint32_t& slot_index [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint slot_base = HCLW_GLOBAL_HDR + slot_index * HCLW_SLOT_STRIDE;
  threadgroup uint front_ep;
  threadgroup uint back_ep;
  threadgroup uint torn;

  if (tid == 0u) {
    front_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + OFF_S_FRONT_EPOCH);
    torn = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint payload_base = slot_base + HCLW_SLOT_HDR;
  for (uint k = 0u; k < 8u; k++) {
    uint i = tid * 8u + k;
    uint16_t v = *reinterpret_cast<const device uint16_t*>(slab + payload_base + i * 2u);
    h_out[i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0u) {
    back_ep = *reinterpret_cast<const device uint32_t*>(slab + slot_base + HCLW_OFF_SLOT_BACK_EPOCH);
    if (front_ep != back_ep) {
      torn = 1u;
      status[0] = 1u;
    } else {
      status[0] = 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (torn != 0u) {
    for (uint k = 0u; k < 8u; k++) {
      uint i = tid * 8u + k;
      h_out[i] = 0u;
    }
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

static void validate_write_v5_stamped(const array& in) {
    if (in.dtype() != bfloat16) {
        throw std::invalid_argument("WriteSlab v5: input must be bfloat16");
    }
    const auto& sh = in.shape();
    if (sh.size() != 3 || sh[0] != 1 || sh[1] != 1 || sh[2] != static_cast<int>(HCLW_SCENT_ELEMS)) {
        throw std::invalid_argument("WriteSlab v5: input must be [1,1,256] bfloat16");
    }
    if (in.nbytes() != HCLW_SCENT_BYTES) {
        throw std::invalid_argument("WriteSlab v5: expected 512-byte payload");
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

array SlabHandle::write_slot_v5(uint32_t slot_index,
                                array latent,
                                std::optional<array> dep) {
    validate_write_v5_stamped(latent);
    return write_slot(slot_index, std::move(latent), std::move(dep));
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

array SlabHandle::read_slot_v5(uint32_t slot_index, std::optional<array> dep) {
    if (slot_index >= HCLW_N_SLOTS) {
        throw std::runtime_error("read_slot_v5: slot_index out of range");
    }
    Stream s = default_stream(Device::gpu);
    if (dep && dep->has_primitive()) {
        s = dep->primitive().stream();
    }
    auto prim = std::make_shared<ReadSlab>(
        slab_buf_,
        slot_payload(slot_index),
        Shape{1, 1, static_cast<mlx::core::ShapeElem>(HCLW_SCENT_ELEMS)},
        s,
        slot_index);
    std::vector<array> inputs;
    if (dep) {
        inputs.push_back(*dep);
    }
    return array(
        Shape{1, 1, static_cast<mlx::core::ShapeElem>(HCLW_SCENT_ELEMS)},
        bfloat16,
        std::static_pointer_cast<Primitive>(prim),
        inputs);
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
                     uint32_t stamp_slot_index)
    : UnaryPrimitive(s),
      slab_buf_(slab),
      byte_offset_(byte_offset),
      num_bytes_(num_bytes),
      stamp_slot_index_(stamp_slot_index) {}

void WriteSlab::eval_cpu(const std::vector<array>& inputs, array& out) {
    auto& in = inputs[0];
    if (stamp_slot_index_ != 0xFFFFFFFFu) {
        validate_write_v5_stamped(in);
    } else if (in.nbytes() != num_bytes_) {
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
                   std::optional<uint32_t> v5_slot_for_epoch)
    : Primitive(s),
      slab_buf_(slab),
      byte_offset_(byte_offset),
      shape_(std::move(shape)),
      v5_slot_for_epoch_(v5_slot_for_epoch) {
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
    size_t sb = slot_base(static_cast<size_t>(si));
    char* hdr = slab_base + sb;

    auto load_u32 = [hdr](size_t off) -> uint32_t {
        return reinterpret_cast<std::atomic<uint32_t>*>(hdr + off)->load(
            std::memory_order_acquire);
    };

    const uint32_t fe = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be_pre = load_u32(HCLW_OFF_SLOT_BACK_EPOCH);
    std::memcpy(h_dst, hdr + HCLW_SLOT_HDR, HCLW_SCENT_BYTES);
    const uint32_t fe2 = load_u32(OFF_S_FRONT_EPOCH);
    const uint32_t be2 = load_u32(HCLW_OFF_SLOT_BACK_EPOCH);
    if (fe != be_pre || fe2 != fe || be2 != be_pre || fe2 != be2) {
        std::memset(h_dst, 0, HCLW_SCENT_BYTES);
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
