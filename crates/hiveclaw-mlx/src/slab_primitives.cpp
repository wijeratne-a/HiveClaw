#include "slab_bridge.h"
#include "slab_primitives.h"
#include "slab_layout.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mach/mach_time.h>
#include <stdexcept>

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
