#include "slab_kernels.h"
#include "slab_layout.h"
#include "slab_primitives.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mach/mach_time.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

using mlx::core::Shape;
using mlx::core::Stream;
using mlx::core::array;
using mlx::core::default_stream;
using mlx::core::depends;
using mlx::core::Device;
using mlx::core::Primitive;
using mlx::core::zeros_like;

ClaimSlabTask::ClaimSlabTask(
    MTL::Buffer* slab,
    uint32_t agent_id,
    uint32_t stride,
    uint32_t n_slots,
    Stream s)
    : UnaryPrimitive(s),
      slab_buf_(slab),
      agent_id_(agent_id),
      stride_(stride),
      n_slots_(n_slots) {}

std::vector<Shape> ClaimSlabTask::output_shapes(
    const std::vector<array>& inputs) {
    (void)inputs;
    return {Shape{}};
}

void ClaimSlabTask::eval_cpu(const std::vector<array>& inputs, array& out) {
    auto& cand = inputs[0];
    if (cand.dtype() != mlx::core::int32) {
        throw std::runtime_error("ClaimSlabTask: candidates must be int32");
    }
    out.set_data(mlx::core::allocator::malloc(sizeof(int32_t)));
    int32_t* po = out.data<int32_t>();
    po[0] = -1;

    char* base = static_cast<char*>(slab_buf_->contents());
    const int32_t* c = cand.data<int32_t>();
    const size_t k = cand.size();
    const uint32_t desired = hclw_pack_claimed(agent_id_ & 0xFFFFu);
    for (size_t i = 0; i < k; i++) {
        int s = c[i];
        if (s < 0 || s >= static_cast<int>(n_slots_)) {
            continue;
        }
        size_t off =
            hclw_slot_base(static_cast<size_t>(s), stride_) + OFF_S_CLAIM_FLAG;
        auto* state = reinterpret_cast<std::atomic<uint32_t>*>(base + off);
        uint32_t expected = 0;
        if (state->compare_exchange_strong(
                expected, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            *reinterpret_cast<uint64_t*>(
                base + hclw_slot_base(static_cast<size_t>(s), stride_) +
                OFF_S_LAST_CLAIM_MACH) =
                mach_absolute_time();
            po[0] = s;
            return;
        }
    }
}

void ClaimSlabTask::eval_gpu(const std::vector<array>& inputs, array& out) {
    eval_cpu(inputs, out);
}

std::vector<array> ClaimSlabTask::jvp(const std::vector<array>& primals,
                                      const std::vector<array>& tangents,
                                      const std::vector<int>& argnums) {
    (void)primals;
    std::vector<array> o;
    for (int a : argnums) {
        (void)a;
        o.push_back(zeros_like(tangents[0], stream()));
    }
    return o;
}

std::vector<array> ClaimSlabTask::vjp(const std::vector<array>& primals,
                                      const std::vector<array>& cotangents,
                                      const std::vector<int>& argnums,
                                      const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (int a : argnums) {
        (void)a;
        vjps.push_back(zeros_like(primals[0], stream()));
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> ClaimSlabTask::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[ClaimSlabTask] vmap not implemented.");
}

InhibitSlab::InhibitSlab(MTL::Buffer* slab,
                         uint32_t slot_index,
                         uint32_t agent_id,
                         uint32_t stride,
                         uint32_t n_slots,
                         Stream s)
    : Primitive(s),
      slab_buf_(slab),
      slot_index_(slot_index),
      agent_id_(agent_id),
      stride_(stride),
      n_slots_(n_slots) {}

std::vector<Shape> InhibitSlab::output_shapes(
    const std::vector<array>& inputs) {
    (void)inputs;
    return {Shape{}};
}

void InhibitSlab::eval_cpu(const std::vector<array>& inputs,
                           std::vector<array>& outputs) {
    (void)inputs;
    if (slot_index_ >= n_slots_) {
        throw std::runtime_error("InhibitSlab: slot_index out of range");
    }
    char* base = static_cast<char*>(slab_buf_->contents());
    size_t sb = hclw_slot_base(slot_index_, stride_);
    std::memset(base + sb, 0, static_cast<size_t>(stride_));
    reinterpret_cast<std::atomic<uint32_t>*>(base + sb + OFF_S_CLAIM_FLAG)
        ->store(HCLW_SLOT_STATUS_INHIBITED, std::memory_order_release);

    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(sizeof(int32_t)));
    out.data<int32_t>()[0] = 0;
}

void InhibitSlab::eval_gpu(const std::vector<array>& inputs,
                           std::vector<array>& outputs) {
    eval_cpu(inputs, outputs);
}

std::vector<array> InhibitSlab::jvp(const std::vector<array>& primals,
                                    const std::vector<array>& tangents,
                                    const std::vector<int>& argnums) {
    (void)tangents;
    std::vector<array> o;
    for (int a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            o.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return o;
}

std::vector<array> InhibitSlab::vjp(const std::vector<array>& primals,
                                    const std::vector<array>& cotangents,
                                    const std::vector<int>& argnums,
                                    const std::vector<array>& /*outputs*/) {
    (void)cotangents;
    std::vector<array> vjps;
    for (int a : argnums) {
        (void)a;
        if (a < static_cast<int>(primals.size())) {
            vjps.push_back(zeros_like(primals[static_cast<size_t>(a)], stream()));
        }
    }
    return vjps;
}

std::pair<std::vector<array>, std::vector<int>> InhibitSlab::vmap(
    const std::vector<array>& /*inputs*/,
    const std::vector<int>& /*axes*/) {
    throw std::runtime_error("[InhibitSlab] vmap not implemented.");
}

array SlabHandle::claim(array candidate_indices,
                        uint32_t agent_id,
                        std::optional<array> dep) {
    Stream s = default_stream(Device::cpu);
    auto prim = std::make_shared<ClaimSlabTask>(
        slab_buf_, agent_id, stride_, n_slots_, s);
    std::vector<array> inputs = {std::move(candidate_indices)};
    array out(Shape{},
              mlx::core::int32,
              std::static_pointer_cast<Primitive>(prim),
              inputs);
    if (dep) {
        return depends({out}, {*dep})[0];
    }
    return out;
}

array SlabHandle::inhibit(uint32_t slot_index,
                          uint32_t agent_id,
                          std::optional<array> dep) {
    Stream s = default_stream(Device::cpu);
    if (dep && (*dep).has_primitive()) {
        s = (*dep).primitive().stream();
    }
    auto prim = std::make_shared<InhibitSlab>(
        slab_buf_, slot_index, agent_id, stride_, n_slots_, s);
    std::vector<array> inputs;
    if (dep) {
        inputs.push_back(*dep);
    }
    return array(Shape{},
                 mlx::core::int32,
                 std::static_pointer_cast<Primitive>(prim),
                 inputs);
}

void SlabHandle::release_slot(uint32_t slot_index) {
    if (slot_index >= n_slots_) {
        throw std::runtime_error("release_slot: slot_index out of range");
    }
    char* base = static_cast<char*>(slab_buf_->contents());
    size_t off = hclw_slot_base(slot_index, stride_) + OFF_S_CLAIM_FLAG;
    auto* st = reinterpret_cast<std::atomic<uint32_t>*>(base + off);
    st->store(0u, std::memory_order_release);
}

std::vector<std::pair<bool, uint32_t>> SlabHandle::get_slot_states() const {
    char* base = static_cast<char*>(slab_buf_->contents());
    std::vector<std::pair<bool, uint32_t>> out;
    out.reserve(n_slots_);
    for (size_t i = 0; i < static_cast<size_t>(n_slots_); ++i) {
        size_t b = hclw_slot_base(i, stride_);
        auto* st = reinterpret_cast<std::atomic<uint32_t>*>(base + b + OFF_S_CLAIM_FLAG);
        uint32_t w = st->load(std::memory_order_relaxed);
        bool claimed = (hclw_slot_status(w) == HCLW_SLOT_STATUS_CLAIMED);
        uint32_t owner = hclw_slot_owner16(w);
        out.emplace_back(claimed, owner);
    }
    return out;
}
