#include "slab_kernels.h"
#include "slab_layout.h"
#include "slab_primitives.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
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

namespace {

static const char* CLAIM_INHIBIT_MSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void claim_slab_task(
    device void* slab_base [[buffer(0)]],
    device const int* candidates [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    constant uint& agent_id [[buffer(3)]],
    device int* out_idx [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid != 0) return;
  device uchar* base = (device uchar*)slab_base;
  out_idx[0] = -1;
  const uint global_hdr = 128u;
  const uint slot_stride = 8256u;
  const int nslots = 32;
  for (uint i = 0u; i < k; i++) {
    int s = candidates[i];
    if (s < 0 || s >= nslots) continue;
    uint slot_off = global_hdr + (uint)s * slot_stride;
    device volatile atomic_uint* flag =
        (device volatile atomic_uint*)(base + slot_off);
    uint expected = 0u;
    bool won = atomic_compare_exchange_weak_explicit(
        flag, &expected, 1u, memory_order_relaxed, memory_order_relaxed);
    if (won) {
      device uint* owner = (device uint*)(base + slot_off + 4u);
      *owner = agent_id;
      out_idx[0] = s;
      return;
    }
  }
}

kernel void inhibit_slot(
    device void* slab_base [[buffer(0)]],
    constant uint& slot_index [[buffer(1)]],
    constant uint& agent_id [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
  (void)agent_id;
  device uchar* base = (device uchar*)slab_base;
  const uint global_hdr = 128u;
  const uint slot_stride = 8256u;
  uint slot_off = global_hdr + slot_index * slot_stride;
  // MSL has no bfloat16_t; payload is bf16 bits — match CPU path (uint16_t zeros).
  device ushort* payload = (device ushort*)(base + slot_off + 64u);
  payload[idx] = 0u;
  if (idx == 0u) {
    device volatile atomic_uint* claim =
        (device volatile atomic_uint*)(base + slot_off);
    atomic_store_explicit(claim, 0u, memory_order_relaxed);
    device uint* watchdog = (device uint*)(base + slot_off + 12u);
    *watchdog = (*watchdog) | 1u;
    device float* inh_clk = (device float*)(base + slot_off + 16u);
    device float* zeta = (device float*)(base + 16u);
    *inh_clk = *zeta;
  }
}
)";

} // namespace

ClaimSlabTask::ClaimSlabTask(MTL::Buffer* slab, uint32_t agent_id, Stream s)
    : UnaryPrimitive(s), slab_buf_(slab), agent_id_(agent_id) {}

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
    for (size_t i = 0; i < k; i++) {
        int s = c[i];
        if (s < 0 || s >= static_cast<int>(HCLW_N_SLOTS)) {
            continue;
        }
        size_t off = slot_base(static_cast<size_t>(s)) + OFF_S_CLAIM_FLAG;
        auto* flag = reinterpret_cast<std::atomic<uint32_t>*>(base + off);
        uint32_t expected = 0;
        if (flag->compare_exchange_strong(
                expected, 1u, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            *reinterpret_cast<uint32_t*>(
                base + slot_base(static_cast<size_t>(s)) + OFF_S_OWNER_ID) = agent_id_;
            po[0] = s;
            return;
        }
    }
}

void ClaimSlabTask::eval_gpu(const std::vector<array>& inputs, array& out) {
    auto& cand = inputs[0];
    if (cand.dtype() != mlx::core::int32) {
        throw std::runtime_error("ClaimSlabTask: candidates must be int32");
    }

    out.set_data(mlx::core::allocator::malloc(sizeof(int32_t)));
    {
        int32_t* po = out.data<int32_t>();
        po[0] = -1;
    }

    auto& d = mlx::core::metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    auto* lib = d.get_library("hiveclaw_claim_inhibit", [] {
        return std::string(CLAIM_INHIBIT_MSL);
    });
    auto* kernel = d.get_kernel("claim_slab_task", lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_input_array(cand, 1);
    uint32_t k = static_cast<uint32_t>(cand.size());
    enc.set_bytes(k, 2);
    enc.set_bytes(agent_id_, 3);
    enc.set_output_array(out, 4);

    enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
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
                         Stream s)
    : Primitive(s),
      slab_buf_(slab),
      slot_index_(slot_index),
      agent_id_(agent_id) {}

std::vector<Shape> InhibitSlab::output_shapes(
    const std::vector<array>& inputs) {
    (void)inputs;
    return {Shape{}};
}

void InhibitSlab::eval_cpu(const std::vector<array>& inputs,
                           std::vector<array>& outputs) {
    (void)inputs;
    if (slot_index_ >= HCLW_N_SLOTS) {
        throw std::runtime_error("InhibitSlab: slot_index out of range");
    }
    char* base = static_cast<char*>(slab_buf_->contents());
    size_t sb = slot_base(slot_index_);
    auto* claim = reinterpret_cast<std::atomic<uint32_t>*>(base + sb + OFF_S_CLAIM_FLAG);
    claim->store(0u, std::memory_order_release);
    uint32_t* watchdog = reinterpret_cast<uint32_t*>(base + sb + OFF_S_WATCHDOG_FLAGS);
    *watchdog |= 0x1u;
    float zeta = *reinterpret_cast<float*>(base + OFF_G_ZETA_T);
    *reinterpret_cast<float*>(base + sb + OFF_S_LAST_INHIBIT_CLK) = zeta;

    uint16_t* payload = reinterpret_cast<uint16_t*>(base + slot_payload(slot_index_));
    for (size_t j = 0; j < HCLW_SCENT_ELEMS; j++) {
        payload[j] = 0;
    }

    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(sizeof(int32_t)));
    out.data<int32_t>()[0] = 0;
}

void InhibitSlab::eval_gpu(const std::vector<array>& inputs,
                           std::vector<array>& outputs) {
    (void)inputs;
    if (slot_index_ >= HCLW_N_SLOTS) {
        throw std::runtime_error("InhibitSlab: slot_index out of range");
    }

    auto& out = outputs[0];
    out.set_data(mlx::core::allocator::malloc(sizeof(int32_t)));
    out.data<int32_t>()[0] = 0;

    auto& d = mlx::core::metal::device(stream().device);
    auto& enc = d.get_command_encoder(stream().index);

    auto* lib = d.get_library("hiveclaw_claim_inhibit", [] {
        return std::string(CLAIM_INHIBIT_MSL);
    });
    auto* kernel = d.get_kernel("inhibit_slot", lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_buffer(slab_buf_, 0, 0);
    enc.set_bytes(slot_index_, 1);
    enc.set_bytes(agent_id_, 2);

    size_t n = HCLW_SCENT_ELEMS;
    size_t tgp = std::min(n, static_cast<size_t>(256));
    enc.dispatch_threads(MTL::Size(static_cast<NS::UInteger>(n), 1, 1),
                         MTL::Size(static_cast<NS::UInteger>(tgp), 1, 1));
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
    Stream s = default_stream(Device::gpu);
    auto prim =
        std::make_shared<ClaimSlabTask>(slab_buf_, agent_id, s);
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
    Stream s = default_stream(Device::gpu);
    if (dep && (*dep).has_primitive()) {
        s = (*dep).primitive().stream();
    }
    auto prim = std::make_shared<InhibitSlab>(slab_buf_, slot_index, agent_id, s);
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
    if (slot_index >= HCLW_N_SLOTS) {
        throw std::runtime_error("release_slot: slot_index out of range");
    }
    char* base = static_cast<char*>(slab_buf_->contents());
    size_t off = slot_base(slot_index) + OFF_S_CLAIM_FLAG;
    auto* claim = reinterpret_cast<std::atomic<uint32_t>*>(base + off);
    claim->store(0u, std::memory_order_release);
}

std::vector<std::pair<bool, uint32_t>> SlabHandle::get_slot_states() const {
    char* base = static_cast<char*>(slab_buf_->contents());
    std::vector<std::pair<bool, uint32_t>> out;
    out.reserve(HCLW_N_SLOTS);
    for (size_t i = 0; i < HCLW_N_SLOTS; ++i) {
        size_t b = slot_base(i);
        auto* claim = reinterpret_cast<std::atomic<uint32_t>*>(base + b + OFF_S_CLAIM_FLAG);
        bool claimed = (claim->load(std::memory_order_relaxed) != 0u);
        uint32_t owner = *reinterpret_cast<uint32_t*>(base + b + OFF_S_OWNER_ID);
        out.emplace_back(claimed, owner);
    }
    return out;
}
