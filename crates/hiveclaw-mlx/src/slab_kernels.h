#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <mlx/array.h>
#include <mlx/mlx.h>
#include <mlx/primitives.h>

/// Try to claim the first free slot among `candidates` (int32). Returns scalar int32: slot index or -1.
class ClaimSlabTask : public mlx::core::UnaryPrimitive {
   public:
    ClaimSlabTask(MTL::Buffer* slab,
                  uint32_t agent_id,
                  uint32_t stride,
                  uint32_t n_slots,
                  mlx::core::Stream s);
    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  mlx::core::array& out) override;
    void eval_gpu(const std::vector<mlx::core::array>& inputs,
                  mlx::core::array& out) override;
    std::vector<mlx::core::array> jvp(
        const std::vector<mlx::core::array>& primals,
        const std::vector<mlx::core::array>& tangents,
        const std::vector<int>& argnums) override;
    std::vector<mlx::core::array> vjp(
        const std::vector<mlx::core::array>& primals,
        const std::vector<mlx::core::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mlx::core::array>& outputs) override;
    std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<int>& axes) override;
    std::vector<mlx::core::Shape> output_shapes(
        const std::vector<mlx::core::array>& inputs) override;
    const char* name() const override { return "ClaimSlabTask"; }

   private:
    MTL::Buffer* slab_buf_;
    uint32_t agent_id_;
    uint32_t stride_{0};
    uint32_t n_slots_{0};
};

/// v5: memsets full slot stride for the target slot and marks it INHIBITED (overseer / fault path).
class InhibitSlab : public mlx::core::Primitive {
   public:
    InhibitSlab(MTL::Buffer* slab,
                uint32_t slot_index,
                uint32_t agent_id,
                uint32_t stride,
                uint32_t n_slots,
                mlx::core::Stream s);
    void eval_cpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override;
    void eval_gpu(const std::vector<mlx::core::array>& inputs,
                  std::vector<mlx::core::array>& outputs) override;
    std::vector<mlx::core::array> jvp(
        const std::vector<mlx::core::array>& primals,
        const std::vector<mlx::core::array>& tangents,
        const std::vector<int>& argnums) override;
    std::vector<mlx::core::array> vjp(
        const std::vector<mlx::core::array>& primals,
        const std::vector<mlx::core::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mlx::core::array>& outputs) override;
    std::pair<std::vector<mlx::core::array>, std::vector<int>> vmap(
        const std::vector<mlx::core::array>& inputs,
        const std::vector<int>& axes) override;
    std::vector<mlx::core::Shape> output_shapes(
        const std::vector<mlx::core::array>& inputs) override;
    const char* name() const override { return "InhibitSlab"; }

   private:
    MTL::Buffer* slab_buf_;
    uint32_t slot_index_;
    uint32_t agent_id_;
    uint32_t stride_{0};
    uint32_t n_slots_{0};
};
