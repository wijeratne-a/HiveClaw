#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/mlx.h>
#include <mlx/primitives.h>

constexpr size_t HIVECLAW_SLAB_SIZE = 4'718'720;

// Holds the long-lived MTL::Buffer* for the IOSurface slab.
class SlabHandle {
   public:
    explicit SlabHandle(uint32_t surface_id);
    ~SlabHandle();
    SlabHandle(const SlabHandle&) = delete;
    SlabHandle& operator=(const SlabHandle&) = delete;

    mlx::core::array write(size_t byte_offset,
                           mlx::core::array scent_c,
                           std::optional<mlx::core::array> dep);

    /// Phase C: write scent at `slot_payload(slot_index)` and stamp `last_write_clock`.
    mlx::core::array write_slot(uint32_t slot_index,
                                mlx::core::array scent_c,
                                std::optional<mlx::core::array> dep);

    mlx::core::array read(size_t byte_offset,
                          mlx::core::Shape shape,
                          std::optional<mlx::core::array> dep);

    mlx::core::array read_slot(uint32_t slot_index,
                               mlx::core::Shape shape,
                               std::optional<mlx::core::array> dep);

    mlx::core::array claim(mlx::core::array candidate_indices,
                           uint32_t agent_id,
                           std::optional<mlx::core::array> dep);

    mlx::core::array inhibit(uint32_t slot_index,
                             uint32_t agent_id,
                             std::optional<mlx::core::array> dep);

    /// CPU: clear claim_flag (call when done with a held slot).
    void release_slot(uint32_t slot_index);

    /// CPU best-effort snapshot: [{claimed, owner_id}, ...] for all 32 slots.
    std::vector<std::pair<bool, uint32_t>> get_slot_states() const;

    MTL::Buffer* raw_buffer() const { return slab_buf_; }

   private:
    MTL::Buffer* slab_buf_{nullptr};
    uint32_t surface_id_{0};
};

// MLX Primitive: copies scent_c → slab[byte_offset], returns scent_c (identity).
class WriteSlab : public mlx::core::UnaryPrimitive {
   public:
    WriteSlab(MTL::Buffer* slab,
              size_t byte_offset,
              size_t num_bytes,
              mlx::core::Stream s,
              uint32_t stamp_slot_index = 0xFFFFFFFFu);
    void eval_cpu(const std::vector<mlx::core::array>& in, mlx::core::array& out) override;
    void eval_gpu(const std::vector<mlx::core::array>& in, mlx::core::array& out) override;
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
    const char* name() const override { return "WriteSlab"; }

   private:
    MTL::Buffer* slab_buf_;
    size_t byte_offset_;
    size_t num_bytes_;
    uint32_t stamp_slot_index_;
};

// MLX Primitive: copies slab[byte_offset] → fresh bf16 array.
class ReadSlab : public mlx::core::Primitive {
   public:
    ReadSlab(MTL::Buffer* slab,
             size_t byte_offset,
             mlx::core::Shape shape,
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
    const char* name() const override { return "ReadSlab"; }

   private:
    MTL::Buffer* slab_buf_;
    size_t byte_offset_;
    mlx::core::Shape shape_;
};
