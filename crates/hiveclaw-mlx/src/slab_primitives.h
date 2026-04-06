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

    /// Phase C v5: write 512-byte latent at `slot_payload(slot_index)`; stamps front/back epoch.
    mlx::core::array write_slot(uint32_t slot_index,
                                mlx::core::array scent_c,
                                std::optional<mlx::core::array> dep);

    mlx::core::array read(size_t byte_offset,
                          mlx::core::Shape shape,
                          std::optional<mlx::core::array> dep);

    mlx::core::array read_slot(uint32_t slot_index,
                               mlx::core::Shape shape,
                               std::optional<mlx::core::array> dep);

    /// v5: read [1,1,256] bf16 with torn-epoch detection (zeros on torn).
    mlx::core::array read_slot_v5(uint32_t slot_index,
                                  std::optional<mlx::core::array> dep);

    /// v5: write [1,1,256] bf16 strictly; 512 bytes + epoch stamp.
    mlx::core::array write_slot_v5(uint32_t slot_index,
                                   mlx::core::array latent,
                                   std::optional<mlx::core::array> dep);

    /// Batched v5 read: returns ([B,1,256] bf16, [B] uint8 status). Row i matches slot_indices[i].
    std::pair<mlx::core::array, mlx::core::array> read_slots_v5(
        std::vector<uint32_t> slot_indices,
        std::optional<mlx::core::array> dep);

    /// Batched v5 write: latents [B,1,256] bf16. Returns (latents passthrough, [B] uint8 status).
    std::pair<mlx::core::array, mlx::core::array> write_slots_v5(
        std::vector<uint32_t> slot_indices,
        mlx::core::array latents,
        std::optional<mlx::core::array> dep);

    mlx::core::array claim(mlx::core::array candidate_indices,
                           uint32_t agent_id,
                           std::optional<mlx::core::array> dep);

    mlx::core::array inhibit(uint32_t slot_index,
                             uint32_t agent_id,
                             std::optional<mlx::core::array> dep);

    /// CPU: clear claim_flag (call when done with a held slot).
    void release_slot(uint32_t slot_index);

    /// CPU snapshot: [{claimed, owner_id}, ...] for all slots.
    std::vector<std::pair<bool, uint32_t>> get_slot_states() const;

    MTL::Buffer* raw_buffer() const { return slab_buf_; }

    int get_latent_dim() const { return static_cast<int>(latent_elems_); }

   private:
    MTL::Buffer* slab_buf_{nullptr};
    uint32_t surface_id_{0};
    uint32_t latent_elems_{0};
    uint32_t stride_{0};
    uint32_t back_epoch_off_{0};
    uint32_t n_slots_{0};
    size_t slab_bytes_{0};
};

// MLX Primitive: copies scent_c → slab[byte_offset], returns scent_c (identity).
class WriteSlab : public mlx::core::UnaryPrimitive {
   public:
    WriteSlab(MTL::Buffer* slab,
              size_t byte_offset,
              size_t num_bytes,
              mlx::core::Stream s,
              uint32_t stamp_slot_index = 0xFFFFFFFFu,
              uint32_t v5_back_epoch_off = 0,
              uint32_t v5_latent_elems = 0);
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
    uint32_t v5_back_epoch_off_{0};
    uint32_t v5_latent_elems_{0};
};

// MLX Primitive: copies slab[byte_offset] → fresh bf16 array; optional v5 epoch torn check.
class ReadSlab : public mlx::core::Primitive {
   public:
    ReadSlab(MTL::Buffer* slab,
             size_t byte_offset,
             mlx::core::Shape shape,
             mlx::core::Stream s,
             std::optional<uint32_t> v5_slot_for_epoch = std::nullopt,
             uint32_t v5_back_epoch_off = 0,
             uint32_t v5_latent_elems = 0);
    ~ReadSlab() override;
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
    std::optional<uint32_t> v5_slot_for_epoch_;
    uint32_t v5_back_epoch_off_{0};
    uint32_t v5_latent_elems_{0};
    /// 1-byte shared; v5 GPU path only; completion handler may emit telemetry.
    MTL::Buffer* status_buf_{nullptr};
};

/// Shared B-byte status buffer (Metal shared memory) filled by batched read/write GPU/CPU paths.
class BatchStatusBuffer {
   public:
    BatchStatusBuffer(MTL::Device* dev, uint32_t b);
    ~BatchStatusBuffer();
    BatchStatusBuffer(const BatchStatusBuffer&) = delete;
    BatchStatusBuffer& operator=(const BatchStatusBuffer&) = delete;
    MTL::Buffer* buf() const { return buf_; }
    uint32_t batch_size() const { return B_; }

   private:
    MTL::Buffer* buf_{nullptr};
    uint32_t B_{0};
};

/// Batched v5 slab read → [B,1,256] bf16; writes per-row status bytes into BatchStatusBuffer.
class ReadSlabBatchedOp : public mlx::core::Primitive {
   public:
    ReadSlabBatchedOp(MTL::Buffer* slab,
                      std::vector<uint32_t> slot_indices,
                      std::shared_ptr<BatchStatusBuffer> status_ctx,
                      mlx::core::Stream s,
                      uint32_t latent_elems,
                      uint32_t stride,
                      uint32_t back_epoch_off);
    ~ReadSlabBatchedOp() override;
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
    const char* name() const override { return "ReadSlabBatchedOp"; }

   private:
    MTL::Buffer* slab_buf_;
    std::vector<uint32_t> slot_indices_;
    std::shared_ptr<BatchStatusBuffer> status_ctx_;
    uint32_t latent_elems_{0};
    uint32_t stride_{0};
    uint32_t back_epoch_off_{0};
};

/// Copies BatchStatusBuffer → mlx uint8 [B]; input array is graph dependency only.
class CopyBatchStatusOp : public mlx::core::UnaryPrimitive {
   public:
    CopyBatchStatusOp(std::shared_ptr<BatchStatusBuffer> status_ctx, mlx::core::Stream s);
    void eval_cpu(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) override;
    void eval_gpu(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) override;
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
    const char* name() const override { return "CopyBatchStatusOp"; }

   private:
    std::shared_ptr<BatchStatusBuffer> status_ctx_;
};

/// Batched v5 stamped writes; status per row in BatchStatusBuffer.
class WriteSlabBatchedOp : public mlx::core::Primitive {
   public:
    WriteSlabBatchedOp(MTL::Buffer* slab,
                       std::vector<uint32_t> slot_indices,
                       std::shared_ptr<BatchStatusBuffer> status_ctx,
                       mlx::core::Stream s,
                       uint32_t latent_elems,
                       uint32_t stride,
                       uint32_t back_epoch_off);
    ~WriteSlabBatchedOp() override;
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
    const char* name() const override { return "WriteSlabBatchedOp"; }

   private:
    MTL::Buffer* slab_buf_;
    std::vector<uint32_t> slot_indices_;
    std::shared_ptr<BatchStatusBuffer> status_ctx_;
    uint32_t latent_elems_{0};
    uint32_t stride_{0};
    uint32_t back_epoch_off_{0};
};
