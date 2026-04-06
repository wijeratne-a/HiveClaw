#pragma once

#include <cstddef>
#include <cstdint>

// Creates an id<MTLBuffer> (returned as void*) that aliases the IOSurface.
// Uses the MTL::Device* from MLX (passed as void* to avoid Obj-C in this header).
// Caller owns the retain; pass to release_slab_buffer() when done.
void* create_slab_buffer(void* mlx_mtl_device_ptr, uint32_t surface_id);
void release_slab_buffer(void* buf_ptr);
