#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "slab_bridge.h"

void* create_slab_buffer(void* mlx_mtl_device_ptr, uint32_t surface_id, size_t slab_size) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mlx_mtl_device_ptr;
    IOSurfaceRef surf = IOSurfaceLookup(surface_id);
    if (!surf) {
        return nullptr;
    }
    IOSurfaceIncrementUseCount(surf);
    void* base = IOSurfaceGetBaseAddress(surf);
    if (!base) {
        IOSurfaceDecrementUseCount(surf);
        CFRelease(surf);
        return nullptr;
    }
    id<MTLBuffer> buf = [device
        newBufferWithBytesNoCopy:base
                            length:slab_size
                           options:MTLResourceStorageModeShared
                       deallocator:^(void*, NSUInteger) {
                           IOSurfaceDecrementUseCount(surf);
                           CFRelease(surf);
                       }];
    if (!buf) {
        IOSurfaceDecrementUseCount(surf);
        CFRelease(surf);
        return nullptr;
    }
    return (__bridge_retained void*)buf;
}

void release_slab_buffer(void* buf_ptr) {
    if (!buf_ptr) {
        return;
    }
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_ptr;
    (void)buf;
}
