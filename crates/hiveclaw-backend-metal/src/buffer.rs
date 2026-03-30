//! `MTLBuffer` aliasing IOSurface memory (`MTLStorageModeShared`).

use crate::sys::{self, OwnedIosurface};
use hiveclaw_core::math::SLAB_SIZE;
use metal::{Device, MTLResourceOptions};

/// IOSurface-backed slab with a Metal buffer alias (no extra H2D copy of the body).
///
/// `surface` is dropped **after** `_mtl_buffer` so Metal releases the buffer before `CFRelease`.
pub struct MetalPheromoneBuffer {
    surface: OwnedIosurface,
    _mtl_buffer: metal::Buffer,
    base: *mut u8,
}

impl MetalPheromoneBuffer {
    /// Daemon: create IOSurface, lock, bind `MTLBuffer` with `newBufferWithBytesNoCopy`.
    pub fn new() -> Self {
        let surface = sys::OwnedIosurface::create_slab();
        sys::lock_surface(surface.as_ptr());
        let base = sys::base_address(surface.as_ptr());
        let sz = sys::alloc_size(surface.as_ptr());
        assert_eq!(
            sz, SLAB_SIZE,
            "IOSurface alloc size mismatch: got {sz}, expected {SLAB_SIZE}"
        );

        let device = Device::system_default().expect("Metal device must exist on Apple Silicon");
        let mtl_buffer = device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            SLAB_SIZE as metal::NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Self {
            surface,
            _mtl_buffer: mtl_buffer,
            base,
        }
    }

    /// Worker / Python: `IOSurfaceLookup` then same Metal bind.
    pub fn from_surface_id(id: u32) -> Self {
        let surface = sys::OwnedIosurface::lookup(id);
        sys::lock_surface(surface.as_ptr());
        let base = sys::base_address(surface.as_ptr());
        let sz = sys::alloc_size(surface.as_ptr());
        assert_eq!(sz, SLAB_SIZE, "IOSurface alloc size mismatch");

        let device = Device::system_default().expect("Metal device must exist on Apple Silicon");
        let mtl_buffer = device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            SLAB_SIZE as metal::NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Self {
            surface,
            _mtl_buffer: mtl_buffer,
            base,
        }
    }

    #[inline]
    pub fn surface_id(&self) -> u32 {
        self.surface.id()
    }

    #[inline]
    pub fn base_ptr_usize(&self) -> usize {
        self.base as usize
    }

    #[inline]
    pub fn base_ptr(&self) -> *mut u8 {
        self.base
    }

    #[inline]
    pub fn slab_size(&self) -> usize {
        SLAB_SIZE
    }
}

impl Drop for MetalPheromoneBuffer {
    fn drop(&mut self) {
        sys::unlock_surface(self.surface.as_ptr());
    }
}

unsafe impl Send for MetalPheromoneBuffer {}
unsafe impl Sync for MetalPheromoneBuffer {}
