//! `MTLBuffer` aliasing IOSurface memory (`MTLStorageModeShared`).

use crate::sys::{self, OwnedIosurface};
use hiveclaw_core::math::{SlabLayout, DEFAULT_LATENT_ELEMS, N_SLOTS};
use metal::{Device, MTLResourceOptions};

/// IOSurface-backed slab with a Metal buffer alias (no extra H2D copy of the body).
///
/// `surface` is dropped **after** `_mtl_buffer` so Metal releases the buffer before `CFRelease`.
pub struct MetalPheromoneBuffer {
    surface: OwnedIosurface,
    _mtl_buffer: metal::Buffer,
    base: *mut u8,
    size: usize,
}

impl MetalPheromoneBuffer {
    /// Daemon: create IOSurface with layout-sized allocation, lock, bind `MTLBuffer`.
    pub fn new_with_layout(layout: &SlabLayout) -> Self {
        let surface = sys::OwnedIosurface::create_slab_with_size(layout.iosurface_bytes);
        sys::lock_surface(surface.as_ptr());
        let base = sys::base_address(surface.as_ptr());
        let sz = sys::alloc_size(surface.as_ptr());
        assert!(
            sz >= layout.iosurface_bytes,
            "IOSurface alloc size {sz} < requested {}",
            layout.iosurface_bytes
        );

        let device = Device::system_default().expect("Metal device must exist on Apple Silicon");
        let mtl_buffer = device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            sz as metal::NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Self {
            surface,
            _mtl_buffer: mtl_buffer,
            base,
            size: sz,
        }
    }

    /// Daemon/tests: default 256-D latent layout.
    pub fn new() -> Self {
        let layout = SlabLayout::try_from_latent_elems(DEFAULT_LATENT_ELEMS, N_SLOTS as u32)
            .expect("default layout");
        Self::new_with_layout(&layout)
    }

    /// Worker / Python: `IOSurfaceLookup` then same Metal bind.
    pub fn from_surface_id(id: u32) -> Self {
        let surface = sys::OwnedIosurface::lookup(id);
        sys::lock_surface(surface.as_ptr());
        let base = sys::base_address(surface.as_ptr());
        let sz = sys::alloc_size(surface.as_ptr());
        assert!(sz > 0, "IOSurface alloc size is zero");

        let device = Device::system_default().expect("Metal device must exist on Apple Silicon");
        let mtl_buffer = device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            sz as metal::NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        Self {
            surface,
            _mtl_buffer: mtl_buffer,
            base,
            size: sz,
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
        self.size
    }
}

impl Drop for MetalPheromoneBuffer {
    fn drop(&mut self) {
        sys::unlock_surface(self.surface.as_ptr());
    }
}

unsafe impl Send for MetalPheromoneBuffer {}
unsafe impl Sync for MetalPheromoneBuffer {}
