//! Minimal IOSurface FFI for the Phase 2 slab POC (macOS).

use core::ffi::c_void;
use core_foundation::base::{CFRelease, TCFType};
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::string::CFStringRef;
use hiveclaw_core::math::SLAB_SIZE;

/// Opaque IOSurface pointer (avoid exposing a private struct through `pub` APIs).
pub type IOSurfaceRef = *const c_void;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    static kIOSurfaceAllocSize: CFStringRef;
    static kIOSurfaceIsGlobal: CFStringRef;

    fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLookup(id: u32) -> IOSurfaceRef;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;

    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;

    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
}

/// Owns an `IOSurfaceRef` and releases it on drop (`CFRelease`).
pub struct OwnedIosurface(IOSurfaceRef);

impl OwnedIosurface {
    pub fn create_slab() -> Self {
        let size = CFNumber::from(SLAB_SIZE as i32);
        unsafe {
            let key_size = CFString::wrap_under_get_rule(kIOSurfaceAllocSize as *const _);
            let key_global = CFString::wrap_under_get_rule(kIOSurfaceIsGlobal as *const _);
            let dict = CFDictionary::from_CFType_pairs(&[
                (key_size, size.as_CFType()),
                (key_global, CFBoolean::true_value().as_CFType()),
            ]);
            let surf = IOSurfaceCreate(dict.as_concrete_TypeRef());
            assert!(!surf.is_null(), "IOSurfaceCreate failed");
            Self(surf)
        }
    }

    pub fn lookup(id: u32) -> Self {
        unsafe {
            let surf = IOSurfaceLookup(id);
            assert!(!surf.is_null(), "IOSurfaceLookup failed for id {id}");
            Self(surf)
        }
    }

    #[inline]
    pub fn id(&self) -> u32 {
        unsafe { IOSurfaceGetID(self.0) }
    }

    #[inline]
    pub fn as_ptr(&self) -> IOSurfaceRef {
        self.0
    }
}

impl Drop for OwnedIosurface {
    fn drop(&mut self) {
        unsafe {
            CFRelease(self.0 as *const c_void);
        }
    }
}

#[inline]
pub fn lock_surface(surface: IOSurfaceRef) {
    let mut seed = 0u32;
    let r = unsafe { IOSurfaceLock(surface, 0, &mut seed) };
    assert_eq!(r, 0, "IOSurfaceLock failed: {r}");
}

#[inline]
pub fn unlock_surface(surface: IOSurfaceRef) {
    let mut seed = 0u32;
    let r = unsafe { IOSurfaceUnlock(surface, 0, &mut seed) };
    assert_eq!(r, 0, "IOSurfaceUnlock failed: {r}");
}

#[inline]
pub fn base_address(surface: IOSurfaceRef) -> *mut u8 {
    unsafe { IOSurfaceGetBaseAddress(surface) as *mut u8 }
}

#[inline]
pub fn alloc_size(surface: IOSurfaceRef) -> usize {
    unsafe { IOSurfaceGetAllocSize(surface) }
}
