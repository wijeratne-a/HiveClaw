//! PyO3 bridge: `SlabClient` — XPC + IOSurface slab (BF16), Phase 4 v5.

use block2::{Block, RcBlock};
use half::bf16;
use hiveclaw_backend_metal::MetalPheromoneBuffer;
use hiveclaw_core::math::{
    layout_magic_version_u64, OFF_G_VERSION_V5, SCENT_ELEMS, SLAB_SIZE, SLAB_VERSION_V5,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{fence, Ordering};

type XpcObjectT = *mut c_void;
type XpcConnectionT = *mut c_void;
type XpcTypeT = *const c_void;
type DispatchQueueT = *mut c_void;

unsafe extern "C" {
    fn xpc_connection_create_mach_service(
        name: *const c_char,
        targetq: DispatchQueueT,
        flags: u64,
    ) -> XpcConnectionT;
    fn xpc_connection_set_event_handler(
        connection: XpcConnectionT,
        handler: Option<&Block<dyn Fn(*mut c_void)>>,
    );
    fn xpc_connection_resume(connection: XpcConnectionT);
    fn xpc_connection_send_message_with_reply_sync(
        connection: XpcConnectionT,
        message: XpcObjectT,
    ) -> XpcObjectT;

    fn xpc_get_type(object: XpcObjectT) -> XpcTypeT;
    fn xpc_release(object: XpcObjectT);
    fn xpc_type_get_name(ty: XpcTypeT) -> *const c_char;
    fn xpc_copy_description(obj: XpcObjectT) -> *mut c_char;

    fn xpc_dictionary_create_empty() -> XpcObjectT;
    fn xpc_dictionary_set_string(dict: XpcObjectT, key: *const c_char, value: *const c_char);
    fn xpc_dictionary_get_uint64(dict: XpcObjectT, key: *const c_char) -> u64;
    fn xpc_dictionary_get_string(dict: XpcObjectT, key: *const c_char) -> *const c_char;
}

unsafe fn is_xpc_type(obj: XpcObjectT, name: &[u8]) -> bool {
    let ty = xpc_get_type(obj);
    let p = xpc_type_get_name(ty);
    if p.is_null() {
        return false;
    }
    CStr::from_ptr(p).to_bytes() == name
}

unsafe fn xpc_object_description(obj: XpcObjectT) -> Option<String> {
    let d = xpc_copy_description(obj);
    if d.is_null() {
        return None;
    }
    let s = CStr::from_ptr(d).to_string_lossy().into_owned();
    libc::free(d.cast());
    Some(s)
}

/// Owns an `xpc_connection_t`; released on drop so daemon sees invalidation when `SlabClient` is GC'd.
struct XpcConn(XpcConnectionT);

impl Drop for XpcConn {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                xpc_release(self.0 as XpcObjectT);
            }
        }
    }
}

unsafe impl Send for XpcConn {}

fn connect_and_fetch_surface_v5() -> Result<(XpcConn, u32), String> {
    let cname = CString::new("com.hiveclaw.pheromoned").map_err(|e| e.to_string())?;
    let conn = unsafe { xpc_connection_create_mach_service(cname.as_ptr(), ptr::null_mut(), 0) };
    if conn.is_null() {
        return Err("xpc_connection_create_mach_service(client) returned NULL".into());
    }

    let log = RcBlock::new(|evt: *mut c_void| {
        if evt.is_null() {
            return;
        }
        unsafe {
            if is_xpc_type(evt, b"error") {
                let d = xpc_copy_description(evt);
                if !d.is_null() {
                    eprintln!("[hiveclaw_python] {}", CStr::from_ptr(d).to_string_lossy());
                    libc::free(d.cast());
                }
            }
        }
    });

    unsafe {
        xpc_connection_set_event_handler(conn, Some(&*log));
        xpc_connection_resume(conn);
    }

    let dict = unsafe { xpc_dictionary_create_empty() };
    if dict.is_null() {
        unsafe {
            xpc_release(conn as XpcObjectT);
        }
        return Err("xpc_dictionary_create_empty failed".into());
    }

    unsafe {
        xpc_dictionary_set_string(
            dict,
            b"cmd\0".as_ptr().cast::<c_char>(),
            b"get_surface_v5\0".as_ptr().cast::<c_char>(),
        );
    }

    let reply = unsafe { xpc_connection_send_message_with_reply_sync(conn, dict) };
    unsafe {
        xpc_release(dict);
    }

    if reply.is_null() {
        unsafe {
            xpc_release(conn as XpcObjectT);
        }
        return Err("xpc_connection_send_message_with_reply_sync returned NULL".into());
    }

    let err_ptr = unsafe { xpc_dictionary_get_string(reply, b"error\0".as_ptr().cast::<c_char>()) };
    if !err_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() };
        unsafe {
            xpc_release(reply);
            xpc_release(conn as XpcObjectT);
        }
        return Err(msg);
    }

    let sid =
        unsafe { xpc_dictionary_get_uint64(reply, b"surface_id\0".as_ptr().cast::<c_char>()) };
    let magic_version =
        unsafe { xpc_dictionary_get_uint64(reply, b"magic_version\0".as_ptr().cast::<c_char>()) };
    let daemon_exe = unsafe {
        let p = xpc_dictionary_get_string(reply, b"daemon_exe\0".as_ptr().cast::<c_char>());
        if p.is_null() {
            None
        } else {
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    };
    let daemon_crate_version = unsafe {
        let p =
            xpc_dictionary_get_string(reply, b"daemon_crate_version\0".as_ptr().cast::<c_char>());
        if p.is_null() {
            None
        } else {
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    };
    let reply_desc = unsafe { xpc_object_description(reply) };
    unsafe {
        xpc_release(reply);
    }

    let expected = layout_magic_version_u64();
    if magic_version != expected {
        unsafe {
            xpc_release(conn as XpcObjectT);
        }
        let mut parts = vec![format!(
            "layout magic_version mismatch: got 0x{magic_version:x}, expected 0x{expected:x}"
        )];
        if magic_version == 0 {
            parts.push(
                "magic_version 0x0 usually means an empty or non-v5 reply (service unloaded, wrong/stale pheromoned binary, or XPC connection invalid).".into(),
            );
        }
        if let Some(ref exe) = daemon_exe {
            parts.push(format!("daemon reported exe: {exe}"));
        }
        if let Some(ref ver) = daemon_crate_version {
            parts.push(format!("daemon reported crate version: {ver}"));
        }
        if let Some(desc) = reply_desc {
            parts.push(format!("full XPC reply: {desc}"));
        }
        parts.push("From repo root run: make doctor".into());
        return Err(parts.join(" | "));
    }

    eprintln!(
        "[hiveclaw_python] XPC v5 handshake ok surface_id={} pheromoned_exe={} daemon_crate={}",
        sid,
        daemon_exe.as_deref().unwrap_or("(missing)"),
        daemon_crate_version.as_deref().unwrap_or("(missing)"),
    );

    Ok((XpcConn(conn), sid as u32))
}

fn bounds_end(byte_offset: usize, num_bytes: usize) -> PyResult<usize> {
    let end = byte_offset
        .checked_add(num_bytes)
        .ok_or_else(|| PyValueError::new_err("byte offset + length overflow"))?;
    if end > SLAB_SIZE {
        return Err(PyValueError::new_err(format!(
            "range exceeds slab: end {end} > SLAB_SIZE {SLAB_SIZE}"
        )));
    }
    Ok(end)
}

#[pyclass(subclass)]
pub struct SlabClient {
    /// Fields drop in declaration order: `buf` first (Metal/IOSurface), then `_conn` (XPC invalidates daemon).
    buf: MetalPheromoneBuffer,
    _conn: XpcConn,
}

#[pymethods]
impl SlabClient {
    #[new]
    pub fn new() -> PyResult<Self> {
        let (xpc_conn, id) = connect_and_fetch_surface_v5().map_err(|e| {
            PyValueError::new_err(format!(
                "XPC handshake failed (is pheromoned running under launchd?): {e}\n\
                 From repo root: make doctor   (checks launchctl program path vs target/release/pheromoned and SlabClient). \
                 Then: cargo build --release -p hiveclaw-daemon && make daemon-load (use Terminal.app if launchctl returns EIO in an IDE terminal)."
            ))
        })?;
        let buf = MetalPheromoneBuffer::from_surface_id(id);
        let base = buf.base_ptr();
        let magic = unsafe { ptr::read_unaligned(base as *const u64) };
        let ver = unsafe { ptr::read_unaligned(base.add(OFF_G_VERSION_V5) as *const u32) };
        if magic != layout_magic_version_u64() || ver != SLAB_VERSION_V5 {
            return Err(PyValueError::new_err(
                "Corrupted slab: global header validation failed",
            ));
        }
        Ok(Self {
            buf,
            _conn: xpc_conn,
        })
    }

    pub fn surface_id(&self) -> u32 {
        self.buf.surface_id()
    }

    pub fn write_bf16_at(&mut self, byte_offset: usize, data: Vec<f32>) -> PyResult<()> {
        if byte_offset % 2 != 0 {
            return Err(PyValueError::new_err(
                "byte_offset must be 2-byte aligned for BF16",
            ));
        }
        let num_bytes = data
            .len()
            .checked_mul(2)
            .ok_or_else(|| PyValueError::new_err("length overflow"))?;
        bounds_end(byte_offset, num_bytes)?;

        let bf16s: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
        let base = self.buf.base_ptr();
        unsafe {
            let dst = base.add(byte_offset) as *mut bf16;
            ptr::copy_nonoverlapping(bf16s.as_ptr(), dst, bf16s.len());
        }
        fence(Ordering::SeqCst);
        Ok(())
    }

    pub fn read_bf16_at(&self, byte_offset: usize, num_elements: usize) -> PyResult<Vec<f32>> {
        if byte_offset % 2 != 0 {
            return Err(PyValueError::new_err(
                "byte_offset must be 2-byte aligned for BF16",
            ));
        }
        let num_bytes = num_elements
            .checked_mul(2)
            .ok_or_else(|| PyValueError::new_err("length overflow"))?;
        bounds_end(byte_offset, num_bytes)?;

        let base = self.buf.base_ptr();
        fence(Ordering::SeqCst);
        let out: Vec<f32> = unsafe {
            let src = base.add(byte_offset) as *const bf16;
            (0..num_elements)
                .map(|i| src.add(i).read().to_f32())
                .collect()
        };
        Ok(out)
    }

    /// Number of bf16 latent elements per Phase C slot (256 for v5 SAE).
    pub fn get_latent_dim(&self) -> usize {
        SCENT_ELEMS
    }

    /// Read a little-endian `u32` from the mapped slab (epoch headers).
    pub fn read_u32_at(&self, byte_offset: usize) -> PyResult<u32> {
        bounds_end(byte_offset, 4)?;
        let base = self.buf.base_ptr();
        let v = unsafe { ptr::read_unaligned(base.add(byte_offset) as *const u32) };
        fence(Ordering::Acquire);
        Ok(v)
    }

    /// Write a little-endian `u32` (e.g. to perturb `front_epoch` in slab tests).
    pub fn write_u32_at(&mut self, byte_offset: usize, value: u32) -> PyResult<()> {
        bounds_end(byte_offset, 4)?;
        let base = self.buf.base_ptr();
        unsafe {
            ptr::write_unaligned(base.add(byte_offset) as *mut u32, value.to_le());
        }
        fence(Ordering::SeqCst);
        Ok(())
    }

    /// Read a little-endian `u64` from the mapped slab (global header magic).
    pub fn read_u64_at(&self, byte_offset: usize) -> PyResult<u64> {
        bounds_end(byte_offset, 8)?;
        let base = self.buf.base_ptr();
        let v = unsafe { ptr::read_unaligned(base.add(byte_offset) as *const u64) };
        fence(Ordering::Acquire);
        Ok(v)
    }
}

unsafe impl Send for SlabClient {}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SlabClient>()?;
    Ok(())
}
