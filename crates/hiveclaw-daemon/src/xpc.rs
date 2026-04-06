//! Raw XPC C API (`xpc.h`) — Mach service `com.hiveclaw.pheromoned`.

use block2::{Block, RcBlock};
use hiveclaw_backend_metal::MetalPheromoneBuffer;
use hiveclaw_core::math::{
    layout_magic_version_u64, OFF_G_DECAY_RATE, OFF_G_LATENT_ELEMS, OFF_G_MAGIC_V5,
    OFF_G_N_SLOTS_V5, OFF_G_STRIDE_V5, OFF_G_VERSION_V5, OFF_G_ZETA_T, SlabLayout,
    SLAB_VERSION_V6,
};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set when a peer XPC connection invalidates (Phase 5 may mirror to VRAM lock line).
pub static SLOT_0_DIRTY: AtomicBool = AtomicBool::new(false);

pub(crate) type XpcObjectT = *mut c_void;
type XpcConnectionT = *mut c_void;
type XpcTypeT = *const c_void;
type DispatchQueueT = *mut c_void;

const XPC_CONNECTION_MACH_SERVICE_LISTENER: u64 = 1;

unsafe extern "C" {
    pub(crate) fn dispatch_main() -> !;

    pub(crate) fn xpc_connection_create_mach_service(
        name: *const c_char,
        targetq: DispatchQueueT,
        flags: u64,
    ) -> XpcConnectionT;
    fn xpc_connection_set_event_handler(
        connection: XpcConnectionT,
        handler: Option<&Block<dyn Fn(*mut c_void)>>,
    );
    fn xpc_connection_resume(connection: XpcConnectionT);
    fn xpc_connection_send_message(connection: XpcConnectionT, message: XpcObjectT);
    fn xpc_connection_send_message_with_reply_sync(
        connection: XpcConnectionT,
        message: XpcObjectT,
    ) -> XpcObjectT;

    pub(crate) fn xpc_get_type(object: XpcObjectT) -> XpcTypeT;
    pub(crate) fn xpc_release(object: XpcObjectT);
    fn xpc_type_get_name(ty: XpcTypeT) -> *const c_char;
    fn xpc_copy_description(obj: XpcObjectT) -> *mut c_char;

    fn xpc_dictionary_create_reply(original: XpcObjectT) -> XpcObjectT;
    fn xpc_dictionary_create_empty() -> XpcObjectT;
    fn xpc_dictionary_set_string(dict: XpcObjectT, key: *const c_char, value: *const c_char);
    fn xpc_dictionary_set_uint64(dict: XpcObjectT, key: *const c_char, value: u64);
    fn xpc_dictionary_get_string(dict: XpcObjectT, key: *const c_char) -> *const c_char;
    fn xpc_dictionary_get_uint64(dict: XpcObjectT, key: *const c_char) -> u64;
}

#[inline]
pub(crate) unsafe fn is_xpc_type(obj: XpcObjectT, name: &[u8]) -> bool {
    let ty = xpc_get_type(obj);
    let p = xpc_type_get_name(ty);
    if p.is_null() {
        return false;
    }
    CStr::from_ptr(p).to_bytes() == name
}

#[inline]
unsafe fn is_connection_invalid(obj: XpcObjectT) -> bool {
    if !is_xpc_type(obj, b"error") {
        return false;
    }
    let desc = xpc_copy_description(obj);
    if desc.is_null() {
        return false;
    }
    let s = CStr::from_ptr(desc).to_string_lossy();
    libc::free(desc.cast());
    s.contains("Connection invalid") || s.contains("connection invalid")
}

/// Client: v5 handshake — IOSurface ID + packed `(magic << 32) | version`.
pub fn fetch_surface_v5() -> Result<(u32, u64), String> {
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
                    eprintln!(
                        "[hiveclaw xpc client] {}",
                        CStr::from_ptr(d).to_string_lossy()
                    );
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
    unsafe {
        xpc_release(reply);
        xpc_release(conn as XpcObjectT);
    }

    let expected = layout_magic_version_u64();
    if magic_version != expected {
        return Err(format!(
            "layout magic_version mismatch: got 0x{magic_version:x}, expected 0x{expected:x}"
        ));
    }

    Ok((sid as u32, magic_version))
}

/// Client: synchronous RPC returning the daemon's IOSurface ID (v5 handshake).
pub fn fetch_surface_id() -> Result<u32, String> {
    fetch_surface_v5().map(|(id, _)| id)
}

/// After mapping the IOSurface, validate bytes 0–11 (u64 magic + u32 version).
pub fn validate_mapped_global_header_v5(base: *const u8) -> Result<(), String> {
    let magic = unsafe { (base as *const u64).read_unaligned() };
    let ver = unsafe { (base.add(OFF_G_VERSION_V5) as *const u32).read_unaligned() };
    if magic != hiveclaw_core::math::layout_magic_version_u64() {
        return Err(format!(
            "global header magic mismatch: got 0x{magic:x}, expected 0x{:x}",
            layout_magic_version_u64()
        ));
    }
    if ver != SLAB_VERSION_V6 {
        return Err(format!(
            "global header version mismatch: got {ver}, expected {SLAB_VERSION_V6}"
        ));
    }
    Ok(())
}

/// Daemon entry: IOSurface slab + XPC listener (never returns).
/// Writes Phase C global header after the slab has been zeroed.
unsafe fn initialize_global_header(base: *mut u8, layout: &SlabLayout, decay_rate: f32) {
    let p = base;
    (p.add(OFF_G_MAGIC_V5) as *mut u64).write_unaligned(layout_magic_version_u64());
    (p.add(OFF_G_VERSION_V5) as *mut u32).write_unaligned(SLAB_VERSION_V6);
    (p.add(OFF_G_N_SLOTS_V5) as *mut u32).write_unaligned(layout.n_slots);
    (p.add(OFF_G_STRIDE_V5) as *mut u32).write_unaligned(layout.stride);
    (p.add(OFF_G_LATENT_ELEMS) as *mut u32).write_unaligned(layout.latent_elems);
    (p.add(OFF_G_ZETA_T) as *mut f32).write_volatile(0.0);
    (p.add(OFF_G_DECAY_RATE) as *mut f32).write_volatile(decay_rate);
}

pub fn run_pheromoned_daemon(layout: SlabLayout) -> ! {
    let iosz = layout.iosurface_bytes;
    let slab = MetalPheromoneBuffer::new_with_layout(&layout);
    let base = slab.base_ptr();
    unsafe {
        ptr::write_bytes(base, 0, iosz);
        let decay_rate = 0.05_f32;
        initialize_global_header(base, &layout, decay_rate);
        crate::decay::start_decay_loop(base, 100, layout);
    }
    let surface_id = slab.surface_id();
    let _slab = Box::leak(Box::new(slab));

    let service_name = CString::new("com.hiveclaw.pheromoned").expect("service name");

    let listener = unsafe {
        xpc_connection_create_mach_service(
            service_name.as_ptr(),
            ptr::null_mut(),
            XPC_CONNECTION_MACH_SERVICE_LISTENER,
        )
    };
    assert!(
        !listener.is_null(),
        "xpc_connection_create_mach_service(LISTENER) failed — is launchd MachServices registered?"
    );

    let daemon_exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(unknown)".to_string());

    let listener_block = RcBlock::new(move |event: *mut c_void| {
        if event.is_null() {
            return;
        }
        unsafe {
            if is_connection_invalid(event) {
                return;
            }

            if is_xpc_type(event, b"connection") {
                let peer = event as XpcConnectionT;
                let sid = surface_id;
                let exe_for_peer = daemon_exe.clone();

                let peer_block = RcBlock::new(move |msg: *mut c_void| {
                    if msg.is_null() {
                        return;
                    }
                    if is_connection_invalid(msg) {
                        SLOT_0_DIRTY.store(true, Ordering::SeqCst);
                        eprintln!("[pheromoned] client disconnected — slot 0 marked dirty");
                        return;
                    }

                    if !is_xpc_type(msg, b"dictionary") {
                        return;
                    }

                    let cmd_ptr =
                        xpc_dictionary_get_string(msg, b"cmd\0".as_ptr().cast::<c_char>());
                    if cmd_ptr.is_null() {
                        return;
                    }
                    let cmd = match CStr::from_ptr(cmd_ptr).to_str() {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    let reply = xpc_dictionary_create_reply(msg);
                    if reply.is_null() {
                        return;
                    }

                    if cmd != "get_surface_v5" {
                        xpc_dictionary_set_string(
                            reply,
                            b"error\0".as_ptr().cast::<c_char>(),
                            b"INVALID_COMMAND_OR_UNSUPPORTED_VERSION\0"
                                .as_ptr()
                                .cast::<c_char>(),
                        );
                        xpc_connection_send_message(peer, reply);
                        xpc_release(reply);
                        return;
                    }

                    xpc_dictionary_set_uint64(
                        reply,
                        b"surface_id\0".as_ptr().cast::<c_char>(),
                        u64::from(sid),
                    );
                    xpc_dictionary_set_uint64(
                        reply,
                        b"magic_version\0".as_ptr().cast::<c_char>(),
                        layout_magic_version_u64(),
                    );
                    if let Ok(exe_c) = CString::new(exe_for_peer.as_str()) {
                        xpc_dictionary_set_string(
                            reply,
                            b"daemon_exe\0".as_ptr().cast::<c_char>(),
                            exe_c.as_ptr(),
                        );
                    }
                    if let Ok(ver_c) = CString::new(env!("CARGO_PKG_VERSION")) {
                        xpc_dictionary_set_string(
                            reply,
                            b"daemon_crate_version\0".as_ptr().cast::<c_char>(),
                            ver_c.as_ptr(),
                        );
                    }
                    xpc_connection_send_message(peer, reply);
                    xpc_release(reply);
                });

                xpc_connection_set_event_handler(peer, Some(&*peer_block));
                xpc_connection_resume(peer);
            }
        }
    });

    unsafe {
        xpc_connection_set_event_handler(listener, Some(&*listener_block));
        xpc_connection_resume(listener);
    }

    eprintln!(
        "[pheromoned] XPC Mach service com.hiveclaw.pheromoned ready — surface_id={surface_id}"
    );

    unsafe { dispatch_main() }
}
