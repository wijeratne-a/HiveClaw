//! HiveClaw daemon library (XPC helpers shared with integration tests).

#[cfg(target_os = "macos")]
pub mod decay;
#[cfg(target_os = "macos")]
pub mod xpc;

// Non-macOS: empty library (Metal / IOSurface targets are macOS-only).
