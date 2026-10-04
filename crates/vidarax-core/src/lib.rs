// Unsafe operations stay within two platform boundaries: libvpx FFI under
// `--features vp8` and macOS durable file synchronization in timeline. Each
// boundary has a scoped allow and documents its pointer or descriptor lifetime.
// A new unsafe operation elsewhere fails the build.
#![deny(unsafe_code)]

pub mod admission;
pub mod audio_sidecar;
pub mod backends;
pub mod coordinates;
pub mod crop;
pub mod dedup;
pub mod embedding_sidecar;
pub mod gate;
pub mod gemini;
pub mod ingest;
pub mod loop_detector;
mod media_process;
pub mod metrics;
pub mod novelty;
pub mod pipeline;
pub mod provider;
mod sidecar_io;
pub mod tiered_vlm;
pub mod timeline;
pub mod trigger;
pub mod webrtc;
pub mod zone;

#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
