// The core engine is memory-safe Rust. The only unsafe in the crate is the
// libvpx FFI in webrtc::decode, which is compiled solely under `--features vp8`
// and carries a scoped allow at its module boundary. Denying unsafe everywhere
// else keeps that boundary the one audited place a raw pointer can appear, so a
// new unsafe block anywhere else fails the build instead of slipping in.
#![deny(unsafe_code)]

pub mod backends;
pub mod crop;
pub mod dedup;
pub mod gate;
pub mod gemini;
pub mod ingest;
pub mod loop_detector;
pub mod metrics;
pub mod novelty;
pub mod pipeline;
pub mod provider;
pub mod semantic_merge;
pub mod tiered_vlm;
pub mod timeline;
#[cfg(feature = "training")]
pub mod training_data;
pub mod webrtc;

#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
