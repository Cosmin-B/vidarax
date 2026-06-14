pub mod backends;
pub mod dedup;
pub mod gemini;
pub mod gate;
pub mod ingest;
pub mod webrtc;
pub mod loop_detector;
pub mod metrics;
pub mod pipeline;
pub mod provider;
pub mod tiered_vlm;
pub mod timeline;
#[cfg(feature = "training")]
pub mod training_data;

#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
