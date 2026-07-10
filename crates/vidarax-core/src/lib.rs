pub mod backends;
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
