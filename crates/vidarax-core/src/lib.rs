pub mod gate;
pub mod ingest;
pub mod webrtc;
pub mod lockfree;
pub mod loop_detector;
pub mod metrics;
pub mod pipeline;
pub mod provider;
pub mod tiered_vlm;
pub mod timeline;
#[cfg(feature = "training")]
pub mod training_data;

#[cfg(test)]
mod tests {
    use crate::lockfree::spsc::spsc_channel;

    #[test]
    fn channel_basics_compile() {
        let (producer, consumer) = spsc_channel::<u64, 8>();
        assert!(producer.push(7).is_ok());
        assert_eq!(consumer.pop(), Some(7));
    }
}
