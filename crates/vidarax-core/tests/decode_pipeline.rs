use std::sync::Arc;
use vidarax_core::ingest::pipeline::{
    build_decode_pipeline, create_pipeline, register_decode_backend, BackendCapabilities,
    DecodePipeline, PipelineBackend,
};
use vidarax_core::ingest::{DecodedJpegFrame, DecodedMp4Batch, InputSource, Mp4DecodeConfig};

#[test]
fn registry_builds_known_backend() {
    let pipeline = build_decode_pipeline("cpu").expect("cpu backend should build");

    assert!(matches!(pipeline.backend(), PipelineBackend::CpuFfmpeg));
}

#[test]
fn backend_from_env_string() {
    assert!(matches!(
        PipelineBackend::parse("cpu"),
        Ok(PipelineBackend::CpuFfmpeg)
    ));
    assert!(matches!(
        PipelineBackend::parse("ffmpeg"),
        Ok(PipelineBackend::CpuFfmpeg)
    ));
    assert!(matches!(
        PipelineBackend::parse("nvdec"),
        Ok(PipelineBackend::NvdecCuda)
    ));
    assert!(matches!(
        PipelineBackend::parse("cuda"),
        Ok(PipelineBackend::NvdecCuda)
    ));
    assert!(matches!(
        PipelineBackend::parse("mlx"),
        Ok(PipelineBackend::VideoToolbox)
    ));
    assert!(matches!(
        PipelineBackend::parse("apple"),
        Ok(PipelineBackend::VideoToolbox)
    ));
    assert!(matches!(
        PipelineBackend::parse("metal"),
        Ok(PipelineBackend::VideoToolbox)
    ));
    assert!(matches!(
        PipelineBackend::parse("videotoolbox"),
        Ok(PipelineBackend::VideoToolbox)
    ));
    assert!(PipelineBackend::parse("invalid").is_err());
}

#[test]
fn registry_builds_videotoolbox_backend_under_every_alias() {
    for alias in ["mlx", "apple", "metal", "videotoolbox"] {
        let pipeline =
            build_decode_pipeline(alias).unwrap_or_else(|_| panic!("{alias} backend should build"));
        assert!(matches!(pipeline.backend(), PipelineBackend::VideoToolbox));
        assert!(pipeline.capabilities().hardware_decode);
    }
}

#[test]
fn create_pipeline_builds_videotoolbox_backend() {
    let pipeline = create_pipeline(PipelineBackend::VideoToolbox);
    assert!(matches!(pipeline.backend(), PipelineBackend::VideoToolbox));
    assert_eq!(pipeline.backend().label(), "videotoolbox");
}

#[test]
fn unknown_registry_backend_returns_error() {
    let err = match build_decode_pipeline("missing-backend") {
        Ok(_) => panic!("unknown backend should fail"),
        Err(err) => err,
    };

    assert!(err.contains("unknown decode backend"));
    assert!(err.contains("missing-backend"));
}

#[test]
fn registered_backend_round_trips_by_name() {
    register_decode_backend("test-custom-backend", || Arc::new(CustomPipeline));

    let pipeline =
        build_decode_pipeline("test-custom-backend").expect("registered backend should build");

    assert!(matches!(pipeline.backend(), PipelineBackend::CpuFfmpeg));
    assert_eq!(pipeline.capabilities().notes, "custom test backend");
}

#[test]
fn auto_detect_resolves_to_buildable_pipeline() {
    let detected = PipelineBackend::auto_detect();

    let pipeline = build_decode_pipeline(detected.label()).expect("auto backend should build");

    assert_eq!(pipeline.backend(), detected);
}

struct CustomPipeline;

impl DecodePipeline for CustomPipeline {
    fn decode_signals(
        &self,
        _source: &InputSource,
        _config: Mp4DecodeConfig,
    ) -> Result<DecodedMp4Batch, String> {
        unimplemented!("test backend is only built, not executed")
    }

    fn decode_jpegs(
        &self,
        _source: &InputSource,
        _sample_fps: f32,
        _frame_indices: &[u64],
        _max_frames: usize,
        _max_edge: Option<u32>,
        _crop: Option<vidarax_core::crop::CropRegion>,
    ) -> Result<Vec<DecodedJpegFrame>, String> {
        unimplemented!("test backend is only built, not executed")
    }

    fn extract_clip(
        &self,
        _source: &InputSource,
        _start_s: f32,
        _duration_s: f32,
    ) -> Result<Vec<u8>, String> {
        unimplemented!("test backend is only built, not executed")
    }

    fn backend(&self) -> PipelineBackend {
        PipelineBackend::CpuFfmpeg
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            hardware_decode: false,
            notes: "custom test backend",
        }
    }
}
