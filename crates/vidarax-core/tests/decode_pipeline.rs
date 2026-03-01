use vidarax_core::ingest::pipeline::{
    CpuFfmpegPipeline, DecodePipeline, DecodePipelineConfig, PipelineBackend,
};
use vidarax_core::ingest::InputSource;
use std::path::PathBuf;

#[test]
fn cpu_pipeline_config_defaults() {
    let config = DecodePipelineConfig::default();
    assert!(matches!(config.backend, PipelineBackend::CpuFfmpeg));
    assert!(config.sample_fps > 0.0);
}

#[test]
fn backend_from_env_string() {
    assert!(matches!(PipelineBackend::parse("cpu"), Ok(PipelineBackend::CpuFfmpeg)));
    assert!(matches!(PipelineBackend::parse("ffmpeg"), Ok(PipelineBackend::CpuFfmpeg)));
    assert!(matches!(PipelineBackend::parse("nvdec"), Ok(PipelineBackend::NvdecCuda)));
    assert!(matches!(PipelineBackend::parse("cuda"), Ok(PipelineBackend::NvdecCuda)));
    assert!(matches!(PipelineBackend::parse("mlx"), Ok(PipelineBackend::Mlx)));
    assert!(PipelineBackend::parse("invalid").is_err());
}

#[test]
fn cpu_pipeline_creates_without_panic() {
    let config = DecodePipelineConfig::default();
    let _pipeline = CpuFfmpegPipeline::new(config);
}
