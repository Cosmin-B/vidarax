use std::path::PathBuf;
use vidarax_core::ingest::InputSource;

#[test]
fn rtsp_url_is_accepted() {
    let roots: Vec<PathBuf> = vec![];
    // Use example.com which resolves to a public IP (93.184.216.34).
    let result = InputSource::parse_and_validate("rtsp://example.com/stream1", &roots);
    assert!(result.is_ok(), "rtsp URL should be accepted: {result:?}");
    assert_eq!(result.unwrap(), InputSource::Url("rtsp://example.com/stream1".to_string()));
}

#[test]
fn rtsp_private_ip_is_blocked() {
    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://192.168.1.100/stream", &roots);
    assert!(result.is_err(), "rtsp to private IP should be blocked");
}

#[test]
fn rtsp_localhost_is_blocked() {
    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://localhost/stream", &roots);
    assert!(result.is_err(), "rtsp to localhost should be blocked");
}
