use std::path::PathBuf;
use std::sync::Mutex;
use vidarax_core::ingest::InputSource;

// Env-var mutation is process-global, so tests that depend on
// VIDARAX_ALLOW_UNENCRYPTED_RTSP must not run concurrently.
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn unencrypted_rtsp_is_rejected_by_default() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::remove_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP");

    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://example.com/stream1", &roots);
    assert!(
        result.is_err(),
        "unencrypted rtsp:// should be rejected by default: {result:?}"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("rtsps://"),
        "error should suggest rtsps:// alternative: {msg}"
    );
}

#[test]
fn unencrypted_rtsp_accepted_when_env_allows() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP", "true");

    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://example.com/stream1", &roots);

    std::env::remove_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP");
    assert!(
        result.is_ok(),
        "rtsp:// should be accepted with VIDARAX_ALLOW_UNENCRYPTED_RTSP=true: {result:?}"
    );
}

#[test]
fn rtsp_private_ip_is_blocked() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP", "true");

    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://192.168.1.100/stream", &roots);

    std::env::remove_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP");
    assert!(result.is_err(), "rtsp to private IP should be blocked");
}

#[test]
fn rtsp_localhost_is_blocked() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP", "true");

    let roots: Vec<PathBuf> = vec![];
    let result = InputSource::parse_and_validate("rtsp://localhost/stream", &roots);

    std::env::remove_var("VIDARAX_ALLOW_UNENCRYPTED_RTSP");
    assert!(result.is_err(), "rtsp to localhost should be blocked");
}
