use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    FilePath(String),
    Url(String),
    /// Live HLS stream (HTTP/HTTPS .m3u8 manifest or hls:// URL).
    /// ffmpeg handles HLS natively via the `hls` demuxer.
    HlsStream(String),
    /// Live WebRTC stream identified by session ID (processed via worker pools,
    /// not through ffmpeg).
    WebRtcStream(String),
}

impl InputSource {
    pub fn parse_and_validate(input: &str, allowed_file_roots: &[PathBuf]) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err("source_uri must not be empty".to_string());
        }

        if trimmed.contains("://") {
            let url =
                reqwest::Url::parse(trimmed).map_err(|err| format!("invalid source_uri: {err}"))?;
            return match url.scheme() {
                "http" | "https" => {
                    if url.path().to_ascii_lowercase().ends_with(".m3u8") {
                        validate_hls_url(url)
                    } else {
                        validate_remote_url(url)
                    }
                }
                "hls" => validate_hls_url(url),
                "rtsps" => validate_remote_url(url),
                "rtsp" => {
                    // Operators can opt in for legacy cameras on trusted networks.
                    let allow_plain = std::env::var("VIDARAX_ALLOW_UNENCRYPTED_RTSP")
                        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                        .unwrap_or(false);
                    if !allow_plain {
                        return Err(
                            "unencrypted rtsp:// is not allowed; use rtsps:// or set \
                             VIDARAX_ALLOW_UNENCRYPTED_RTSP=true"
                                .to_string(),
                        );
                    }
                    validate_remote_url(url)
                }
                "file" => validate_file_url(url, allowed_file_roots),
                other => Err(format!(
                    "unsupported source_uri scheme '{other}', expected one of: \
                     file, hls, http, https, rtsps"
                )),
            };
        }

        validate_file_path(trimmed, allowed_file_roots)
    }

    pub fn as_ffmpeg_input(&self) -> &str {
        match self {
            InputSource::FilePath(path) | InputSource::Url(path) | InputSource::HlsStream(path) => {
                path
            }
            InputSource::WebRtcStream(session_id) => session_id.as_str(),
        }
    }
}

fn validate_remote_url(url: reqwest::Url) -> Result<InputSource, String> {
    if !url.username().is_empty() || url.password().is_some() {
        return Err("source_uri must not contain embedded credentials".to_string());
    }
    let host = url
        .host_str()
        .ok_or_else(|| "source_uri must include a valid host".to_string())?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
        return Err("source_uri host must not target localhost/local domains".to_string());
    }
    if let Ok(ip) = lower.parse::<IpAddr>() {
        if blocked_ip(&ip) {
            return Err("source_uri host must not be private, loopback, or link-local".to_string());
        }
    }

    // Resolve DNS and validate every address to prevent DNS rebinding.
    let port = url.port_or_known_default().unwrap_or(80);
    let resolve_target = format!("{host}:{port}");
    match resolve_target.to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            if addrs.is_empty() {
                return Err("source_uri host did not resolve to any address".to_string());
            }
            for addr in &addrs {
                if blocked_ip(&addr.ip()) {
                    return Err(
                        "source_uri host must not be private, loopback, or link-local".to_string(),
                    );
                }
            }
        }
        Err(_) => {
            // DNS resolution failed — reject rather than allow a potentially
            // unvalidated host through.
            return Err("source_uri host could not be resolved".to_string());
        }
    }

    Ok(InputSource::Url(url.to_string()))
}

/// Validate an HLS URL (http/https .m3u8 or hls:// scheme).
///
/// Applies the same SSRF guards as [`validate_remote_url`] (no embedded
/// credentials, no private/loopback hosts, DNS rebinding check), then
/// returns [`InputSource::HlsStream`].
///
/// For `hls://` URLs the host validation re-parses the URL as `https://`
/// so that `reqwest::Url::host_str()` and DNS resolution work correctly.
fn validate_hls_url(url: reqwest::Url) -> Result<InputSource, String> {
    if !url.username().is_empty() || url.password().is_some() {
        return Err("source_uri must not contain embedded credentials".to_string());
    }

    // For hls:// scheme, substitute https:// for SSRF host validation.
    let validation_url = if url.scheme() == "hls" {
        let https_str = format!("https{}", &url.as_str()["hls".len()..]);
        reqwest::Url::parse(&https_str)
            .map_err(|_| "invalid hls:// source_uri host".to_string())?
    } else {
        url.clone()
    };

    let host = validation_url
        .host_str()
        .ok_or_else(|| "source_uri must include a valid host".to_string())?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
        return Err("source_uri host must not target localhost/local domains".to_string());
    }
    if let Ok(ip) = lower.parse::<IpAddr>() {
        if blocked_ip(&ip) {
            return Err("source_uri host must not be private, loopback, or link-local".to_string());
        }
    }

    let port = validation_url.port_or_known_default().unwrap_or(443);
    let resolve_target = format!("{host}:{port}");
    match resolve_target.to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            if addrs.is_empty() {
                return Err("source_uri host did not resolve to any address".to_string());
            }
            for addr in &addrs {
                if blocked_ip(&addr.ip()) {
                    return Err(
                        "source_uri host must not be private, loopback, or link-local".to_string(),
                    );
                }
            }
        }
        Err(_) => {
            return Err("source_uri host could not be resolved".to_string());
        }
    }

    Ok(InputSource::HlsStream(url.to_string()))
}

fn validate_file_url(
    url: reqwest::Url,
    allowed_file_roots: &[PathBuf],
) -> Result<InputSource, String> {
    if url.host_str().map(|host| !host.is_empty()).unwrap_or(false) {
        return Err("file:// source_uri must not include a host".to_string());
    }
    let path = url
        .to_file_path()
        .map_err(|_| "file:// source_uri path is invalid".to_string())?;
    validate_file_path(path.to_string_lossy().as_ref(), allowed_file_roots)
}

fn validate_file_path(path: &str, allowed_file_roots: &[PathBuf]) -> Result<InputSource, String> {
    let canonical = Path::new(path)
        .canonicalize()
        .map_err(|_| "source_uri file path is invalid or does not exist".to_string())?;
    if !allowed_file_roots
        .iter()
        .any(|root| canonical.starts_with(root))
    {
        return Err("source_uri file path is outside configured ingest roots".to_string());
    }
    Ok(InputSource::FilePath(
        canonical.to_string_lossy().to_string(),
    ))
}

fn blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
                || (octets[0] == 169 && octets[1] == 254)
        }
        IpAddr::V6(v6) => {
            let octets = v6.octets();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6.is_multicast()
                || (octets[0] == 0x20
                    && octets[1] == 0x01
                    && octets[2] == 0x0d
                    && octets[3] == 0xb8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InputSource;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_url_and_file_sources() {
        let allowed = vec![std::env::temp_dir()];
        assert!(matches!(
            InputSource::parse_and_validate("https://example.com/video.mp4", &allowed).unwrap(),
            InputSource::Url(_)
        ));

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vidarax-ingest-test-{nanos}"));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let video = root.join("video.mp4");
        fs::write(&video, b"fixture").unwrap();
        let source = InputSource::parse_and_validate(video.to_string_lossy().as_ref(), &[root])
            .expect("file path should be valid");
        assert!(matches!(source, InputSource::FilePath(_)));
    }

    #[test]
    fn rejects_private_or_metadata_hosts() {
        let allowed = vec![std::env::temp_dir()];
        assert!(InputSource::parse_and_validate("http://127.0.0.1/video.mp4", &allowed).is_err());
        assert!(InputSource::parse_and_validate(
            "http://169.254.169.254/latest/meta-data",
            &allowed
        )
        .is_err());
        assert!(InputSource::parse_and_validate("https://localhost/video.mp4", &allowed).is_err());
    }

    #[test]
    fn rejects_paths_outside_allowed_roots() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vidarax-ingest-root-{nanos}"));
        let outside = std::env::temp_dir().join(format!("vidarax-ingest-outside-{nanos}.mp4"));
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        fs::write(&outside, b"fixture").unwrap();
        let result = InputSource::parse_and_validate(outside.to_string_lossy().as_ref(), &[root]);
        assert!(result.is_err(), "outside path should fail allowlist check");
    }

    #[test]
    fn accepts_https_m3u8_as_hls_stream() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("https://example.com/live/stream.m3u8", &allowed);
        assert!(result.is_ok(), "https .m3u8 should be valid: {result:?}");
        assert!(
            matches!(result.unwrap(), InputSource::HlsStream(_)),
            "expected HlsStream variant"
        );
    }

    #[test]
    fn accepts_http_m3u8_as_hls_stream() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("http://example.com/live/stream.m3u8", &allowed);
        assert!(result.is_ok(), "http .m3u8 should be valid: {result:?}");
        assert!(matches!(result.unwrap(), InputSource::HlsStream(_)));
    }

    #[test]
    fn rejects_hls_with_embedded_credentials() {
        let allowed = vec![std::env::temp_dir()];
        assert!(
            InputSource::parse_and_validate(
                "https://user:pass@example.com/stream.m3u8",
                &allowed
            )
            .is_err(),
            "credentials in HLS URL should be rejected"
        );
    }

    #[test]
    fn rejects_hls_targeting_private_host() {
        let allowed = vec![std::env::temp_dir()];
        assert!(
            InputSource::parse_and_validate("https://192.168.1.100/stream.m3u8", &allowed)
                .is_err(),
            "private IP in HLS URL should be rejected"
        );
    }

    #[test]
    fn non_m3u8_https_remains_url_variant() {
        let allowed = vec![std::env::temp_dir()];
        let result =
            InputSource::parse_and_validate("https://example.com/video.mp4", &allowed).unwrap();
        assert!(
            matches!(result, InputSource::Url(_)),
            "non-.m3u8 https should be Url, not HlsStream"
        );
    }

    #[test]
    fn hls_ffmpeg_input_returns_url_string() {
        let src = InputSource::HlsStream("https://example.com/live.m3u8".to_string());
        assert_eq!(src.as_ffmpeg_input(), "https://example.com/live.m3u8");
    }
}
