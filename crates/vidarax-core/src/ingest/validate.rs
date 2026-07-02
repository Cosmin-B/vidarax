use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
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
                "http" => {
                    if !insecure_http_enabled() {
                        return Err(
                            "insecure http:// sources are not allowed; use https:// or set \
                             VIDARAX_ALLOW_INSECURE_HTTP=true"
                                .to_string(),
                        );
                    }
                    if url.path().to_ascii_lowercase().ends_with(".m3u8") {
                        validate_hls_url(url)
                    } else {
                        validate_remote_url(url)
                    }
                }
                "https" => {
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
                        return Err("unencrypted rtsp:// is not allowed; use rtsps:// or set \
                             VIDARAX_ALLOW_UNENCRYPTED_RTSP=true"
                            .to_string());
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
    Ok(InputSource::Url(validate_remote_url_string(url, 80)?))
}

/// Validate an HLS URL (http/https .m3u8 or hls:// scheme).
///
/// Applies the same SSRF guards as [`validate_remote_url`] (no embedded
/// credentials, no private/loopback hosts, DNS resolution check), then
/// returns [`InputSource::HlsStream`].
///
/// For `hls://` URLs the stored ffmpeg input is normalized to a real transport
/// URL (`https://` by default, or `http://` only when insecure HTTP is opted in).
fn validate_hls_url(url: reqwest::Url) -> Result<InputSource, String> {
    if !remote_hls_enabled() {
        return Err(
            "remote HLS sources are disabled by default; set VIDARAX_ALLOW_REMOTE_HLS=true \
             only for trusted manifests"
                .to_string(),
        );
    }
    Ok(InputSource::HlsStream(validate_hls_url_string(url)?))
}

fn validate_remote_url_string(url: reqwest::Url, default_port: u16) -> Result<String, String> {
    validate_remote_url_string_with_resolver(url, default_port, validate_resolved_public_host)
}

pub(crate) fn validate_remote_fetch_url(url: &reqwest::Url) -> Result<Vec<SocketAddr>, String> {
    match url.scheme() {
        "https" => {}
        "http" => {
            if !insecure_http_enabled() {
                return Err("insecure http scheme".to_string());
            }
        }
        _ => return Err("remote media fetch only supports http/https URLs".to_string()),
    }
    let host = validate_remote_host(url)?;
    let port = url.port_or_known_default().unwrap_or(match url.scheme() {
        "http" => 80,
        _ => 443,
    });
    validate_resolved_public_host(host, port)
}

fn validate_remote_url_string_with_resolver<F, R>(
    url: reqwest::Url,
    default_port: u16,
    mut validate_resolved: F,
) -> Result<String, String>
where
    F: FnMut(&str, u16) -> Result<R, String>,
{
    let host = validate_remote_host(&url)?;
    let port = url.port_or_known_default().unwrap_or(default_port);
    validate_resolved(host, port)?;
    // We intentionally preserve the original host for TLS SNI, certificate
    // identity, and HTTP Host routing. DNS can still rebind after validation;
    // that sub-second TOCTOU is accepted until ffmpeg can be given a verified
    // connection or a host-preserving DNS pinning mechanism. See
    // docs/security.md for the network egress control required to fully close
    // redirect and nested-resource SSRF residuals.
    Ok(url.to_string())
}

fn validate_hls_url_string(url: reqwest::Url) -> Result<String, String> {
    validate_hls_url_string_with_resolver(url, validate_resolved_public_host)
}

fn validate_hls_url_string_with_resolver<F, R>(
    url: reqwest::Url,
    mut validate_resolved: F,
) -> Result<String, String>
where
    F: FnMut(&str, u16) -> Result<R, String>,
{
    let transport_url = if url.scheme() == "hls" {
        let transport_scheme = if insecure_http_enabled() {
            "http"
        } else {
            "https"
        };
        let transport_str = format!("{transport_scheme}{}", &url.as_str()["hls".len()..]);
        reqwest::Url::parse(&transport_str)
            .map_err(|_| "invalid hls:// source_uri host".to_string())?
    } else {
        url.clone()
    };
    let host = validate_remote_host(&transport_url)?;
    let port = transport_url
        .port_or_known_default()
        .unwrap_or(match transport_url.scheme() {
            "http" => 80,
            _ => 443,
        });
    validate_resolved(host, port)?;
    // Remote HLS remains opt-in because ffmpeg may fetch absolute segment,
    // variant, key, and map URIs from a manifest that changes after this check.
    // See docs/security.md for the network egress control required to fully
    // close redirect and nested-resource SSRF residuals.
    Ok(transport_url.to_string())
}

fn remote_hls_enabled() -> bool {
    std::env::var("VIDARAX_ALLOW_REMOTE_HLS")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

pub(crate) fn insecure_http_enabled() -> bool {
    std::env::var("VIDARAX_ALLOW_INSECURE_HTTP")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

fn validate_remote_host(url: &reqwest::Url) -> Result<&str, String> {
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
    Ok(host)
}

fn validate_resolved_public_host(host: &str, port: u16) -> Result<Vec<SocketAddr>, String> {
    let resolve_target = format!("{host}:{port}");
    let addrs: Vec<SocketAddr> = resolve_target
        .to_socket_addrs()
        .map_err(|_| "source_uri host could not be resolved".to_string())?
        .collect();
    if addrs.is_empty() {
        return Err("source_uri host did not resolve to any address".to_string());
    }
    for addr in &addrs {
        if blocked_ip(&addr.ip()) {
            return Err("source_uri host must not be private, loopback, or link-local".to_string());
        }
    }
    Ok(addrs)
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
                || v4.is_multicast()
                || octets[0] == 0
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
                || (octets[0] == 169 && octets[1] == 254)
                || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
                || (octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
                || octets[0] >= 240
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return blocked_ip(&IpAddr::V4(v4));
            }
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

    struct EnvRestore {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            match self.old.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn with_env<T>(key: &'static str, value: Option<&str>, test: impl FnOnce() -> T) -> T {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old = std::env::var_os(key);
        let _restore = EnvRestore { key, old };
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        test()
    }

    fn with_remote_hls_env<T>(value: Option<&str>, test: impl FnOnce() -> T) -> T {
        with_env("VIDARAX_ALLOW_REMOTE_HLS", value, test)
    }

    fn with_insecure_http_env<T>(value: Option<&str>, test: impl FnOnce() -> T) -> T {
        with_env("VIDARAX_ALLOW_INSECURE_HTTP", value, test)
    }

    fn validate_remote_without_dns(raw: &str, expected_port: u16) -> String {
        let url = reqwest::Url::parse(raw).unwrap();
        super::validate_remote_url_string_with_resolver(url, 80, |host, port| {
            assert_eq!(host, "example.com");
            assert_eq!(port, expected_port);
            Ok(())
        })
        .unwrap()
    }

    fn validate_hls_without_dns(raw: &str, expected_port: u16) -> String {
        let url = reqwest::Url::parse(raw).unwrap();
        super::validate_hls_url_string_with_resolver(url, |host, port| {
            assert_eq!(host, "example.com");
            assert_eq!(port, expected_port);
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn parses_url_and_file_sources() {
        let remote = validate_remote_without_dns("https://example.com/video.mp4", 443);
        assert!(matches!(InputSource::Url(remote), InputSource::Url(_)));

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
    fn rejects_plain_http_remote_sources_by_default() {
        with_insecure_http_env(None, || {
            let allowed = vec![std::env::temp_dir()];
            let result = InputSource::parse_and_validate("http://example.com/video.mp4", &allowed);
            assert!(result.is_err(), "plain http should require explicit opt-in");
        });
    }

    #[test]
    fn opt_in_plain_http_remote_sources_preserve_hostname() {
        with_insecure_http_env(Some("true"), || {
            let result = InputSource::Url(validate_remote_without_dns(
                "http://example.com/video.mp4",
                80,
            ));
            let ffmpeg_input = reqwest::Url::parse(result.as_ffmpeg_input()).unwrap();
            assert_eq!(ffmpeg_input.host_str(), Some("example.com"));
        });
    }

    #[test]
    fn rejects_private_or_metadata_hosts() {
        let allowed = vec![std::env::temp_dir()];
        assert!(InputSource::parse_and_validate("http://127.0.0.1/video.mp4", &allowed).is_err());
        assert!(
            InputSource::parse_and_validate("http://[::ffff:127.0.0.1]/video.mp4", &allowed)
                .is_err()
        );
        assert!(InputSource::parse_and_validate(
            "http://169.254.169.254/latest/meta-data",
            &allowed
        )
        .is_err());
        assert!(InputSource::parse_and_validate("https://localhost/video.mp4", &allowed).is_err());
    }

    #[test]
    fn rejects_non_global_ipv4_literals() {
        let blocked = ["224.0.0.1", "240.0.0.1", "100.64.0.1", "255.255.255.255"];
        for host in blocked {
            let ip = host.parse().unwrap();
            assert!(super::blocked_ip(&ip), "{host} should be blocked");
            let url = format!("https://{host}/video.mp4");
            assert!(
                InputSource::parse_and_validate(&url, &[std::env::temp_dir()]).is_err(),
                "{url} should be rejected"
            );
        }
    }

    #[test]
    fn live_https_hostname_is_preserved_after_real_dns_validation() {
        let source = InputSource::parse_and_validate(
            "https://example.com/video.mp4",
            &[std::env::temp_dir()],
        )
        .expect("public HTTPS hostname should pass real DNS validation");
        let ffmpeg_input = reqwest::Url::parse(source.as_ffmpeg_input()).unwrap();
        assert_eq!(ffmpeg_input.host_str(), Some("example.com"));
    }

    #[test]
    fn live_public_hostname_that_resolves_is_accepted() {
        let source = InputSource::parse_and_validate("https://example.com/media/movie.mp4", &[])
            .expect("public hostname with real DNS should be accepted");
        assert!(matches!(source, InputSource::Url(_)));
    }

    #[test]
    fn live_ipv4_mapped_ipv6_loopback_is_rejected() {
        let result = InputSource::parse_and_validate("https://[::ffff:127.0.0.1]/video.mp4", &[]);
        assert!(result.is_err(), "IPv4-mapped IPv6 loopback must be blocked");
    }

    #[test]
    fn live_opt_in_http_preserves_hostname_after_real_dns_validation() {
        with_insecure_http_env(Some("true"), || {
            let source = InputSource::parse_and_validate(
                "http://example.com/video.mp4",
                &[std::env::temp_dir()],
            )
            .expect("opt-in HTTP public hostname should pass real DNS validation");
            let ffmpeg_input = reqwest::Url::parse(source.as_ffmpeg_input()).unwrap();
            assert_eq!(ffmpeg_input.host_str(), Some("example.com"));
        });
    }

    #[test]
    fn live_private_and_metadata_literals_are_rejected() {
        let urls = [
            "https://127.0.0.1/video.mp4",
            "https://10.0.0.1/video.mp4",
            "https://169.254.169.254/latest/meta-data",
        ];
        for url in urls {
            assert!(
                InputSource::parse_and_validate(url, &[std::env::temp_dir()]).is_err(),
                "{url} should be rejected"
            );
        }
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
    fn rejects_remote_hls_by_default() {
        with_remote_hls_env(None, || {
            let allowed = vec![std::env::temp_dir()];
            let result =
                InputSource::parse_and_validate("https://example.com/live/stream.m3u8", &allowed);
            assert!(result.is_err(), "remote HLS should require explicit opt-in");
        });
    }

    #[test]
    fn opt_in_remote_hls_preserves_original_hostname_for_tls() {
        with_remote_hls_env(Some("true"), || {
            let result = InputSource::HlsStream(validate_hls_without_dns(
                "https://example.com/live/stream.m3u8",
                443,
            ));
            let ffmpeg_input = reqwest::Url::parse(result.as_ffmpeg_input()).unwrap();
            assert_eq!(ffmpeg_input.host_str(), Some("example.com"));
        });
    }

    #[test]
    fn hls_scheme_normalizes_to_https_transport_by_default() {
        with_insecure_http_env(None, || {
            let stored = validate_hls_without_dns("hls://example.com/live.m3u8", 443);

            assert_eq!(stored, "https://example.com/live.m3u8");
        });
    }

    #[test]
    fn hls_scheme_normalizes_to_http_transport_only_with_insecure_http_opt_in() {
        with_insecure_http_env(Some("true"), || {
            let stored = validate_hls_without_dns("hls://example.com/live.m3u8", 80);

            assert_eq!(stored, "http://example.com/live.m3u8");
        });
    }

    #[test]
    fn rejects_hls_with_embedded_credentials() {
        with_remote_hls_env(Some("true"), || {
            let allowed = vec![std::env::temp_dir()];
            assert!(
                InputSource::parse_and_validate(
                    "https://user:pass@example.com/stream.m3u8",
                    &allowed
                )
                .is_err(),
                "credentials in HLS URL should be rejected"
            );
        });
    }

    #[test]
    fn rejects_hls_targeting_private_host() {
        with_remote_hls_env(Some("true"), || {
            let allowed = vec![std::env::temp_dir()];
            assert!(
                InputSource::parse_and_validate("https://192.168.1.100/stream.m3u8", &allowed)
                    .is_err(),
                "private IP in HLS URL should be rejected"
            );
        });
    }

    #[test]
    fn non_m3u8_https_remains_url_variant() {
        let result = InputSource::Url(validate_remote_without_dns(
            "https://example.com/video.mp4",
            443,
        ));
        assert!(
            matches!(result, InputSource::Url(_)),
            "non-.m3u8 https should be Url, not HlsStream"
        );
    }

    #[test]
    fn remote_url_preserves_original_hostname_for_ffmpeg_tls() {
        let result = InputSource::Url(validate_remote_without_dns(
            "https://example.com/video.mp4",
            443,
        ));
        let ffmpeg_input = reqwest::Url::parse(result.as_ffmpeg_input()).unwrap();
        assert_eq!(ffmpeg_input.host_str(), Some("example.com"));
    }

    #[test]
    fn hls_ffmpeg_input_returns_url_string() {
        let src = InputSource::HlsStream("https://example.com/live.m3u8".to_string());
        assert_eq!(src.as_ffmpeg_input(), "https://example.com/live.m3u8");
    }
}
