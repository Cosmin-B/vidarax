use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use reqwest::blocking::Client;
use reqwest::redirect;

use super::ffmpeg::{ffprobe_path, FFMPEG_LOCAL_PROTOCOL_WHITELIST};
use super::validate::{validate_remote_fetch_url, InputSource};

pub const REMOTE_MEDIA_PREFETCH_MAX_BYTES: u64 = 256 * 1024 * 1024;
const REMOTE_MEDIA_PREFETCH_TIMEOUT: Duration = Duration::from_secs(30);
const REMOTE_MEDIA_PREFETCH_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const REMOTE_MEDIA_PREFETCH_MAX_REDIRECTS: usize = 5;
const REMOTE_MEDIA_PREFETCH_PROBE_SIZE: &str = "1048576";
const REMOTE_MEDIA_PREFETCH_ANALYZE_DURATION_US: &str = "5000000";
const PREFETCH_DIR_NAME: &str = "vidarax-prefetches";
static PREFETCH_COUNTER: AtomicU64 = AtomicU64::new(0);
pub(crate) type FetchUrlValidator =
    Arc<dyn Fn(&reqwest::Url) -> Result<Vec<SocketAddr>, String> + Send + Sync + 'static>;

pub(crate) fn with_prefetched_downloadable_source<T>(
    source: &InputSource,
    op: impl FnOnce(&InputSource) -> T,
) -> Result<T, String> {
    if !is_downloadable_remote_url(source) {
        return Ok(op(source));
    }

    let prefetched = prefetch_remote_media(source.as_ffmpeg_input())?;
    let local_source = InputSource::FilePath(prefetched.path.to_string_lossy().to_string());
    Ok(op(&local_source))
}

fn is_downloadable_remote_url(source: &InputSource) -> bool {
    matches!(source, InputSource::Url(url) if url.starts_with("https://") || url.starts_with("http://"))
}

/// A media source made ready for repeated decode passes. A downloadable remote
/// URL is fetched once into a temp file that lives as long as this guard, and
/// every read of [`source`](Self::source) hands back that local path. Reuse it
/// across probe, signal decode, JPEG decode, and per-chunk clip extraction so
/// none of them re-download the same bytes. A local or non-downloadable source
/// passes through untouched and holds no temp file.
pub struct PreparedSource {
    source: InputSource,
    // Kept only for its Drop: deleting the temp file once the guard falls out of
    // scope. A local passthrough leaves this None.
    _prefetched: Option<PrefetchedMedia>,
}

impl PreparedSource {
    /// The source to hand every decode call. Points at the local temp file for a
    /// prefetched remote URL, or the original source when nothing was fetched.
    pub fn source(&self) -> &InputSource {
        &self.source
    }
}

/// Fetch a downloadable remote source once so later decode passes reuse the local
/// copy instead of re-downloading per call. Applies the same URL validation,
/// address pinning, and redirect rules as the per-call prefetch path. The caller
/// must keep the returned guard alive until every decode over the source is done.
pub fn prepare_source_for_reuse(source: &InputSource) -> Result<PreparedSource, String> {
    prepare_source_for_reuse_with_validator(source, Arc::new(validate_remote_fetch_url))
}

fn prepare_source_for_reuse_with_validator(
    source: &InputSource,
    validate_url: FetchUrlValidator,
) -> Result<PreparedSource, String> {
    if !is_downloadable_remote_url(source) {
        return Ok(PreparedSource {
            source: source.clone(),
            _prefetched: None,
        });
    }

    let prefetched = prefetch_remote_media_with_limit_and_validator(
        source.as_ffmpeg_input(),
        REMOTE_MEDIA_PREFETCH_MAX_BYTES,
        validate_url,
    )?;
    let source = InputSource::FilePath(prefetched.path.to_string_lossy().to_string());
    Ok(PreparedSource {
        source,
        _prefetched: Some(prefetched),
    })
}

pub(crate) fn prefetch_remote_media(url: &str) -> Result<PrefetchedMedia, String> {
    prefetch_remote_media_with_limit(url, REMOTE_MEDIA_PREFETCH_MAX_BYTES)
}

pub(crate) fn prefetch_remote_media_with_limit(
    url: &str,
    max_bytes: u64,
) -> Result<PrefetchedMedia, String> {
    prefetch_remote_media_with_limit_and_validator(
        url,
        max_bytes,
        Arc::new(validate_remote_fetch_url),
    )
}

fn prefetch_remote_media_with_limit_and_validator(
    url: &str,
    max_bytes: u64,
    validate_url: FetchUrlValidator,
) -> Result<PrefetchedMedia, String> {
    let parsed =
        reqwest::Url::parse(url).map_err(|err| format!("invalid remote media URL: {err}"))?;
    let validated_addrs = validate_url(&parsed)?;
    let started = Instant::now();

    let mut response = fetch_remote_media_response(parsed, validated_addrs, validate_url, started)?;
    if !response.status().is_success() {
        return Err(format!(
            "remote media fetch failed with HTTP status {}",
            response.status()
        ));
    }
    if response
        .content_length()
        .is_some_and(|content_length| content_length > max_bytes)
    {
        return Err("remote media exceeds prefetch size limit".to_string());
    }

    let (path, mut file) = create_prefetch_file()?;
    // Own the temp file from here on. Every early return below then deletes it
    // through PrefetchedMedia::drop, including the timeout, read, write, and
    // sync bails that previously left a partial file behind.
    let prefetched = PrefetchedMedia { path };
    let mut total = 0u64;
    let mut first_bytes = Vec::with_capacity(64);
    let mut buf = [0u8; 64 * 1024];
    loop {
        ensure_prefetch_time_remaining(started)?;
        let read = response
            .read(&mut buf)
            .map_err(|err| format!("remote media fetch failed: {err}"))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > max_bytes {
            return Err("remote media exceeds prefetch size limit".to_string());
        }
        if first_bytes.len() < 64 {
            let remaining = 64 - first_bytes.len();
            first_bytes.extend_from_slice(&buf[..read.min(remaining)]);
        }
        file.write_all(&buf[..read])
            .map_err(|err| format!("failed to store remote media: {err}"))?;
    }
    file.sync_all()
        .map_err(|err| format!("failed to store remote media: {err}"))?;
    drop(file);

    if total == 0 {
        return Err("remote media response was empty".to_string());
    }
    if has_extm3u_magic(&first_bytes) {
        return Err("remote media must be a media container, not a playlist manifest".to_string());
    }
    validate_prefetched_media_container(&prefetched.path)?;

    Ok(prefetched)
}

fn fetch_remote_media_response(
    mut url: reqwest::Url,
    mut validated_addrs: Vec<SocketAddr>,
    validate_url: FetchUrlValidator,
    started: Instant,
) -> Result<reqwest::blocking::Response, String> {
    for redirect_count in 0..=REMOTE_MEDIA_PREFETCH_MAX_REDIRECTS {
        let client = pinned_fetch_client(&url, &validated_addrs)?;
        let response = client
            .get(url.clone())
            .timeout(prefetch_time_remaining(started)?)
            .send()
            .map_err(|err| format!("remote media fetch failed: {err}"))?;

        if !response.status().is_redirection() {
            return Ok(response);
        }

        let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
            return Ok(response);
        };
        if redirect_count == REMOTE_MEDIA_PREFETCH_MAX_REDIRECTS {
            return Err("too many redirects".to_string());
        }

        let location = location
            .to_str()
            .map_err(|err| format!("invalid redirect Location header: {err}"))?;
        let next_url = url
            .join(location)
            .map_err(|err| format!("invalid redirect Location header: {err}"))?;
        validated_addrs = validate_redirect_target(&url, &next_url, &validate_url)?;
        url = next_url;
    }

    Err("too many redirects".to_string())
}

fn pinned_fetch_client(
    url: &reqwest::Url,
    validated_addrs: &[SocketAddr],
) -> Result<Client, String> {
    let host = url
        .host_str()
        .ok_or_else(|| "remote media fetch URL must include a host".to_string())?;
    Client::builder()
        .redirect(redirect::Policy::none())
        .resolve_to_addrs(host, validated_addrs)
        // This fetch path must dial the addresses that passed validation. A proxy
        // would resolve the origin name itself and skip the pinned address list.
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build remote media fetch client: {err}"))
}

fn validate_redirect_target(
    current_url: &reqwest::Url,
    next_url: &reqwest::Url,
    validate_url: &FetchUrlValidator,
) -> Result<Vec<SocketAddr>, String> {
    if !matches!(next_url.scheme(), "http" | "https") {
        return Err("redirect target rejected: unsupported scheme".to_string());
    }
    if current_url.scheme() == "https" && next_url.scheme() == "http" {
        return Err("redirect target rejected: https->http downgrade".to_string());
    }
    match validate_url(next_url) {
        Ok(addrs) => Ok(addrs),
        Err(err) => {
            if err == "insecure http scheme" {
                return Err("redirect target rejected: insecure http scheme".to_string());
            }
            let host = next_url.host_str().unwrap_or("<missing host>");
            Err(format!("redirect target rejected: {host} ({err})"))
        }
    }
}

fn prefetch_time_remaining(started: Instant) -> Result<Duration, String> {
    REMOTE_MEDIA_PREFETCH_TIMEOUT
        .checked_sub(started.elapsed())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| "remote media fetch timed out".to_string())
}

fn ensure_prefetch_time_remaining(started: Instant) -> Result<(), String> {
    prefetch_time_remaining(started).map(|_| ())
}

#[cfg(test)]
pub(crate) fn prefetch_remote_media_with_limit_and_validator_for_test(
    url: &str,
    max_bytes: u64,
    validate_url: FetchUrlValidator,
) -> Result<PrefetchedMedia, String> {
    prefetch_remote_media_with_limit_and_validator(url, max_bytes, validate_url)
}

#[cfg(test)]
pub(crate) fn with_prefetched_downloadable_source_and_validator<T>(
    source: &InputSource,
    validate_url: FetchUrlValidator,
    op: impl FnOnce(&InputSource) -> T,
) -> Result<T, String> {
    if !is_downloadable_remote_url(source) {
        return Ok(op(source));
    }

    let prefetched = prefetch_remote_media_with_limit_and_validator(
        source.as_ffmpeg_input(),
        REMOTE_MEDIA_PREFETCH_MAX_BYTES,
        validate_url,
    )?;
    let local_source = InputSource::FilePath(prefetched.path.to_string_lossy().to_string());
    Ok(op(&local_source))
}

fn create_prefetch_file() -> Result<(PathBuf, File), String> {
    let dir = std::env::temp_dir().join(PREFETCH_DIR_NAME);
    fs::create_dir_all(&dir).map_err(|err| format!("failed to create prefetch temp dir: {err}"))?;
    for _ in 0..16 {
        let path = dir.join(format!(
            "media-{}-{}-{}.bin",
            std::process::id(),
            now_nanos(),
            PREFETCH_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(format!("failed to create prefetch temp file: {err}")),
        }
    }
    Err("failed to create unique prefetch temp file".to_string())
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn has_extm3u_magic(data: &[u8]) -> bool {
    let trimmed = data
        .iter()
        .copied()
        .skip_while(|b| b.is_ascii_whitespace())
        .take(7)
        .collect::<Vec<_>>();
    trimmed.eq_ignore_ascii_case(b"#EXTM3U")
}

fn validate_prefetched_media_container(path: &Path) -> Result<(), String> {
    validate_prefetched_media_container_with_probe(
        path,
        ffprobe_path(),
        &[],
        REMOTE_MEDIA_PREFETCH_PROBE_TIMEOUT,
    )
}

fn validate_prefetched_media_container_with_probe(
    path: &Path,
    probe_path: &str,
    extra_probe_args: &[&str],
    timeout: Duration,
) -> Result<(), String> {
    let mut command = Command::new(probe_path);
    command.args(extra_probe_args);
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            FFMPEG_LOCAL_PROTOCOL_WHITELIST,
            "-probesize",
            REMOTE_MEDIA_PREFETCH_PROBE_SIZE,
            "-analyzeduration",
            REMOTE_MEDIA_PREFETCH_ANALYZE_DURATION_US,
            "-show_entries",
            "format=format_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path);
    let output = run_probe_command_with_timeout(&mut command, timeout)?;
    if !output.status.success() {
        return Err("remote media must be a valid media container".to_string());
    }
    let format_name = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_ascii_lowercase();
    if format_name.is_empty() {
        return Err("remote media must declare a media container format".to_string());
    }
    if format_name
        .split(',')
        .any(|name| matches!(name, "hls" | "concat"))
    {
        return Err("remote media must be a media container, not a playlist manifest".to_string());
    }
    Ok(())
}

#[cfg(test)]
fn validate_prefetched_media_container_with_probe_for_test(
    path: &Path,
    probe_path: &str,
    extra_probe_args: &[&str],
    timeout: Duration,
) -> Result<(), String> {
    validate_prefetched_media_container_with_probe(path, probe_path, extra_probe_args, timeout)
}

struct ProbeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
}

fn run_probe_command_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> Result<ProbeOutput, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|_| "failed to inspect prefetched media".to_string())?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                if let Some(mut pipe) = child.stdout.take() {
                    pipe.read_to_end(&mut stdout)
                        .map_err(|_| "failed to inspect prefetched media".to_string())?;
                }
                return Ok(ProbeOutput { status, stdout });
            }
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("remote media probe timed out".to_string());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("failed to inspect prefetched media".to_string());
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct PrefetchedMedia {
    path: PathBuf,
}

impl Drop for PrefetchedMedia {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        has_extm3u_magic, is_downloadable_remote_url,
        prefetch_remote_media_with_limit_and_validator_for_test,
        prepare_source_for_reuse_with_validator,
        validate_prefetched_media_container_with_probe_for_test, validate_redirect_target,
        InputSource, REMOTE_MEDIA_PREFETCH_MAX_BYTES,
    };
    use std::fs;
    use std::process::Command;
    use std::sync::Arc;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
        with_envs(&[(key, value)], test)
    }

    fn with_envs<T>(values: &[(&'static str, Option<&str>)], test: impl FnOnce() -> T) -> T {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _restore = values
            .iter()
            .map(|(key, _)| EnvRestore {
                key,
                old: std::env::var_os(key),
            })
            .collect::<Vec<_>>();
        for (key, value) in values {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        test()
    }

    #[test]
    fn detects_extm3u_magic_with_leading_whitespace() {
        assert!(has_extm3u_magic(b"\n\t #EXTM3U\n#EXT-X-VERSION:3"));
        assert!(!has_extm3u_magic(b"\0\0\0\x18ftypmp42"));
    }

    #[test]
    fn public_https_playlist_body_is_rejected() {
        let server = super::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "application/vnd.apple.mpegurl")],
            b"#EXTM3U\n#EXT-X-VERSION:3\n".to_vec(),
        );
        let err = prefetch_remote_media_with_limit_and_validator_for_test(
            &server.url("/playlist"),
            REMOTE_MEDIA_PREFETCH_MAX_BYTES,
            server.allow_origin_validator(),
        )
        .expect_err("playlist body must be rejected");
        assert!(err.contains("playlist manifest"), "{err}");
    }

    #[test]
    fn public_https_redirect_to_private_host_is_rejected() {
        with_env("VIDARAX_ALLOW_INSECURE_HTTP", Some("true"), || {
            let server = super::test_helpers::MockHttpServer::serve_once(
                "302 Found",
                &[("Location", "http://127.0.0.1:9/media.mp4")],
                Vec::new(),
            );
            let err = prefetch_remote_media_with_limit_and_validator_for_test(
                &server.url("/redirect"),
                REMOTE_MEDIA_PREFETCH_MAX_BYTES,
                server.allow_origin_validator(),
            )
            .expect_err("redirect to loopback must be rejected");
            assert!(err.contains("redirect target rejected"), "{err}");
            assert!(err.contains("127.0.0.1"), "{err}");
            assert!(err.contains("private") || err.contains("loopback"), "{err}");
        });
    }

    #[test]
    fn https_redirect_to_plain_http_is_rejected_before_url_validator() {
        let current = reqwest::Url::parse("https://example.com/media.mp4").unwrap();
        let next = reqwest::Url::parse("http://127.0.0.1:9/media.mp4").unwrap();
        let allow_all: super::FetchUrlValidator = Arc::new(|url: &reqwest::Url| {
            Ok(vec![std::net::SocketAddr::new(
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                url.port_or_known_default().unwrap_or(80),
            )])
        });

        let err = validate_redirect_target(&current, &next, &allow_all)
            .expect_err("https to http downgrade must be rejected");

        assert_eq!(err, "redirect target rejected: https->http downgrade");
    }

    #[test]
    fn redirect_to_plain_http_is_rejected_when_insecure_http_is_off() {
        with_env("VIDARAX_ALLOW_INSECURE_HTTP", None, || {
            let current = reqwest::Url::parse("http://example.com/media.mp4").unwrap();
            let next = reqwest::Url::parse("http://example.org/media.mp4").unwrap();
            let validator: super::FetchUrlValidator = Arc::new(super::validate_remote_fetch_url);

            let err = validate_redirect_target(&current, &next, &validator)
                .expect_err("plain http redirect must require opt-in");

            assert_eq!(err, "redirect target rejected: insecure http scheme");
        });
    }

    #[test]
    fn mock_origin_redirect_allow_seam_still_follows_local_redirect() {
        let server = super::test_helpers::MockHttpServer::serve_responses(vec![
            super::test_helpers::MockResponse {
                status: "302 Found",
                headers: &[("Location", "/playlist")],
                body: Vec::new(),
            },
            super::test_helpers::MockResponse {
                status: "200 OK",
                headers: &[("Content-Type", "application/vnd.apple.mpegurl")],
                body: b"#EXTM3U\n#EXT-X-VERSION:3\n".to_vec(),
            },
        ]);
        let err = prefetch_remote_media_with_limit_and_validator_for_test(
            &server.url("/redirect"),
            REMOTE_MEDIA_PREFETCH_MAX_BYTES,
            server.allow_origin_validator(),
        )
        .expect_err("redirected playlist body must be reached and rejected");

        assert!(err.contains("playlist manifest"), "{err}");
    }

    #[test]
    fn prefetch_connects_to_validated_address_for_hostname() {
        let server = super::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "application/vnd.apple.mpegurl")],
            b"#EXTM3U\n#EXT-X-VERSION:3\n".to_vec(),
        );
        let err = prefetch_remote_media_with_limit_and_validator_for_test(
            &server.url_for_host("vidarax-prefetch.test", "/playlist"),
            REMOTE_MEDIA_PREFETCH_MAX_BYTES,
            server.allow_host_validator("vidarax-prefetch.test"),
        )
        .expect_err("pinned hostname request should reach the mock server");

        assert!(err.contains("playlist manifest"), "{err}");
    }

    #[test]
    fn prefetch_proxy_env_cannot_bypass_pinned_address() {
        let proxy = super::test_helpers::ProxyTrap::serve();
        let proxy_url = proxy.url();
        let server = super::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "video/mp4")],
            create_test_mp4_bytes(),
        );

        with_envs(
            &[
                ("HTTP_PROXY", Some(proxy_url.as_str())),
                ("HTTPS_PROXY", Some(proxy_url.as_str())),
                ("ALL_PROXY", Some(proxy_url.as_str())),
                ("http_proxy", Some(proxy_url.as_str())),
                ("https_proxy", Some(proxy_url.as_str())),
                ("all_proxy", Some(proxy_url.as_str())),
                ("NO_PROXY", None),
                ("no_proxy", None),
            ],
            || {
                prefetch_remote_media_with_limit_and_validator_for_test(
                    &server.url_for_host("vidarax-prefetch.test", "/media.mp4"),
                    REMOTE_MEDIA_PREFETCH_MAX_BYTES,
                    server.allow_host_validator("vidarax-prefetch.test"),
                )
                .expect("prefetch must use the validated address instead of proxy env");
            },
        );

        assert!(
            !proxy.saw_connection(),
            "pinned prefetch client must not connect to proxy env listeners"
        );
    }

    #[test]
    fn public_https_oversize_body_is_rejected_by_cap() {
        let server = super::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "application/octet-stream")],
            vec![b'x'; 1024],
        );
        let err = prefetch_remote_media_with_limit_and_validator_for_test(
            &server.url("/bytes"),
            128,
            server.allow_origin_validator(),
        )
        .expect_err("body larger than cap fails");
        assert!(err.contains("size limit"), "{err}");
    }

    #[test]
    fn prepare_source_reuse_passes_through_local_source() {
        let source = InputSource::FilePath("/tmp/example.mp4".to_string());
        let prepared = prepare_source_for_reuse_with_validator(
            &source,
            Arc::new(super::validate_remote_fetch_url),
        )
        .expect("a local source needs no prefetch");
        assert_eq!(prepared.source(), &source);
    }

    #[test]
    fn prepare_source_reuse_fetches_remote_once_to_local_file() {
        let server = super::test_helpers::MockHttpServer::serve_once(
            "200 OK",
            &[("Content-Type", "video/mp4")],
            create_test_mp4_bytes(),
        );
        let prepared = prepare_source_for_reuse_with_validator(
            &InputSource::Url(server.url("/media.mp4")),
            server.allow_origin_validator(),
        )
        .expect("remote source prefetches to a local file");

        let local_path = match prepared.source() {
            InputSource::FilePath(path) => path.clone(),
            other => panic!("expected a local file source, got {other:?}"),
        };
        assert!(
            std::path::Path::new(&local_path).exists(),
            "prefetched file should exist while the guard is alive"
        );
        // Reuse reads the local copy: the source is no longer downloadable, so
        // every later decode skips the network path. The mock served its one
        // response, so a second fetch would have nothing to answer it.
        assert!(!is_downloadable_remote_url(prepared.source()));

        drop(prepared);
        assert!(
            !std::path::Path::new(&local_path).exists(),
            "dropping the guard should delete the prefetched file"
        );
    }

    #[test]
    fn prefetched_media_probe_times_out_and_kills_slow_child() {
        let path =
            std::env::temp_dir().join(format!("vidarax-prefetch-timeout-{}", std::process::id()));
        fs::write(&path, b"not used by fake ffprobe").unwrap();
        let started = Instant::now();

        let err = validate_prefetched_media_container_with_probe_for_test(
            &path,
            "sh",
            &["-c", "sleep 5"],
            Duration::from_millis(50),
        )
        .expect_err("slow ffprobe command must time out");

        assert_eq!(err, "remote media probe timed out");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "probe timeout must kill the child instead of waiting for sleep to finish"
        );
        let _ = fs::remove_file(path);
    }

    fn create_test_mp4_bytes() -> Vec<u8> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vidarax-prefetch-test-media-{}-{nanos}.mp4",
            std::process::id()
        ));
        let output = Command::new(crate::ingest::ffmpeg::ffmpeg_path())
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=1:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
            ])
            .arg(&path)
            .output()
            .expect("run ffmpeg to generate test mp4");
        assert!(
            output.status.success(),
            "ffmpeg failed to generate test mp4: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = fs::read(&path).expect("read generated test mp4");
        let _ = fs::remove_file(&path);
        assert!(!bytes.is_empty(), "generated test mp4 must not be empty");
        bytes
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::{validate_remote_fetch_url, FetchUrlValidator};
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, Sender};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    pub(crate) struct MockHttpServer {
        addr: SocketAddr,
        shutdown: Option<Sender<()>>,
        handle: Option<JoinHandle<Result<(), String>>>,
    }

    pub(crate) struct MockResponse {
        pub(crate) status: &'static str,
        pub(crate) headers: &'static [(&'static str, &'static str)],
        pub(crate) body: Vec<u8>,
    }

    pub(crate) struct ProxyTrap {
        addr: SocketAddr,
        saw_connection: Arc<AtomicBool>,
        shutdown: Option<Sender<()>>,
        handle: Option<JoinHandle<Result<(), String>>>,
    }

    impl MockHttpServer {
        pub(crate) fn serve_once(
            status: &'static str,
            headers: &'static [(&'static str, &'static str)],
            body: Vec<u8>,
        ) -> Self {
            Self::serve_responses(vec![MockResponse {
                status,
                headers,
                body,
            }])
        }

        pub(crate) fn serve_responses(responses: Vec<MockResponse>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock HTTP server");
            listener
                .set_nonblocking(true)
                .expect("configure mock HTTP server");
            let addr = listener.local_addr().expect("read mock HTTP server addr");
            let (shutdown_tx, shutdown_rx) = mpsc::channel();
            let handle = thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut responses: VecDeque<MockResponse> = responses.into();
                loop {
                    if shutdown_rx.try_recv().is_ok() {
                        return Ok(());
                    }
                    if responses.is_empty() {
                        return Ok(());
                    }
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let response = responses
                                .pop_front()
                                .expect("checked non-empty response queue");
                            read_request_headers(&mut stream)?;
                            write_response(
                                &mut stream,
                                response.status,
                                response.headers,
                                &response.body,
                            )?;
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return Err(
                                    "mock HTTP server timed out waiting for request".to_string()
                                );
                            }
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(err) => return Err(format!("mock HTTP accept failed: {err}")),
                    }
                }
            });
            Self {
                addr,
                shutdown: Some(shutdown_tx),
                handle: Some(handle),
            }
        }

        pub(crate) fn url(&self, path: &str) -> String {
            format!("http://{}:{}{path}", self.addr.ip(), self.addr.port())
        }

        pub(crate) fn url_for_host(&self, host: &str, path: &str) -> String {
            format!("http://{}:{}{path}", host, self.addr.port())
        }

        pub(crate) fn allow_origin_validator(&self) -> FetchUrlValidator {
            self.allow_host_validator("127.0.0.1")
        }

        pub(crate) fn allow_host_validator(&self, allowed_host: &'static str) -> FetchUrlValidator {
            let allowed_port = self.addr.port();
            let allowed_addr = self.addr;
            Arc::new(move |url| {
                let is_mock_origin = url.scheme() == "http"
                    && url.host_str() == Some(allowed_host)
                    && url.port_or_known_default() == Some(allowed_port);
                if is_mock_origin {
                    Ok(vec![allowed_addr])
                } else {
                    validate_remote_fetch_url(url)
                }
            })
        }
    }

    impl Drop for MockHttpServer {
        fn drop(&mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    impl ProxyTrap {
        pub(crate) fn serve() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind proxy trap");
            listener
                .set_nonblocking(true)
                .expect("configure proxy trap");
            let addr = listener.local_addr().expect("read proxy trap addr");
            let (shutdown_tx, shutdown_rx) = mpsc::channel();
            let saw_connection = Arc::new(AtomicBool::new(false));
            let saw_connection_for_thread = Arc::clone(&saw_connection);
            let handle = thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    if shutdown_rx.try_recv().is_ok() {
                        return Ok(());
                    }
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            saw_connection_for_thread.store(true, Ordering::SeqCst);
                            let _ = read_request_headers(&mut stream);
                            let _ = write_response(
                                &mut stream,
                                "502 Bad Gateway",
                                &[("Content-Type", "text/plain")],
                                b"proxy trap",
                            );
                            return Ok(());
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return Ok(());
                            }
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(err) => return Err(format!("proxy trap accept failed: {err}")),
                    }
                }
            });
            Self {
                addr,
                saw_connection,
                shutdown: Some(shutdown_tx),
                handle: Some(handle),
            }
        }

        pub(crate) fn url(&self) -> String {
            format!("http://{}:{}", self.addr.ip(), self.addr.port())
        }

        pub(crate) fn saw_connection(&self) -> bool {
            self.saw_connection.load(Ordering::SeqCst)
        }
    }

    impl Drop for ProxyTrap {
        fn drop(&mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn read_request_headers(stream: &mut TcpStream) -> Result<(), String> {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .map_err(|err| format!("configure mock HTTP read timeout: {err}"))?;
        let mut request = Vec::with_capacity(1024);
        let mut buf = [0u8; 512];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream
                .read(&mut buf)
                .map_err(|err| format!("read mock HTTP request: {err}"))?;
            if read == 0 {
                return Err("mock HTTP client closed before headers".to_string());
            }
            request.extend_from_slice(&buf[..read]);
            if request.len() > 8192 {
                return Err("mock HTTP request headers exceeded cap".to_string());
            }
        }
        Ok(())
    }

    fn write_response(
        stream: &mut TcpStream,
        status: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<(), String> {
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        stream
            .write_all(response.as_bytes())
            .and_then(|_| stream.write_all(body))
            .map_err(|err| format!("write mock HTTP response: {err}"))
    }
}
