use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
    Arc<dyn Fn(&reqwest::Url) -> Result<(), String> + Send + Sync + 'static>;

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
    let parsed = reqwest::Url::parse(url).map_err(|err| format!("invalid remote media URL: {err}"))?;
    validate_url(&parsed)?;
    let started = Instant::now();

    let client = Client::builder()
        .redirect(redirect::Policy::none())
        .build()
        .map_err(|err| format!("failed to build remote media fetch client: {err}"))?;

    let mut response = fetch_remote_media_response(&client, parsed, validate_url, started)?;
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
            let _ = fs::remove_file(&path);
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
        let _ = fs::remove_file(&path);
        return Err("remote media response was empty".to_string());
    }
    if has_extm3u_magic(&first_bytes) {
        let _ = fs::remove_file(&path);
        return Err("remote media must be a media container, not a playlist manifest".to_string());
    }
    if let Err(err) = validate_prefetched_media_container(&path) {
        let _ = fs::remove_file(&path);
        return Err(err);
    }

    Ok(PrefetchedMedia { path })
}

fn fetch_remote_media_response(
    client: &Client,
    mut url: reqwest::Url,
    validate_url: FetchUrlValidator,
    started: Instant,
) -> Result<reqwest::blocking::Response, String> {
    for redirect_count in 0..=REMOTE_MEDIA_PREFETCH_MAX_REDIRECTS {
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
        validate_redirect_target(&url, &next_url, &validate_url)?;
        url = next_url;
    }

    Err("too many redirects".to_string())
}

fn validate_redirect_target(
    current_url: &reqwest::Url,
    next_url: &reqwest::Url,
    validate_url: &FetchUrlValidator,
) -> Result<(), String> {
    if !matches!(next_url.scheme(), "http" | "https") {
        return Err("redirect target rejected: unsupported scheme".to_string());
    }
    if current_url.scheme() == "https" && next_url.scheme() == "http" {
        return Err("redirect target rejected: https->http downgrade".to_string());
    }
    if let Err(err) = validate_url(next_url) {
        if err == "insecure http scheme" {
            return Err("redirect target rejected: insecure http scheme".to_string());
        }
        let host = next_url.host_str().unwrap_or("<missing host>");
        return Err(format!("redirect target rejected: {host} ({err})"));
    }
    Ok(())
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
        has_extm3u_magic, prefetch_remote_media_with_limit_and_validator_for_test,
        validate_prefetched_media_container_with_probe_for_test, validate_redirect_target,
        REMOTE_MEDIA_PREFETCH_MAX_BYTES,
    };
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        let _guard = ENV_LOCK
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
        let allow_all: super::FetchUrlValidator = Arc::new(|_: &reqwest::Url| Ok(()));

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
    fn prefetched_media_probe_times_out_and_kills_slow_child() {
        let path = std::env::temp_dir().join(format!(
            "vidarax-prefetch-timeout-{}",
            std::process::id()
        ));
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
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::{FetchUrlValidator, validate_remote_fetch_url};
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::Arc;
    use std::sync::mpsc::{self, Sender};
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
                                return Err("mock HTTP server timed out waiting for request".to_string());
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

        pub(crate) fn allow_origin_validator(&self) -> FetchUrlValidator {
            let allowed_port = self.addr.port();
            Arc::new(move |url| {
                let is_mock_origin = url.scheme() == "http"
                    && url.host_str() == Some("127.0.0.1")
                    && url.port_or_known_default() == Some(allowed_port);
                if is_mock_origin {
                    Ok(())
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
