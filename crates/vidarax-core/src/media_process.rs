//! Own one bounded media-tool invocation through child exit and pipe cleanup.
//! Fetch, decode and startup probes share this process-lifetime policy.

use std::process::Command;

pub(crate) const MEDIA_OUTPUT_MAX_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const MEDIA_STDERR_MAX_BYTES: u64 = 1024 * 1024;
pub(crate) const MEDIA_PROCESS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Bound media-tool pipes while draining stderr concurrently to avoid deadlock.
pub(crate) trait BoundedMediaOutput {
    fn bounded_media_output(&mut self) -> std::io::Result<std::process::Output>;
    fn bounded_media_output_with_limit(
        &mut self,
        limit: u64,
    ) -> std::io::Result<std::process::Output>;
}
impl BoundedMediaOutput for Command {
    fn bounded_media_output(&mut self) -> std::io::Result<std::process::Output> {
        bounded_media_output(self, MEDIA_OUTPUT_MAX_BYTES, MEDIA_PROCESS_TIMEOUT)
    }

    fn bounded_media_output_with_limit(
        &mut self,
        limit: u64,
    ) -> std::io::Result<std::process::Output> {
        bounded_media_output(self, limit, MEDIA_PROCESS_TIMEOUT)
    }
}

// Recorded decode policy: bound output and child lifetime, including after the
// requesting task is dropped. Pipe readers drain while the supervisor polls.
pub(crate) fn bounded_media_output(
    command: &mut Command,
    limit: u64,
    timeout: std::time::Duration,
) -> std::io::Result<std::process::Output> {
    bounded_media_output_with_input(command, limit, timeout, None)
}

pub(crate) fn bounded_media_output_with_input(
    command: &mut Command,
    limit: u64,
    timeout: std::time::Duration,
    input: Option<Vec<u8>>,
) -> std::io::Result<std::process::Output> {
    use std::io::Read;
    use std::process::Stdio;
    let started = std::time::Instant::now();
    let mut child = command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout_pipe = child.stdout.take().expect("piped stdout");
    let mut stderr_pipe = child.stderr.take().expect("piped stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    let stdout_task = std::thread::Builder::new()
        .name("vx-media-stdout".into())
        .spawn(move || {
            let mut stdout = Vec::new();
            let result = stdout_pipe
                .take(limit.saturating_add(1))
                .read_to_end(&mut stdout)
                .map(|_| stdout);
            let _ = tx.send(result);
        });
    let stdout_task = match stdout_task {
        Ok(task) => task,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let stderr_task = std::thread::Builder::new()
        .name("vx-media-stderr".into())
        .spawn(move || {
            let mut stderr = Vec::new();
            let _ = (&mut stderr_pipe)
                .take(MEDIA_STDERR_MAX_BYTES)
                .read_to_end(&mut stderr);
            let _ = std::io::copy(&mut stderr_pipe, &mut std::io::sink());
            stderr
        });
    let stderr_task = match stderr_task {
        Ok(task) => task,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_task.join();
            return Err(error);
        }
    };
    let (input_tx, input_rx) = std::sync::mpsc::channel();
    let input_task = if let Some(input) = input {
        let mut stdin = child.stdin.take().expect("piped stdin");
        let task = std::thread::Builder::new()
            .name("vx-media-stdin".into())
            .spawn(move || {
                use std::io::Write;
                let result = stdin.write_all(&input).and_then(|_| stdin.flush());
                let _ = input_tx.send(result);
            });
        match task {
            Ok(task) => Some(task),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_task.join();
                let _ = stderr_task.join();
                return Err(error);
            }
        }
    } else {
        None
    };
    let mut stdout = None;
    let mut failure = None;
    let status = loop {
        if let Ok(Err(error)) = input_rx.try_recv() {
            failure = Some(error);
        }
        if stdout.is_none() {
            if let Ok(result) = rx.try_recv() {
                if result
                    .as_ref()
                    .is_ok_and(|bytes| bytes.len() as u64 > limit)
                {
                    failure = Some(std::io::Error::other(
                        "media tool output exceeds byte budget",
                    ));
                } else if result.is_err() {
                    failure = Some(std::io::Error::other("failed to read media tool output"));
                }
                stdout = Some(result);
            }
        }
        if failure.is_none() && started.elapsed() >= timeout {
            failure = Some(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "media tool timed out",
            ));
        }
        if failure.is_some() {
            let _ = child.kill();
            break child.wait();
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(error) => {
                failure = Some(error);
                let _ = child.kill();
                break child.wait();
            }
        }
    };
    let _ = stdout_task.join();
    let stderr = stderr_task.join().unwrap_or_default();
    if let Some(task) = input_task {
        let _ = task.join();
        if failure.is_none() {
            if let Ok(Err(error)) = input_rx.try_recv() {
                failure = Some(error);
            }
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let stdout = stdout
        .or_else(|| rx.recv().ok())
        .ok_or_else(|| std::io::Error::other("media tool output reader failed"))??;
    if stdout.len() as u64 > limit {
        return Err(std::io::Error::other(
            "media tool output exceeds byte budget",
        ));
    }
    Ok(std::process::Output {
        status: status?,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn bounded_media_input_and_output_make_progress_concurrently() {
        let output = super::bounded_media_output_with_input(
            std::process::Command::new("python3").args(["-c", "import sys; sys.stdout.buffer.write(b'x'*1000000); sys.stdout.buffer.flush(); data=sys.stdin.buffer.read(); sys.stdout.buffer.write(str(len(data)).encode())"]),
            2 * 1024 * 1024,
            std::time::Duration::from_secs(3),
            Some(vec![0; 2 * 1024 * 1024]),
        ).unwrap();
        assert!(output.status.success());
        assert_eq!(&output.stdout[1_000_000..], b"2097152");
    }

    #[test]
    fn bounded_media_timeout_releases_blocked_input_writer() {
        let started = std::time::Instant::now();
        let error = super::bounded_media_output_with_input(
            std::process::Command::new("python3").args(["-c", "import time; time.sleep(30)"]),
            32,
            std::time::Duration::from_millis(100),
            Some(vec![0; 2 * 1024 * 1024]),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn bounded_media_input_failure_kills_child_and_releases_readers() {
        let started = std::time::Instant::now();
        let error = super::bounded_media_output_with_input(
            std::process::Command::new("python3")
                .args(["-c", "import os,time; os.close(0); time.sleep(30)"]),
            32,
            std::time::Duration::from_secs(3),
            Some(vec![0; 2 * 1024 * 1024]),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn bounded_decoder_times_out_and_reaps_child() {
        let pid_file = std::env::temp_dir().join(format!(
            "vidarax_media_process_pid_{}_{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let script = format!("echo $$ > '{}'; exec sleep 30", pid_file.display());
        let started = std::time::Instant::now();
        let error = super::bounded_media_output(
            std::process::Command::new("sh").args(["-c", &script]),
            32,
            std::time::Duration::from_millis(100),
        )
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
        let pid = std::fs::read_to_string(&pid_file).unwrap();
        let live = std::process::Command::new("kill")
            .args(["-0", pid.trim()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(!live.success(), "decoder child must be reaped");
        let _ = std::fs::remove_file(pid_file);
    }

    #[test]
    fn bounded_decoder_drains_saturated_stderr_and_rejects_overflow() {
        let output = super::bounded_media_output(
            std::process::Command::new("python3").args([
                "-c",
                "import sys; sys.stderr.write('x' * 2000000); sys.stdout.write('jpeg')",
            ]),
            32,
            std::time::Duration::from_secs(3),
        )
        .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"jpeg");
        assert_eq!(output.stderr.len(), 1024 * 1024);
        let error = super::bounded_media_output(
            std::process::Command::new("python3")
                .args(["-c", "import sys; sys.stdout.write('x' * 1000000)"]),
            32,
            std::time::Duration::from_secs(3),
        )
        .unwrap_err();
        assert!(error.to_string().contains("byte budget"));
    }
}
