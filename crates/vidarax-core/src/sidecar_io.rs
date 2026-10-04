use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

pub(crate) fn exchange_deadline(timeout: Duration) -> io::Result<Instant> {
    Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "sidecar timeout is too large"))
}

pub(crate) fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "sidecar exchange timed out"))
}

/// Borrow one connection for an exchange with a single end-to-end deadline.
/// Each partial read or write uses the remaining time, so peer progress cannot
/// restart the request timeout.
pub(crate) struct DeadlineStream<'a> {
    stream: &'a mut TcpStream,
    deadline: Instant,
}

impl<'a> DeadlineStream<'a> {
    pub(crate) fn new(stream: &'a mut TcpStream, deadline: Instant) -> Self {
        Self { stream, deadline }
    }
}

impl Read for DeadlineStream<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.stream
            .set_read_timeout(Some(remaining(self.deadline)?))?;
        self.stream.read(bytes)
    }
}

impl Write for DeadlineStream<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.stream
            .set_write_timeout(Some(remaining(self.deadline)?))?;
        self.stream.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        remaining(self.deadline)?;
        self.stream.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::remaining;
    use std::io::ErrorKind;
    use std::time::Instant;

    #[test]
    fn expired_exchange_returns_timeout_instead_of_zero_socket_timeout() {
        assert_eq!(
            remaining(Instant::now()).unwrap_err().kind(),
            ErrorKind::TimedOut
        );
    }
}
