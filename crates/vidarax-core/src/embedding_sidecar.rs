//! Binary client for the image-embedding sidecar.

use std::fmt::{Display, Formatter};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::sidecar_io::{exchange_deadline, remaining, DeadlineStream};

pub const EMBEDDING_DIM: usize = 768;
pub const MAX_JPEG_BYTES: usize = 10 * 1024 * 1024;

const PROTOCOL_VERSION: u8 = 1;
const REQUEST_MAGIC: [u8; 4] = *b"VXEM";
const RESPONSE_MAGIC: [u8; 4] = *b"VXER";
const HEADER_BYTES: usize = 12;
const EMBEDDING_BYTES: usize = EMBEDDING_DIM * std::mem::size_of::<f32>();
const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 5_000;

#[derive(Debug)]
pub enum EmbeddingSidecarError {
    InvalidAddress(String),
    ImageTooLarge(usize),
    BackingOff,
    Io(std::io::Error),
    Protocol(&'static str),
    SidecarStatus(u8),
}

impl Display for EmbeddingSidecarError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAddress(address) => {
                write!(
                    f,
                    "embedding sidecar address has no TCP endpoint: {address}"
                )
            }
            Self::ImageTooLarge(bytes) => {
                write!(
                    f,
                    "embedding JPEG is {bytes} bytes; maximum is {MAX_JPEG_BYTES}"
                )
            }
            Self::BackingOff => f.write_str("embedding sidecar reconnect backoff is active"),
            Self::Io(err) => write!(f, "embedding sidecar I/O: {err}"),
            Self::Protocol(message) => write!(f, "embedding sidecar protocol: {message}"),
            Self::SidecarStatus(status) => {
                write!(f, "embedding sidecar returned status {status}")
            }
        }
    }
}

impl std::error::Error for EmbeddingSidecarError {}

impl From<std::io::Error> for EmbeddingSidecarError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Persistent connection with one deadline per exchange and reconnect backoff.
/// A failed exchange closes the connection. Callers fail open on errors.
pub struct EmbeddingSidecarClient {
    address: SocketAddr,
    timeout: Duration,
    stream: Option<TcpStream>,
    consecutive_failures: u32,
    retry_after: Option<Instant>,
}

impl EmbeddingSidecarClient {
    pub fn new(address: &str, timeout_ms: u64) -> Result<Self, EmbeddingSidecarError> {
        let address = address.strip_prefix("tcp://").unwrap_or(address);
        let socket = address
            .to_socket_addrs()
            .map_err(EmbeddingSidecarError::Io)?
            .next()
            .ok_or_else(|| EmbeddingSidecarError::InvalidAddress(address.to_string()))?;
        Ok(Self {
            address: socket,
            timeout: Duration::from_millis(timeout_ms.max(1)),
            stream: None,
            consecutive_failures: 0,
            retry_after: None,
        })
    }

    pub fn embed(&mut self, jpeg: &[u8]) -> Result<[f32; EMBEDDING_DIM], EmbeddingSidecarError> {
        if jpeg.len() > MAX_JPEG_BYTES {
            return Err(EmbeddingSidecarError::ImageTooLarge(jpeg.len()));
        }
        if self
            .retry_after
            .is_some_and(|deadline| Instant::now() < deadline)
        {
            return Err(EmbeddingSidecarError::BackingOff);
        }

        match self.exchange(jpeg) {
            Ok(embedding) => {
                self.consecutive_failures = 0;
                self.retry_after = None;
                Ok(embedding)
            }
            Err(err) => {
                self.stream = None;
                self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                let shift = self.consecutive_failures.saturating_sub(1).min(6);
                let backoff_ms = INITIAL_BACKOFF_MS
                    .saturating_mul(1_u64 << shift)
                    .min(MAX_BACKOFF_MS);
                self.retry_after = Some(Instant::now() + Duration::from_millis(backoff_ms));
                Err(err)
            }
        }
    }

    fn exchange(&mut self, jpeg: &[u8]) -> Result<[f32; EMBEDDING_DIM], EmbeddingSidecarError> {
        let deadline = exchange_deadline(self.timeout)?;
        let mut stream = DeadlineStream::new(self.connection(deadline)?, deadline);
        let jpeg_len = u32::try_from(jpeg.len())
            .map_err(|_| EmbeddingSidecarError::ImageTooLarge(jpeg.len()))?;
        let mut request_header = [0_u8; HEADER_BYTES];
        request_header[..4].copy_from_slice(&REQUEST_MAGIC);
        request_header[4] = PROTOCOL_VERSION;
        request_header[8..12].copy_from_slice(&jpeg_len.to_be_bytes());
        stream.write_all(&request_header)?;
        stream.write_all(jpeg)?;

        let mut response_header = [0_u8; HEADER_BYTES];
        stream.read_exact(&mut response_header)?;
        if response_header[..4] != RESPONSE_MAGIC {
            return Err(EmbeddingSidecarError::Protocol("bad response magic"));
        }
        if response_header[4] != PROTOCOL_VERSION {
            return Err(EmbeddingSidecarError::Protocol("unsupported version"));
        }
        let status = response_header[5];
        if status != 0 {
            return Err(EmbeddingSidecarError::SidecarStatus(status));
        }
        let dim = u16::from_be_bytes([response_header[6], response_header[7]]) as usize;
        if dim != EMBEDDING_DIM {
            return Err(EmbeddingSidecarError::Protocol(
                "unexpected embedding width",
            ));
        }
        let payload_len = u32::from_be_bytes(response_header[8..12].try_into().expect("4 bytes"));
        if payload_len as usize != EMBEDDING_BYTES {
            return Err(EmbeddingSidecarError::Protocol("unexpected payload length"));
        }

        let mut response_bytes = [0_u8; EMBEDDING_BYTES];
        stream.read_exact(&mut response_bytes)?;
        let mut embedding = [0_f32; EMBEDDING_DIM];
        for (value, bytes) in embedding
            .iter_mut()
            .zip(response_bytes.chunks_exact(std::mem::size_of::<f32>()))
        {
            *value = f32::from_le_bytes(bytes.try_into().expect("4-byte chunk"));
            if !value.is_finite() {
                return Err(EmbeddingSidecarError::Protocol(
                    "embedding contains a non-finite value",
                ));
            }
        }
        Ok(embedding)
    }

    fn connection(&mut self, deadline: Instant) -> Result<&mut TcpStream, EmbeddingSidecarError> {
        if self.stream.is_none() {
            let stream = TcpStream::connect_timeout(&self.address, remaining(deadline)?)?;
            stream.set_nodelay(true)?;
            self.stream = Some(stream);
        }
        self.stream
            .as_mut()
            .ok_or(EmbeddingSidecarError::Protocol("missing connection"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn trickled_response_cannot_restart_exchange_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture");
        let address = listener.local_addr().expect("fixture address");
        let peer = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fixture");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut header = [0; HEADER_BYTES];
            stream.read_exact(&mut header).expect("request header");
            let mut jpeg = vec![0; u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize];
            stream.read_exact(&mut jpeg).expect("request JPEG");
            let mut response = [0; HEADER_BYTES];
            response[..4].copy_from_slice(&RESPONSE_MAGIC);
            response[4] = PROTOCOL_VERSION;
            response[6..8].copy_from_slice(&(EMBEDDING_DIM as u16).to_be_bytes());
            response[8..12].copy_from_slice(&(EMBEDDING_BYTES as u32).to_be_bytes());
            for byte in response {
                thread::sleep(Duration::from_millis(30));
                if stream.write_all(&[byte]).is_err() {
                    return;
                }
            }
            let _ = stream.write_all(&[0; EMBEDDING_BYTES]);
        });

        let mut client = EmbeddingSidecarClient::new(&address.to_string(), 80).unwrap();
        let result = client.embed(b"jpeg");
        peer.join().unwrap();
        assert!(matches!(result, Err(EmbeddingSidecarError::Io(_))));
        assert!(client.stream.is_none());
        assert!(matches!(
            client.embed(b"jpeg"),
            Err(EmbeddingSidecarError::BackingOff)
        ));
    }
}
