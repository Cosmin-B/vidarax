use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Apply restrictive file permissions (owner read/write only) on Unix (C-4).
#[cfg(unix)]
fn apply_restrictive_permissions(opts: &mut OpenOptions) -> &mut OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;
    opts.mode(0o600)
}

/// No-op on non-Unix platforms.
#[cfg(not(unix))]
fn apply_restrictive_permissions(opts: &mut OpenOptions) -> &mut OpenOptions {
    opts
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEvent {
    pub seq: u64,
    pub run_id: String,
    pub stream_id: String,
    pub pts_ms: u64,
    pub kind: String,
    pub payload: String,
}

impl TimelineEvent {
    fn encode_line(&self) -> String {
        // Pre-allocate enough for the numeric fields plus the string fields
        // (worst-case escape doubles every char, so reserve full field lengths).
        let cap = 40
            + self.run_id.len()
            + self.stream_id.len()
            + self.kind.len()
            + self.payload.len();
        let mut buf = String::with_capacity(cap);
        use std::fmt::Write as _;
        let _ = write!(
            buf,
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.seq,
            sanitize(&self.run_id),
            sanitize(&self.stream_id),
            self.pts_ms,
            sanitize(&self.kind),
            sanitize(&self.payload)
        );
        buf
    }

    fn decode_line(line: &str) -> Option<Self> {
        let mut parts = line.splitn(6, '\t');
        let seq = parts.next()?.parse().ok()?;
        let run_id = restore(parts.next()?);
        let stream_id = restore(parts.next()?);
        let pts_ms = parts.next()?.parse().ok()?;
        let kind = restore(parts.next()?);
        let payload = restore(parts.next()?);
        Some(Self {
            seq,
            run_id,
            stream_id,
            pts_ms,
            kind,
            payload,
        })
    }
}

#[derive(Debug)]
pub enum TimelineError {
    Io(std::io::Error),
    Index(String),
}

impl Display for TimelineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TimelineError::Io(err) => write!(f, "{err}"),
            TimelineError::Index(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for TimelineError {}

impl From<std::io::Error> for TimelineError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub struct WalWriter {
    path: PathBuf,
    file: File,
}

impl WalWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TimelineError> {
        let path = path.as_ref().to_path_buf();
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        apply_restrictive_permissions(&mut opts);
        let file = opts.open(&path)?;
        Ok(Self { path, file })
    }

    pub fn append(&mut self, event: &TimelineEvent) -> Result<(), TimelineError> {
        writeln!(self.file, "{}", event.encode_line())?;
        self.file.flush()?;
        Ok(())
    }

    pub fn read_all(&self) -> Result<Vec<TimelineEvent>, TimelineError> {
        read_all_events(&self.path)
    }
}

pub fn append_event(path: impl AsRef<Path>, event: &TimelineEvent) -> Result<(), TimelineError> {
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    apply_restrictive_permissions(&mut opts);
    let mut file = opts.open(path.as_ref())?;
    writeln!(file, "{}", event.encode_line())?;
    file.flush()?;
    Ok(())
}

pub fn read_all_events(path: impl AsRef<Path>) -> Result<Vec<TimelineEvent>, TimelineError> {
    let path = path.as_ref();
    let file = match OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(TimelineError::Io(err)),
    };
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Some(event) = TimelineEvent::decode_line(&line) {
            out.push(event);
        }
    }
    Ok(out)
}

pub trait EventIndex {
    fn append(&mut self, event: &TimelineEvent) -> Result<(), String>;
    fn has_sequence(&self, seq: u64) -> bool;
}

pub struct DualWriter<I: EventIndex> {
    wal: WalWriter,
    index: I,
}

impl<I: EventIndex> DualWriter<I> {
    pub fn new(wal: WalWriter, index: I) -> Self {
        Self { wal, index }
    }

    pub fn append(&mut self, event: &TimelineEvent) -> Result<(), TimelineError> {
        // WAL first: source of truth.
        self.wal.append(event)?;
        self.index.append(event).map_err(TimelineError::Index)?;
        Ok(())
    }

    pub fn reconcile_missing(&mut self) -> Result<usize, TimelineError> {
        let events = self.wal.read_all()?;
        let mut repaired = 0usize;
        for event in events {
            if !self.index.has_sequence(event.seq) {
                self.index.append(&event).map_err(TimelineError::Index)?;
                repaired += 1;
            }
        }
        Ok(repaired)
    }
}

#[inline]
fn sanitize(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

#[inline]
fn restore(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        append_event, read_all_events, DualWriter, EventIndex, TimelineError, TimelineEvent,
        WalWriter,
    };
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Default)]
    struct InMemoryIndex {
        seqs: HashSet<u64>,
        fail_once: bool,
    }

    impl EventIndex for InMemoryIndex {
        fn append(&mut self, event: &TimelineEvent) -> Result<(), String> {
            if self.fail_once {
                self.fail_once = false;
                return Err("transient index failure".to_string());
            }
            self.seqs.insert(event.seq);
            Ok(())
        }

        fn has_sequence(&self, seq: u64) -> bool {
            self.seqs.contains(&seq)
        }
    }

    fn test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("vidarax-{name}-{nanos}.wal"))
    }

    fn event(seq: u64) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: "run-1".to_string(),
            stream_id: "stream-1".to_string(),
            pts_ms: seq * 10,
            kind: "keepframe".to_string(),
            payload: "{}".to_string(),
        }
    }

    #[test]
    fn wal_and_index_append_success() {
        let path = test_path("ok");
        let wal = WalWriter::open(&path).unwrap();
        let index = InMemoryIndex::default();
        let mut dual = DualWriter::new(wal, index);

        dual.append(&event(1)).unwrap();
        let repaired = dual.reconcile_missing().unwrap();
        assert_eq!(repaired, 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wal_persists_even_if_index_fails() {
        let path = test_path("fail");
        let wal = WalWriter::open(&path).unwrap();
        let index = InMemoryIndex {
            seqs: HashSet::new(),
            fail_once: true,
        };
        let mut dual = DualWriter::new(wal, index);

        let err = dual.append(&event(1)).unwrap_err();
        assert!(matches!(err, TimelineError::Index(_)));

        let repaired = dual.reconcile_missing().unwrap();
        assert_eq!(repaired, 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn append_and_read_helpers_roundtrip() {
        let path = test_path("helpers");
        let event = event(42);
        append_event(&path, &event).unwrap();
        let events = read_all_events(&path).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], event);
        let _ = std::fs::remove_file(path);
    }
}
