use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Flush file data and metadata before reporting a WAL or blob write as complete.
/// On macOS, use `F_FULLFSYNC` because `fsync` can return while data is still
/// in the drive cache.
#[cfg_attr(target_os = "macos", allow(unsafe_code))]
pub fn sync_file_durable(file: &File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;

        // SAFETY: `file` owns a valid descriptor for the duration of this call.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
    }
}

/// Apply restrictive file permissions (owner read/write only) on Unix (C-4).
#[cfg(unix)]
fn apply_restrictive_permissions(opts: &mut OpenOptions) -> &mut OpenOptions {
    use std::os::unix::fs::OpenOptionsExt;
    opts.mode(0o600).custom_flags(libc::O_NOFOLLOW)
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
    /// Exact bytes on disk for this V2 record, including framing. Returns an
    /// error when the escaped body exceeds the WAL record limit.
    pub fn framed_len(&self) -> Result<usize, TimelineError> {
        framed_len_fields(
            self.seq,
            &self.run_id,
            &self.stream_id,
            self.pts_ms,
            &self.kind,
            &self.payload,
        )
    }

    fn encode_body_into(&self, out: &mut Vec<u8>) {
        write!(out, "{}\t", self.seq).expect("Vec writes cannot fail");
        sanitize_into(&self.run_id, out);
        out.push(b'\t');
        sanitize_into(&self.stream_id, out);
        write!(out, "\t{}\t", self.pts_ms).expect("Vec writes cannot fail");
        sanitize_into(&self.kind, out);
        out.push(b'\t');
        sanitize_into(&self.payload, out);
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
    Corrupt { offset: u64, reason: String },
}

impl Display for TimelineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TimelineError::Io(err) => write!(f, "{err}"),
            TimelineError::Index(err) => write!(f, "{err}"),
            TimelineError::Corrupt { offset, reason } => {
                write!(f, "timeline WAL corrupt at byte {offset}: {reason}")
            }
        }
    }
}

impl std::error::Error for TimelineError {}

impl From<std::io::Error> for TimelineError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

struct LockedFile(File);

impl LockedFile {
    fn acquire(file: File) -> Result<Self, std::fs::TryLockError> {
        file.try_lock()?;
        Ok(Self(file))
    }
}

impl std::ops::Deref for LockedFile {
    type Target = File;

    fn deref(&self) -> &File {
        &self.0
    }
}

impl std::ops::DerefMut for LockedFile {
    fn deref_mut(&mut self) -> &mut File {
        &mut self.0
    }
}

impl Drop for LockedFile {
    fn drop(&mut self) {
        // Closing our descriptor can leave the lock held by a child between
        // fork and exec. Release it when this owner finishes, including when
        // opening the WAL fails after we acquired one of its locks.
        let _ = self.0.unlock();
    }
}

pub struct WalWriter {
    path: PathBuf,
    file: LockedFile,
    _ownership: LockedFile,
    committed_end: u64,
    pending_end: u64,
    poisoned: bool,
    #[cfg(test)]
    fail_sync_once: bool,
}

impl WalWriter {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TimelineError> {
        let supplied = path.as_ref();
        let file_name = supplied.file_name().ok_or_else(|| {
            TimelineError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "WAL path has no file name",
            ))
        })?;
        let parent = supplied
            .parent()
            .filter(|directory| !directory.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let path = std::fs::canonicalize(parent)?.join(file_name);
        let lock_path = path.with_file_name(format!(
            "{}.lock",
            path.file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    TimelineError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "WAL path has no file name",
                    ))
                })?
        ));
        reject_symlink(&path)?;
        reject_symlink(&lock_path)?;
        let mut lock_options = OpenOptions::new();
        lock_options.create(true).write(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            lock_options.custom_flags(libc::O_NOFOLLOW);
        }
        let ownership = LockedFile::acquire(lock_options.open(lock_path)?).map_err(|error| {
            TimelineError::Io(std::io::Error::other(format!(
                "timeline WAL already owned or cannot be locked: {error}"
            )))
        })?;

        let is_new = !path.exists();
        let mut opts = OpenOptions::new();
        opts.create(true).read(true).append(true);
        apply_restrictive_permissions(&mut opts);
        // The sidecar lock prevents two writers from creating the same WAL.
        // Lock the WAL file too, so opening it through a hard link cannot bypass
        // that lock. Both locks stay held until this writer is dropped.
        let file = LockedFile::acquire(opts.open(&path)?).map_err(|error| {
            TimelineError::Io(std::io::Error::other(format!(
                "timeline WAL inode already owned or cannot be locked: {error}"
            )))
        })?;
        if is_new {
            sync_file_durable(&file)?;
        }
        // Config may have just created the WAL's parent directories. Sync each
        // directory so their names survive a power loss.
        sync_parent_directory(&path)?;

        let scan = scan_events(file.try_clone()?, None, false)?;
        if scan.incomplete_tail {
            file.set_len(scan.last_complete_end)?;
        }
        // A previous writer may have written a complete record before its sync
        // failed. Sync the recovered file before replay exposes those records.
        sync_file_durable(&file)?;
        let committed_end = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            _ownership: ownership,
            committed_end,
            pending_end: committed_end,
            poisoned: false,
            #[cfg(test)]
            fail_sync_once: false,
        })
    }

    /// Append a checksummed record and wait for the filesystem to sync it.
    /// A write or sync error may leave some or all of the record on disk.
    /// Further appends fail until the caller reopens the WAL and runs recovery.
    pub fn append(&mut self, event: &TimelineEvent) -> Result<(), TimelineError> {
        self.stage(event)?;
        self.sync_pending()
    }

    /// Write one complete record without acknowledging it. Call `sync_pending`
    /// before making the record visible to readers or replying to its caller.
    /// Returns the number of bytes written, including the header and final newline.
    pub fn stage(&mut self, event: &TimelineEvent) -> Result<usize, TimelineError> {
        if self.poisoned {
            return Err(TimelineError::Io(std::io::Error::other(
                "timeline WAL writer is poisoned; reopen for recovery",
            )));
        }
        let body_len = encoded_len_fields(
            event.seq,
            &event.run_id,
            &event.stream_id,
            event.pts_ms,
            &event.kind,
            &event.payload,
        )?;
        let framed_len = framed_len_for_body(body_len);
        let mut line = Vec::with_capacity(framed_len);
        line.extend_from_slice(b"V2\t");
        write!(&mut line, "{body_len}\t").expect("Vec writes cannot fail");
        let checksum_start = line.len();
        line.extend_from_slice(b"00000000\t");
        let body_start = line.len();
        event.encode_body_into(&mut line);
        debug_assert_eq!(line.len() - body_start, body_len);
        let checksum = crc32fast::hash(&line[body_start..]);
        for (index, digit) in line[checksum_start..checksum_start + 8]
            .iter_mut()
            .enumerate()
        {
            let nibble = ((checksum >> (4 * (7 - index))) & 0xf) as u8;
            *digit = if nibble < 10 {
                b'0' + nibble
            } else {
                b'a' + nibble - 10
            };
        }
        line.push(b'\n');
        debug_assert_eq!(line.len(), framed_len);
        let next_end = self
            .pending_end
            .checked_add(line.len() as u64)
            .ok_or_else(|| {
                TimelineError::Io(std::io::Error::other("timeline WAL byte offset overflow"))
            })?;
        self.poisoned = true;
        self.file.write_all(&line)?;
        self.pending_end = next_end;
        self.poisoned = false;
        Ok(line.len())
    }

    /// Sync all staged records in one call. If it fails, none is acknowledged
    /// and further appends fail until the WAL is reopened for recovery.
    pub fn sync_pending(&mut self) -> Result<(), TimelineError> {
        if self.poisoned {
            return Err(TimelineError::Io(std::io::Error::other(
                "timeline WAL writer is poisoned; reopen for recovery",
            )));
        }
        if self.pending_end == self.committed_end {
            return Ok(());
        }
        self.poisoned = true;
        #[cfg(test)]
        if std::mem::take(&mut self.fail_sync_once) {
            return Err(TimelineError::Io(std::io::Error::other(
                "injected WAL synchronization failure after write",
            )));
        }
        sync_file_durable(&self.file)?;
        self.committed_end = self.pending_end;
        self.poisoned = false;
        Ok(())
    }

    pub fn committed_end(&self) -> u64 {
        self.committed_end
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[cfg(test)]
    fn fail_sync_once_for_tests(&mut self) {
        self.fail_sync_once = true;
    }

    pub fn read_all(&self) -> Result<Vec<TimelineEvent>, TimelineError> {
        read_all_events_up_to(&self.path, Some(self.committed_end))
    }
}

fn reject_symlink(path: &Path) -> Result<(), TimelineError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(TimelineError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("timeline path {} must not be a symlink", path.display()),
            )))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(TimelineError::Io(error)),
    }
}

pub fn append_event(path: impl AsRef<Path>, event: &TimelineEvent) -> Result<(), TimelineError> {
    WalWriter::open(path)?.append(event)
}

pub fn read_all_events(path: impl AsRef<Path>) -> Result<Vec<TimelineEvent>, TimelineError> {
    read_all_events_up_to(path, None)
}

pub fn read_all_events_up_to(
    path: impl AsRef<Path>,
    committed_end: Option<u64>,
) -> Result<Vec<TimelineEvent>, TimelineError> {
    let file = match OpenOptions::new().read(true).open(path.as_ref()) {
        Ok(file) => file,
        Err(err)
            if err.kind() == std::io::ErrorKind::NotFound && committed_end.unwrap_or(0) == 0 =>
        {
            return Ok(Vec::new());
        }
        Err(err) => return Err(TimelineError::Io(err)),
    };
    Ok(scan_events(file, committed_end, true)?.events)
}

/// Maximum escaped body size of one V2 WAL event. Framing adds at most 22 bytes.
pub const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;

fn oversize_record() -> TimelineError {
    TimelineError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "timeline event exceeds maximum record size",
    ))
}

fn checked_body_add(current: usize, additional: usize) -> Result<usize, TimelineError> {
    let next = current
        .checked_add(additional)
        .ok_or_else(oversize_record)?;
    if next > MAX_RECORD_BYTES {
        return Err(oversize_record());
    }
    Ok(next)
}

fn checked_escaped_add(current: usize, field: &str) -> Result<usize, TimelineError> {
    let with_raw = checked_body_add(current, field.len())?;
    let escapes = memchr::memchr3_iter(b'\\', b'\t', b'\n', field.as_bytes()).count();
    checked_body_add(with_raw, escapes)
}

fn decimal_len(value: u64) -> usize {
    if value == 0 {
        1
    } else {
        value.ilog10() as usize + 1
    }
}

fn encoded_len_fields(
    seq: u64,
    run_id: &str,
    stream_id: &str,
    pts_ms: u64,
    kind: &str,
    payload: &str,
) -> Result<usize, TimelineError> {
    let mut len = checked_body_add(decimal_len(seq), 5)?;
    len = checked_escaped_add(len, run_id)?;
    len = checked_escaped_add(len, stream_id)?;
    len = checked_body_add(len, decimal_len(pts_ms))?;
    len = checked_escaped_add(len, kind)?;
    checked_escaped_add(len, payload)
}

fn framed_len_for_body(body_len: usize) -> usize {
    body_len + 14 + decimal_len(body_len as u64)
}

/// Return the encoded V2 record size without taking ownership of the fields.
/// Call this before copying a payload into the writer queue.
pub fn framed_len_fields(
    seq: u64,
    run_id: &str,
    stream_id: &str,
    pts_ms: u64,
    kind: &str,
    payload: &str,
) -> Result<usize, TimelineError> {
    Ok(framed_len_for_body(encoded_len_fields(
        seq, run_id, stream_id, pts_ms, kind, payload,
    )?))
}

struct ScanResult {
    events: Vec<TimelineEvent>,
    last_complete_end: u64,
    incomplete_tail: bool,
}

fn scan_events(
    file: File,
    committed_end: Option<u64>,
    collect_events: bool,
) -> Result<ScanResult, TimelineError> {
    let mut reader = BufReader::new(file.take(committed_end.unwrap_or(u64::MAX)));
    let mut out = Vec::new();
    let mut offset = 0u64;
    let mut last_complete_end = 0u64;
    let mut incomplete_tail = false;
    loop {
        let (bytes, terminated) = read_bounded_line(&mut reader, offset)?;
        if bytes.is_empty() {
            break;
        }
        let line_start = offset;
        offset += bytes.len() as u64;
        // Both writers (legacy and V2) write a newline only after the complete
        // record body. A plausible-looking legacy payload without it is still
        // a torn append, and a V2 header may be cut after just one byte.
        if !terminated {
            incomplete_tail = true;
            break;
        }
        let event = decode_record(&bytes[..bytes.len() - 1], line_start)?;
        last_complete_end = offset;
        if collect_events {
            out.push(event);
        }
    }
    if let Some(expected_end) = committed_end {
        if incomplete_tail || offset != expected_end {
            return Err(TimelineError::Corrupt {
                offset,
                reason: format!(
                    "published byte boundary {expected_end} does not end at a complete record"
                ),
            });
        }
    }
    Ok(ScanResult {
        events: out,
        last_complete_end,
        incomplete_tail,
    })
}

fn read_bounded_line(
    reader: &mut impl BufRead,
    offset: u64,
) -> Result<(Vec<u8>, bool), TimelineError> {
    let mut line = Vec::new();
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok((line, false));
        }
        let take = chunk
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(chunk.len(), |pos| pos + 1);
        if line.len().saturating_add(take) > MAX_RECORD_BYTES + 64 {
            return Err(TimelineError::Corrupt {
                offset,
                reason: "record exceeds maximum size".to_string(),
            });
        }
        let ended = chunk[take - 1] == b'\n';
        line.extend_from_slice(&chunk[..take]);
        reader.consume(take);
        if ended {
            return Ok((line, true));
        }
    }
}

fn decode_record(bytes: &[u8], offset: u64) -> Result<TimelineEvent, TimelineError> {
    let text = std::str::from_utf8(bytes).map_err(|error| TimelineError::Corrupt {
        offset,
        reason: format!("invalid UTF-8: {error}"),
    })?;
    if let Some(rest) = text.strip_prefix("V2\t") {
        decode_v2(rest, offset)
    } else {
        TimelineEvent::decode_line(text).ok_or_else(|| TimelineError::Corrupt {
            offset,
            reason: "invalid legacy record".to_string(),
        })
    }
}

fn decode_v2(line: &str, offset: u64) -> Result<TimelineEvent, TimelineError> {
    let corrupt = |reason: &str| TimelineError::Corrupt {
        offset,
        reason: reason.to_string(),
    };
    let mut parts = line.splitn(3, '\t');
    let length: usize = parts
        .next()
        .ok_or_else(|| corrupt("missing length"))?
        .parse()
        .map_err(|_| corrupt("invalid length"))?;
    let checksum =
        u32::from_str_radix(parts.next().ok_or_else(|| corrupt("missing checksum"))?, 16)
            .map_err(|_| corrupt("invalid checksum"))?;
    let body = parts.next().ok_or_else(|| corrupt("missing record body"))?;
    if length != body.len() || crc32fast::hash(body.as_bytes()) != checksum {
        return Err(corrupt("record length or checksum mismatch"));
    }
    TimelineEvent::decode_line(body).ok_or_else(|| corrupt("invalid event body"))
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), TimelineError> {
    let parent = path
        .parent()
        .filter(|directory| !directory.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    for directory in parent.ancestors() {
        // Relative paths end with an empty ancestor, which represents the
        // current directory where the topmost relative name was created.
        let directory = if directory.as_os_str().is_empty() {
            Path::new(".")
        } else {
            directory
        };
        sync_file_durable(&File::open(directory)?)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), TimelineError> {
    Ok(())
}

/// Read at most `limit` events newer than `after_seq` without loading the full WAL.
/// Assumes events have increasing sequence numbers. Live readers should use
/// `read_events_after_up_to` with the writer's committed byte limit.
pub fn read_events_after(
    path: impl AsRef<Path>,
    after_seq: u64,
    limit: usize,
) -> Result<Vec<TimelineEvent>, TimelineError> {
    read_events_after_up_to(path, after_seq, limit, None)
}

/// Read events newer than `after_seq`, stopping at `committed_end` when provided.
/// The byte limit must end at a complete record. Complete corrupt records fail
/// the read. Without a limit, an incomplete final record is left for a later read.
pub fn read_events_after_up_to(
    path: impl AsRef<Path>,
    after_seq: u64,
    limit: usize,
    committed_end: Option<u64>,
) -> Result<Vec<TimelineEvent>, TimelineError> {
    read_events_after_from_up_to(path, after_seq, limit, 0, committed_end).map(|(events, _)| events)
}

/// Continue a WAL scan from a byte offset returned by an earlier call.
/// An offset past the end of the file restarts the scan at zero. The sequence
/// cursor skips records already delivered. Live readers should use
/// `read_events_after_from_up_to` with the writer's committed byte limit.
pub fn read_events_after_from(
    path: impl AsRef<Path>,
    after_seq: u64,
    limit: usize,
    offset: u64,
) -> Result<(Vec<TimelineEvent>, u64), TimelineError> {
    read_events_after_from_up_to(path, after_seq, limit, offset, None)
}

/// Continue a WAL scan without reading past `committed_end` when provided.
/// The returned offset points past the last complete record read. An incomplete
/// tail keeps its starting offset so the next call can read it after completion.
/// Complete corrupt records and an invalid committed byte limit fail the read.
pub fn read_events_after_from_up_to(
    path: impl AsRef<Path>,
    after_seq: u64,
    limit: usize,
    offset: u64,
    committed_end: Option<u64>,
) -> Result<(Vec<TimelineEvent>, u64), TimelineError> {
    if limit == 0 {
        return Ok((Vec::new(), offset.min(committed_end.unwrap_or(offset))));
    }
    let mut file = match OpenOptions::new().read(true).open(path.as_ref()) {
        Ok(file) => file,
        Err(err)
            if err.kind() == std::io::ErrorKind::NotFound && committed_end.unwrap_or(0) == 0 =>
        {
            return Ok((Vec::new(), offset.min(committed_end.unwrap_or(offset))));
        }
        Err(err) => return Err(TimelineError::Io(err)),
    };
    let file_length = file.metadata()?.len();
    let end = committed_end.unwrap_or(file_length);
    if end > file_length {
        return Err(TimelineError::Corrupt {
            offset: file_length,
            reason: format!("file ends before published byte boundary {end}"),
        });
    }
    let start = if offset <= end { offset } else { 0 };
    file.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file.take(end - start));
    let mut out = Vec::with_capacity(limit.min(256));
    let mut next_offset = start;
    while out.len() < limit {
        let (bytes, terminated) = read_bounded_line(&mut reader, next_offset)?;
        if bytes.is_empty() {
            if next_offset != end {
                return Err(TimelineError::Corrupt {
                    offset: next_offset,
                    reason: format!("file ends before published byte boundary {end}"),
                });
            }
            break;
        }
        if !terminated {
            if committed_end.is_some() {
                return Err(TimelineError::Corrupt {
                    offset: next_offset,
                    reason: format!(
                        "published byte boundary {end} does not end at a complete record"
                    ),
                });
            }
            break;
        }
        let event = decode_record(&bytes[..bytes.len() - 1], next_offset)?;
        next_offset += bytes.len() as u64;
        if event.seq > after_seq {
            out.push(event);
        }
    }
    Ok((out, next_offset))
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

/// Single-pass escape: `\` → `\\`, tab → `\t`, newline → `\n`.
/// Writes directly into `out` with no intermediate allocations.
#[inline]
fn sanitize_into(s: &str, out: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc = match b {
            b'\\' => "\\\\",
            b'\t' => "\\t",
            b'\n' => "\\n",
            _ => continue,
        };
        out.extend_from_slice(&bytes[start..i]);
        out.extend_from_slice(esc.as_bytes());
        start = i + 1;
    }
    out.extend_from_slice(&bytes[start..]);
}

#[inline]
fn restore(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut start = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            out.push_str(&s[start..i]);
            i += 1;
            match bytes.get(i).copied() {
                Some(b't') => {
                    out.push('\t');
                    i += 1;
                }
                Some(b'n') => {
                    out.push('\n');
                    i += 1;
                }
                Some(b'\\') => {
                    out.push('\\');
                    i += 1;
                }
                Some(_) => {
                    out.push('\\'); /* leave i at the unrecognised byte */
                }
                None => out.push('\\'),
            }
            start = i;
        } else {
            i += 1;
        }
    }
    out.push_str(&s[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::{
        append_event, framed_len_fields, read_all_events, read_all_events_up_to, read_events_after,
        read_events_after_from, read_events_after_from_up_to, scan_events, DualWriter, EventIndex,
        TimelineError, TimelineEvent, WalWriter, MAX_RECORD_BYTES,
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

    struct TestWalFiles(PathBuf);

    impl Drop for TestWalFiles {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            if let Some(name) = self.0.file_name() {
                let lock = self
                    .0
                    .with_file_name(format!("{}.lock", name.to_string_lossy()));
                let _ = std::fs::remove_file(lock);
            }
        }
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

    #[test]
    fn second_writer_cannot_open_the_same_log() {
        let path = test_path("exclusive-writer");
        let first = WalWriter::open(&path).unwrap();
        assert!(WalWriter::open(&path).is_err());
        drop(first);
        assert!(WalWriter::open(&path).is_ok());
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn dropping_writer_releases_locks_while_duplicated_descriptors_remain_open() {
        let path = test_path("duplicated-lock-descriptors");
        let alias = path.with_extension("hard-link-alias");
        let _path_cleanup = TestWalFiles(path.clone());
        let _alias_cleanup = TestWalFiles(alias.clone());
        let writer = WalWriter::open(&path).unwrap();
        std::fs::hard_link(&path, &alias).unwrap();
        // A forked child holds these same lock references until exec closes
        // its descriptors. Cloning reproduces that lifetime without a fork.
        let inherited_wal = writer.file.try_clone().unwrap();
        let inherited_sidecar = writer._ownership.try_clone().unwrap();
        drop(writer);

        let reopened = WalWriter::open(&path).unwrap();
        assert!(WalWriter::open(&alias).is_err());
        drop(inherited_wal);
        drop(inherited_sidecar);
        assert!(WalWriter::open(&path).is_err());
        assert!(WalWriter::open(&alias).is_err());
        drop(reopened);
        assert!(WalWriter::open(&alias).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn hard_link_alias_cannot_open_the_same_wal_under_another_sidecar_lock() {
        let path = test_path("hard-link-target");
        let alias = path.with_extension("hard-link-alias");
        let _path_cleanup = TestWalFiles(path.clone());
        let _alias_cleanup = TestWalFiles(alias.clone());
        let writer = WalWriter::open(&path).unwrap();
        std::fs::hard_link(&path, &alias).unwrap();

        assert!(WalWriter::open(&alias).is_err());
        drop(writer);
        assert!(WalWriter::open(&alias).is_ok());
    }

    #[test]
    fn framed_size_counts_escaped_bytes_and_rejects_oversize_before_staging() {
        let path = test_path("framed-size");
        let _cleanup = TestWalFiles(path.clone());
        let mut writer = WalWriter::open(&path).unwrap();
        let mut small = event(1);
        small.seq = u64::MAX;
        small.run_id = "\\\t\n".to_string();
        small.stream_id = "caf\u{e9}\u{1f980}".to_string();
        small.pts_ms = u64::MAX;
        assert_eq!(writer.stage(&small).unwrap(), small.framed_len().unwrap());
        writer.sync_pending().unwrap();
        assert_eq!(writer.read_all().unwrap(), vec![small]);

        let committed_end = writer.committed_end();
        let mut huge = event(2);
        huge.payload = "\t".repeat(MAX_RECORD_BYTES / 2 + 1);
        assert!(huge.framed_len().is_err());
        assert!(writer.stage(&huge).is_err());
        assert_eq!(writer.committed_end(), committed_end);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), committed_end);
        assert!(!writer.is_poisoned());
    }

    #[test]
    fn framed_size_accepts_exact_body_limit_and_rejects_one_more_byte() {
        // Two one-digit numbers and five tabs leave the rest for the payload.
        let mut payload = "x".repeat(MAX_RECORD_BYTES - 7);
        assert_eq!(
            framed_len_fields(1, "", "", 0, "", &payload).unwrap(),
            MAX_RECORD_BYTES + 22
        );
        payload.push('x');
        assert!(framed_len_fields(1, "", "", 0, "", &payload).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_alias_cannot_open_the_same_wal_under_another_lock() {
        use std::os::unix::fs::symlink;

        let path = test_path("symlink-target");
        let alias = path.with_extension("alias");
        let writer = WalWriter::open(&path).unwrap();
        drop(writer);
        symlink(&path, &alias).unwrap();

        assert!(WalWriter::open(&alias).is_err());
        let _ = std::fs::remove_file(alias);
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_lock_file_is_rejected() {
        use std::os::unix::fs::symlink;

        let path = test_path("symlink-lock");
        let lock = path.with_file_name(format!(
            "{}.lock",
            path.file_name().unwrap().to_string_lossy()
        ));
        let target = path.with_extension("lock-target");
        std::fs::write(&target, b"lock target").unwrap();
        symlink(&target, &lock).unwrap();

        assert!(WalWriter::open(&path).is_err());
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(lock);
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn relative_wal_path_opens_and_appends() {
        let path = std::path::PathBuf::from(format!(
            "vidarax-relative-wal-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _cleanup = TestWalFiles(path.clone());
        let mut writer = WalWriter::open(&path).expect("open relative WAL path");
        assert!(writer.path().is_absolute());
        writer.append(&event(1)).unwrap();
        assert_eq!(writer.read_all().unwrap(), vec![event(1)]);
    }

    #[test]
    fn validation_scan_checks_the_boundary_without_retaining_events() {
        let path = test_path("validation-scan");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.append(&event(1)).unwrap();
        writer.append(&event(2)).unwrap();
        drop(writer);

        let file = std::fs::File::open(&path).unwrap();
        let scan = scan_events(file, None, false).unwrap();
        assert!(scan.events.is_empty());
        assert_eq!(
            scan.last_complete_end,
            std::fs::metadata(&path).unwrap().len()
        );
        assert!(!scan.incomplete_tail);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reopening_discards_only_an_unterminated_new_record() {
        let path = test_path("incomplete-tail");
        let mut first = WalWriter::open(&path).unwrap();
        first.append(&event(1)).unwrap();
        drop(first);

        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"V2\t100\tdeadbeef\tpartial")
            .unwrap();

        let mut reopened = WalWriter::open(&path).unwrap();
        reopened.append(&event(2)).unwrap();
        assert_eq!(reopened.read_all().unwrap(), vec![event(1), event(2)]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reopening_repairs_every_short_new_record_prefix() {
        for suffix in [b"V".as_slice(), b"V2", b"V2\t"] {
            let path = test_path("short-v2-prefix");
            let mut first = WalWriter::open(&path).unwrap();
            first.append(&event(1)).unwrap();
            drop(first);
            use std::io::Write;
            std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(suffix)
                .unwrap();
            let mut reopened = WalWriter::open(&path).unwrap();
            reopened.append(&event(2)).unwrap();
            assert_eq!(reopened.read_all().unwrap(), vec![event(1), event(2)]);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn plausible_but_unterminated_legacy_tail_is_not_replayed_or_sealed() {
        let path = test_path("legacy-torn-tail");
        std::fs::write(&path, b"1\trun-1\tstream-1\t10\trun_deleted\t{\"partial\":").unwrap();
        assert!(read_all_events(&path).unwrap().is_empty());
        let mut reopened = WalWriter::open(&path).unwrap();
        assert!(reopened.read_all().unwrap().is_empty());
        reopened.append(&event(1)).unwrap();
        assert_eq!(reopened.read_all().unwrap(), vec![event(1)]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn complete_corrupt_record_is_an_error_instead_of_a_missing_event() {
        let path = test_path("corrupt-record");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.append(&event(1)).unwrap();
        drop(writer);

        let mut bytes = std::fs::read(&path).unwrap();
        let pos = bytes.iter().position(|byte| *byte == b'{').unwrap();
        bytes[pos] = b'!';
        std::fs::write(&path, bytes).unwrap();
        assert!(read_all_events(&path).is_err());
        assert!(WalWriter::open(&path).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn failed_sync_poisons_writer_and_recovery_keeps_complete_record() {
        let path = test_path("sync-failure");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.fail_sync_once_for_tests();

        assert!(writer.append(&event(1)).is_err());
        assert!(writer.append(&event(2)).is_err());
        assert_eq!(
            read_all_events_up_to(&path, Some(writer.committed_end())).unwrap(),
            vec![]
        );
        drop(writer);

        let mut recovered = WalWriter::open(&path).unwrap();
        assert_eq!(recovered.read_all().unwrap(), vec![event(1)]);
        recovered.append(&event(2)).unwrap();
        assert_eq!(recovered.read_all().unwrap(), vec![event(1), event(2)]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn committed_reader_rejects_a_file_shorter_than_its_published_boundary() {
        let path = test_path("committed-file-truncated");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.append(&event(1)).unwrap();
        let committed_end = writer.committed_end();
        writer.file.set_len(committed_end - 1).unwrap();
        assert!(read_all_events_up_to(&path, Some(committed_end)).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn staged_records_remain_uncommitted_until_one_sync() {
        let path = test_path("staged-boundary");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.stage(&event(1)).unwrap();
        writer.stage(&event(2)).unwrap();

        assert_eq!(writer.committed_end(), 0);
        assert!(writer.read_all().unwrap().is_empty());
        writer.sync_pending().unwrap();

        assert_eq!(writer.read_all().unwrap(), vec![event(1), event(2)]);
        assert_eq!(
            writer.committed_end(),
            std::fs::metadata(&path).unwrap().len()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn failed_group_sync_poisons_all_staged_records() {
        let path = test_path("staged-sync-failure");
        let mut writer = WalWriter::open(&path).unwrap();
        writer.stage(&event(1)).unwrap();
        writer.stage(&event(2)).unwrap();
        writer.fail_sync_once_for_tests();

        assert!(writer.sync_pending().is_err());
        assert!(writer.is_poisoned());
        assert_eq!(writer.committed_end(), 0);
        assert!(writer.stage(&event(3)).is_err());
        assert!(writer.read_all().unwrap().is_empty());
        drop(writer);

        let recovered = WalWriter::open(&path).unwrap();
        assert_eq!(recovered.read_all().unwrap(), vec![event(1), event(2)]);
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn bounded_cursor_read_is_strictly_after_and_capped() {
        let path = test_path("cursor");
        for seq in 1..=8 {
            append_event(&path, &event(seq)).unwrap();
        }
        let events = read_events_after(&path, 3, 2).unwrap();
        assert_eq!(
            events.iter().map(|event| event.seq).collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert!(read_events_after(&path, 8, 2).unwrap().is_empty());
        assert!(read_events_after(&path, 0, 0).unwrap().is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cursor_read_continues_from_returned_byte_offset() {
        let path = test_path("cursor-offset");
        for seq in 1..=3 {
            append_event(&path, &event(seq)).unwrap();
        }
        let (first, offset) = read_events_after_from(&path, 0, 3, 0).unwrap();
        assert_eq!(first.len(), 3);
        append_event(&path, &event(4)).unwrap();
        let (next, next_offset) = read_events_after_from(&path, 3, 3, offset).unwrap();
        assert_eq!(
            next.iter().map(|event| event.seq).collect::<Vec<_>>(),
            vec![4]
        );
        assert!(next_offset > offset);
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn cursor_read_handles_legacy_records_followed_by_v2() {
        let path = test_path("cursor-mixed");
        let _cleanup = TestWalFiles(path.clone());
        std::fs::write(&path, b"1\trun-1\tstream-1\t10\tkeepframe\t{}\n").unwrap();
        append_event(&path, &event(2)).unwrap();
        assert_eq!(
            read_events_after(&path, 0, 2).unwrap(),
            vec![event(1), event(2)]
        );
    }

    #[test]
    fn cursor_read_stops_at_committed_bytes_and_resumes_after_sync() {
        let path = test_path("cursor-committed");
        let _cleanup = TestWalFiles(path.clone());
        let mut writer = WalWriter::open(&path).unwrap();
        writer.append(&event(1)).unwrap();
        let committed_end = writer.committed_end();
        writer.stage(&event(2)).unwrap();

        let (first, offset) =
            read_events_after_from_up_to(&path, 0, 8, 0, Some(committed_end)).unwrap();
        assert_eq!(first, vec![event(1)]);
        assert_eq!(offset, committed_end);
        let (pending, offset) =
            read_events_after_from_up_to(&path, 1, 8, offset, Some(committed_end)).unwrap();
        assert!(pending.is_empty());
        assert_eq!(offset, committed_end);

        writer.sync_pending().unwrap();
        let (next, offset) =
            read_events_after_from_up_to(&path, 1, 8, offset, Some(writer.committed_end()))
                .unwrap();
        assert_eq!(next, vec![event(2)]);
        assert_eq!(offset, writer.committed_end());
    }

    #[test]
    fn cursor_read_retries_incomplete_record_from_its_start() {
        let path = test_path("cursor-incomplete");
        let record_path = test_path("cursor-incomplete-record");
        let _cleanup = TestWalFiles(path.clone());
        let _record_cleanup = TestWalFiles(record_path.clone());
        append_event(&path, &event(1)).unwrap();
        append_event(&record_path, &event(2)).unwrap();
        let complete_end = std::fs::metadata(&path).unwrap().len();
        let record = std::fs::read(record_path).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write;
        file.write_all(&record[..record.len() - 1]).unwrap();

        let (first, offset) = read_events_after_from(&path, 0, 8, 0).unwrap();
        assert_eq!(first, vec![event(1)]);
        assert_eq!(offset, complete_end);
        let (incomplete, offset) = read_events_after_from(&path, 1, 8, offset).unwrap();
        assert!(incomplete.is_empty());
        assert_eq!(offset, complete_end);

        file.write_all(b"\n").unwrap();
        let (completed, offset) = read_events_after_from(&path, 1, 8, offset).unwrap();
        assert_eq!(completed, vec![event(2)]);
        assert_eq!(offset, std::fs::metadata(&path).unwrap().len());
    }

    #[test]
    fn cursor_read_rejects_corruption_and_invalid_committed_bytes() {
        let path = test_path("cursor-invalid");
        let _cleanup = TestWalFiles(path.clone());
        append_event(&path, &event(1)).unwrap();
        let end = std::fs::metadata(&path).unwrap().len();
        assert!(read_events_after_from_up_to(&path, 0, 8, 0, Some(end - 1)).is_err());
        assert!(read_events_after_from_up_to(&path, 0, 8, 0, Some(end + 1)).is_err());

        let mut bytes = std::fs::read(&path).unwrap();
        let pos = bytes.iter().position(|byte| *byte == b'{').unwrap();
        bytes[pos] = b'!';
        std::fs::write(&path, bytes).unwrap();
        assert!(read_events_after(&path, 0, 8).is_err());
        std::fs::write(&path, b"invalid legacy record\n").unwrap();
        assert!(read_events_after(&path, 0, 8).is_err());
    }
}
