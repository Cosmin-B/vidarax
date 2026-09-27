//! Back up the WAL and its keyframes while the API is stopped.
//!
//! The archive holds the WAL lock until it finishes. It publishes the manifest
//! only after all referenced objects have been uploaded and checked. A failed
//! attempt can leave uploaded objects that no manifest references.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

use futures_util::StreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vidarax_core::timeline::{read_all_events, sync_file_durable, TimelineEvent, WalWriter};

pub type ArchiveResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const FORMAT_VERSION: u32 = 1;
const MAX_RECORD_LINE_BYTES: usize = 64 * 1024 * 1024 + 64;
const MAX_BLOB_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_ITEMS: usize = 100_000;

#[derive(Debug, Clone)]
pub struct ArchiveOptions {
    pub prefix: String,
    pub chunk_bytes: usize,
}

impl Default for ArchiveOptions {
    fn default() -> Self {
        Self {
            prefix: "vidarax".to_string(),
            chunk_bytes: 8 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredObject {
    pub key: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalChunk {
    pub object: StoredObject,
    pub first_seq: u64,
    pub last_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub wal_chunks: Vec<WalChunk>,
    pub keyframes: Vec<StoredObject>,
    pub event_count: u64,
    pub event_coverage_seq: u64,
    pub evidence_coverage_seq: u64,
}

#[derive(Debug, Clone)]
pub struct PublishedArchive {
    pub manifest_key: String,
    pub manifest: Manifest,
}

pub async fn archive<S: ObjectStore>(
    data_dir: &Path,
    store: &S,
    options: &ArchiveOptions,
) -> ArchiveResult<PublishedArchive> {
    archive_with_manifest_limit(data_dir, store, options, MAX_MANIFEST_BYTES).await
}

async fn archive_with_manifest_limit<S: ObjectStore>(
    data_dir: &Path,
    store: &S,
    options: &ArchiveOptions,
    manifest_limit: usize,
) -> ArchiveResult<PublishedArchive> {
    let prefix = validate_prefix(&options.prefix)?;
    if options.chunk_bytes == 0 || options.chunk_bytes > 64 * 1024 * 1024 {
        return Err(invalid("chunk size must be between 1 byte and 64 MiB"));
    }
    if !fs::symlink_metadata(data_dir)?.file_type().is_dir() {
        return Err(invalid("archive data directory must be a real directory"));
    }
    let data_dir = fs::canonicalize(data_dir)?;
    let wal_path = data_dir.join("timeline.wal");
    if !wal_path.exists() {
        return Err(invalid(format!(
            "WAL does not exist: {}",
            wal_path.display()
        )));
    }
    if !fs::symlink_metadata(&wal_path)?.file_type().is_file() {
        return Err(invalid("WAL must be a regular file, not a symlink"));
    }

    // The API takes the same lock, so archiving can proceed only while its
    // writer is stopped. Opening the WAL also removes an incomplete final record.
    let ownership = WalWriter::open(&wal_path)?;
    let committed_end = ownership.committed_end();
    let source = File::open(&wal_path)?;
    let mut reader = BufReader::new(source.take(committed_end));
    let mut bytes_read = 0u64;
    let mut chunk = Vec::with_capacity(options.chunk_bytes.min(1024 * 1024));
    let mut chunk_line_count = 0usize;
    let mut next_seq = 1u64;
    let mut wal_chunks = Vec::new();
    let mut keyframes = BTreeMap::<String, StoredObject>::new();

    while let Some(line) = read_bounded_line(&mut reader)? {
        bytes_read = bytes_read
            .checked_add(line.len() as u64)
            .ok_or_else(|| invalid("WAL byte count overflow"))?;
        if !chunk.is_empty() && chunk.len().saturating_add(line.len()) > options.chunk_bytes {
            let stored = prepare_chunk(
                &prefix,
                std::mem::take(&mut chunk),
                chunk_line_count,
                &mut next_seq,
                &mut keyframes,
            )?;
            wal_chunks.push(stored);
            chunk_line_count = 0;
            if wal_chunks.len() > MAX_MANIFEST_ITEMS {
                return Err(invalid("archive has too many WAL chunks"));
            }
        }
        chunk.extend_from_slice(&line);
        chunk_line_count += 1;
    }
    if bytes_read != committed_end {
        return Err(invalid(
            "WAL changed or ended before its committed boundary",
        ));
    }
    if !chunk.is_empty() {
        if wal_chunks.len() >= MAX_MANIFEST_ITEMS {
            return Err(invalid("archive has too many WAL chunks"));
        }
        wal_chunks.push(prepare_chunk(
            &prefix,
            chunk,
            chunk_line_count,
            &mut next_seq,
            &mut keyframes,
        )?);
    }
    let coverage = next_seq - 1;
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        wal_chunks,
        keyframes: keyframes.into_values().collect(),
        event_count: coverage,
        event_coverage_seq: coverage,
        evidence_coverage_seq: coverage,
    };
    validate_manifest(&manifest)?;
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    if manifest_bytes.len() > manifest_limit.min(MAX_MANIFEST_BYTES) {
        return Err(invalid("manifest exceeds 16 MiB"));
    }

    // Check the complete manifest size before uploading anything. Read the WAL
    // again and compare each chunk with its saved hash so changed bytes cannot
    // be uploaded under a name derived from the old contents.
    let mut source = File::open(&wal_path)?.take(committed_end);
    for chunk in &manifest.wal_chunks {
        let size = usize::try_from(chunk.object.bytes)
            .map_err(|_| invalid("WAL chunk byte count exceeds address space"))?;
        let mut bytes = vec![0; size];
        source.read_exact(&mut bytes)?;
        verify_object(&chunk.object, &bytes)?;
        put_verified(store, &chunk.object.key, bytes, MAX_RECORD_LINE_BYTES).await?;
    }
    if source.limit() != 0 {
        return Err(invalid(
            "WAL preflight did not cover its committed boundary",
        ));
    }
    for blob in &manifest.keyframes {
        let bytes = read_keyframe(&data_dir, &blob.sha256, blob.bytes)?;
        put_verified(store, &blob.key, bytes, MAX_BLOB_BYTES).await?;
    }
    let hash = sha256_hex(&manifest_bytes);
    let manifest_key = format!("{prefix}/manifests/{hash}.json");
    put_verified(store, &manifest_key, manifest_bytes, MAX_MANIFEST_BYTES).await?;
    drop(ownership);
    Ok(PublishedArchive {
        manifest_key,
        manifest,
    })
}

pub async fn restore<S: ObjectStore>(
    target_dir: &Path,
    store: &S,
    manifest_key: &str,
) -> ArchiveResult<Manifest> {
    let parent = target_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .or_else(|| (!target_dir.is_absolute()).then_some(Path::new(".")))
        .ok_or_else(|| invalid("restore target has no parent directory"))?;
    if !fs::symlink_metadata(parent)?.file_type().is_dir() {
        return Err(invalid("restore target parent must be a real directory"));
    }
    let parent = fs::canonicalize(parent)?;
    let target_name = target_dir
        .file_name()
        .ok_or_else(|| invalid("restore target has no directory name"))?;
    let target_dir = parent.join(target_name);
    if path_entry_exists(&target_dir)? {
        return Err(invalid(format!(
            "restore target already exists: {}",
            target_dir.display()
        )));
    }
    let manifest_path = ObjectPath::parse(manifest_key)?;
    let manifest_sha = manifest_key
        .rsplit('/')
        .next()
        .and_then(|name| name.strip_suffix(".json"))
        .ok_or_else(|| invalid("manifest key must end in <sha256>.json"))?;
    validate_sha(manifest_sha)?;
    let manifest_bytes = get_limited(store, &manifest_path, MAX_MANIFEST_BYTES).await?;
    if sha256_hex(&manifest_bytes) != manifest_sha {
        return Err(invalid("manifest SHA-256 mismatch"));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest)?;

    let staging = tempfile::Builder::new()
        .prefix(".vidarax-restore-")
        .tempdir_in(&parent)?;
    restrict_staging_directory(staging.path())?;
    let wal_path = staging.path().join("timeline.wal");
    let mut wal = create_restricted_file(&wal_path)?;
    let mut next_seq = 1u64;
    let mut referenced_keyframes = BTreeMap::<String, u64>::new();
    for chunk in &manifest.wal_chunks {
        let object_path = ObjectPath::parse(&chunk.object.key)?;
        let bytes = get_limited(store, &object_path, MAX_RECORD_LINE_BYTES).await?;
        verify_object(&chunk.object, &bytes)?;
        let events = validate_chunk(&bytes)?;
        if events.first().map(|event| event.seq) != Some(chunk.first_seq)
            || events.last().map(|event| event.seq) != Some(chunk.last_seq)
        {
            return Err(invalid("WAL chunk sequence range differs from manifest"));
        }
        for event in &events {
            if event.seq != next_seq {
                return Err(invalid(format!(
                    "WAL sequence gap: expected {next_seq}, found {}",
                    event.seq
                )));
            }
            next_seq += 1;
            if let Some((sha, bytes)) = keyframe_reference(event)? {
                if referenced_keyframes
                    .insert(sha, bytes)
                    .is_some_and(|previous| previous != bytes)
                {
                    return Err(invalid("one keyframe hash has conflicting byte counts"));
                }
            }
        }
        wal.write_all(&bytes)?;
    }
    if next_seq - 1 != manifest.event_coverage_seq {
        return Err(invalid("restored event coverage differs from manifest"));
    }
    if referenced_keyframes.len() != manifest.keyframes.len()
        || manifest
            .keyframes
            .iter()
            .any(|blob| referenced_keyframes.get(&blob.sha256) != Some(&blob.bytes))
    {
        return Err(invalid(
            "manifest does not contain every referenced keyframe",
        ));
    }
    sync_file_durable(&wal)?;
    drop(wal);

    for blob in &manifest.keyframes {
        validate_sha(&blob.sha256)?;
        let object_path = ObjectPath::parse(&blob.key)?;
        let bytes = get_limited(store, &object_path, MAX_BLOB_BYTES).await?;
        verify_object(blob, &bytes)?;
        let shard = &blob.sha256[..2];
        let dir = staging.path().join("keyframes/blobs").join(shard);
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.jpg", blob.sha256));
        let mut file = create_restricted_file(&path)?;
        file.write_all(&bytes)?;
        sync_file_durable(&file)?;
        sync_directory(&dir)?;
    }
    let blob_root = staging.path().join("keyframes/blobs");
    if blob_root.exists() {
        sync_directory(&blob_root)?;
        sync_directory(&staging.path().join("keyframes"))?;
    }
    sync_directory(staging.path())?;
    if path_entry_exists(&target_dir)? {
        return Err(invalid("restore target appeared during download"));
    }
    rename_no_replace(staging.path(), &target_dir)?;
    sync_directory(&parent).map_err(|error| {
        invalid(format!(
            "restore target published at {} but final parent-directory sync failed; outcome uncertain, inspect target: {error}",
            target_dir.display()
        ))
    })?;
    Ok(manifest)
}

fn prepare_chunk(
    prefix: &str,
    bytes: Vec<u8>,
    line_count: usize,
    next_seq: &mut u64,
    keyframes: &mut BTreeMap<String, StoredObject>,
) -> ArchiveResult<WalChunk> {
    if bytes.len() > MAX_RECORD_LINE_BYTES {
        return Err(invalid("WAL chunk exceeds 64 MiB record limit"));
    }
    let events = validate_chunk(&bytes)?;
    if events.len() != line_count || events.is_empty() {
        return Err(invalid("WAL chunk did not decode every record"));
    }
    let first_seq = *next_seq;
    for event in &events {
        if event.seq != *next_seq {
            return Err(invalid(format!(
                "WAL sequence gap: expected {}, found {}",
                *next_seq, event.seq
            )));
        }
        *next_seq = next_seq
            .checked_add(1)
            .ok_or_else(|| invalid("sequence overflow"))?;
        prepare_keyframe(prefix, event, keyframes)?;
    }
    let last_seq = *next_seq - 1;
    let sha = sha256_hex(&bytes);
    let key = format!("{prefix}/wal/{first_seq:020}-{last_seq:020}-{sha}.wal");
    let byte_count = bytes.len() as u64;
    Ok(WalChunk {
        object: StoredObject {
            key,
            sha256: sha,
            bytes: byte_count,
        },
        first_seq,
        last_seq,
    })
}

fn prepare_keyframe(
    prefix: &str,
    event: &TimelineEvent,
    keyframes: &mut BTreeMap<String, StoredObject>,
) -> ArchiveResult<()> {
    let Some((sha, expected_bytes)) = keyframe_reference(event)? else {
        return Ok(());
    };
    if let Some(previous) = keyframes.get(&sha) {
        if previous.bytes != expected_bytes {
            return Err(invalid(
                "duplicate keyframe hash has conflicting byte count",
            ));
        }
        return Ok(());
    }
    if keyframes.len() >= MAX_MANIFEST_ITEMS {
        return Err(invalid("archive has too many distinct keyframes"));
    }
    let key = format!("{prefix}/blobs/{sha}.jpg");
    keyframes.insert(
        sha.clone(),
        StoredObject {
            key,
            sha256: sha,
            bytes: expected_bytes,
        },
    );
    Ok(())
}

fn keyframe_reference(event: &TimelineEvent) -> ArchiveResult<Option<(String, u64)>> {
    let nested = match event.kind.as_str() {
        "keyframe_stored" => false,
        vidarax_core::zone::RESTRICTED_ZONE_ACTIVITY_EVENT => true,
        kind if kind.starts_with("trigger.") => true,
        _ => return Ok(None),
    };
    let payload: serde_json::Value = serde_json::from_str(&event.payload)?;
    let reference = if nested {
        payload
            .get("evidence")
            .ok_or_else(|| invalid("keyframe event has no evidence object"))?
    } else {
        &payload
    };
    let sha = reference
        .get("image_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid("keyframe event has no image_sha256"))?;
    validate_sha(sha)?;
    let expected_ref = format!("keyframes/blobs/{}/{}.jpg", &sha[..2], sha);
    if reference
        .get("image_ref")
        .and_then(serde_json::Value::as_str)
        != Some(expected_ref.as_str())
    {
        return Err(invalid("keyframe image_ref does not match its SHA-256"));
    }
    let expected_bytes = reference
        .get("image_bytes")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| invalid("keyframe event has no image_bytes"))?;
    if expected_bytes > MAX_BLOB_BYTES as u64 {
        return Err(invalid("keyframe exceeds 16 MiB archive limit"));
    }
    Ok(Some((sha.to_string(), expected_bytes)))
}

fn read_keyframe(data_dir: &Path, sha: &str, expected_bytes: u64) -> ArchiveResult<Vec<u8>> {
    let keyframes = data_dir.join("keyframes");
    let blobs = keyframes.join("blobs");
    let shard = blobs.join(&sha[..2]);
    for directory in [&keyframes, &blobs, &shard] {
        if !fs::symlink_metadata(directory)?.file_type().is_dir() {
            return Err(invalid(format!(
                "keyframe path contains a symlink or non-directory: {}",
                directory.display()
            )));
        }
    }
    let path = shard.join(format!("{sha}.jpg"));
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() != expected_bytes {
        return Err(invalid(
            "keyframe is not a regular file of the declared size",
        ));
    }
    let bytes = fs::read(path)?;
    if bytes.len() as u64 != expected_bytes || sha256_hex(&bytes) != sha {
        return Err(invalid("keyframe bytes do not match event metadata"));
    }
    Ok(bytes)
}

fn validate_chunk(bytes: &[u8]) -> ArchiveResult<Vec<TimelineEvent>> {
    if bytes.is_empty() || !bytes.ends_with(b"\n") {
        return Err(invalid("WAL chunk is empty or has an unterminated record"));
    }
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(bytes)?;
    temporary.flush()?;
    Ok(read_all_events(temporary.path())?)
}

async fn put_verified<S: ObjectStore>(
    store: &S,
    key: &str,
    bytes: Vec<u8>,
    limit: usize,
) -> ArchiveResult<()> {
    if bytes.len() > limit {
        return Err(invalid("object exceeds its archive limit"));
    }
    let expected_sha = sha256_hex(&bytes);
    let expected_size = bytes.len();
    let path = ObjectPath::parse(key)?;
    match store
        .put_opts(&path, bytes.into(), PutMode::Create.into())
        .await
    {
        Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => {}
        Err(error) => return Err(Box::new(error)),
    }
    let actual = get_limited(store, &path, limit).await?;
    if actual.len() != expected_size || sha256_hex(&actual) != expected_sha {
        return Err(invalid(format!(
            "uploaded object differs from source: {key}"
        )));
    }
    Ok(())
}

async fn get_limited<S: ObjectStore>(
    store: &S,
    key: &ObjectPath,
    limit: usize,
) -> ArchiveResult<Vec<u8>> {
    let response = store.get(key).await?;
    if response.meta.size > limit as u64 {
        return Err(invalid(format!(
            "remote object exceeds {limit} bytes: {key}"
        )));
    }
    let mut stream = response.into_stream();
    let mut bytes = Vec::new();
    while let Some(part) = stream.next().await {
        let part = part?;
        if part.len() > limit.saturating_sub(bytes.len()) {
            return Err(invalid(format!(
                "remote object exceeds {limit} bytes: {key}"
            )));
        }
        bytes.extend_from_slice(&part);
    }
    Ok(bytes)
}

fn validate_manifest(manifest: &Manifest) -> ArchiveResult<()> {
    if manifest.format_version != FORMAT_VERSION {
        return Err(invalid("unsupported manifest format"));
    }
    if manifest.wal_chunks.len() > MAX_MANIFEST_ITEMS
        || manifest.keyframes.len() > MAX_MANIFEST_ITEMS
    {
        return Err(invalid("manifest has too many objects"));
    }
    let mut next = 1u64;
    for chunk in &manifest.wal_chunks {
        validate_sha(&chunk.object.sha256)?;
        if chunk.object.bytes == 0
            || chunk.object.bytes > MAX_RECORD_LINE_BYTES as u64
            || chunk.first_seq != next
            || chunk.last_seq < chunk.first_seq
        {
            return Err(invalid("manifest WAL chunk range or size is invalid"));
        }
        next = chunk
            .last_seq
            .checked_add(1)
            .ok_or_else(|| invalid("manifest sequence overflow"))?;
    }
    if manifest.event_count != next - 1
        || manifest.event_coverage_seq != next - 1
        || manifest.evidence_coverage_seq != next - 1
    {
        return Err(invalid("manifest coverage is inconsistent"));
    }
    let mut previous = None;
    for blob in &manifest.keyframes {
        validate_sha(&blob.sha256)?;
        if blob.bytes > MAX_BLOB_BYTES as u64
            || previous.is_some_and(|sha| sha >= blob.sha256.as_str())
        {
            return Err(invalid("manifest keyframes are invalid or unsorted"));
        }
        previous = Some(blob.sha256.as_str());
    }
    Ok(())
}

fn verify_object(reference: &StoredObject, bytes: &[u8]) -> ArchiveResult<()> {
    validate_sha(&reference.sha256)?;
    if bytes.len() as u64 != reference.bytes || sha256_hex(bytes) != reference.sha256 {
        return Err(invalid(format!(
            "object hash or length mismatch: {}",
            reference.key
        )));
    }
    Ok(())
}

fn validate_sha(sha: &str) -> ArchiveResult<()> {
    if sha.len() != 64
        || !sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("expected lowercase 64-character SHA-256"));
    }
    Ok(())
}

fn validate_prefix(prefix: &str) -> ArchiveResult<String> {
    if prefix.is_empty()
        || prefix.len() > 128
        || prefix.split('/').any(|segment| {
            segment.is_empty()
                || segment == "."
                || segment == ".."
                || !segment
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        })
    {
        return Err(invalid("archive prefix must be 1-128 safe path characters"));
    }
    Ok(prefix.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded_line(reader: &mut impl BufRead) -> ArchiveResult<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(invalid("WAL has an unterminated record"))
            };
        }
        let length = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |i| i + 1);
        if length > MAX_RECORD_LINE_BYTES.saturating_sub(line.len()) {
            return Err(invalid("WAL record exceeds 64 MiB limit"));
        }
        let ended = available[length - 1] == b'\n';
        line.extend_from_slice(&available[..length]);
        reader.consume(length);
        if ended {
            return Ok(Some(line));
        }
    }
}

fn invalid(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn path_entry_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn rename_no_replace(from: &Path, to: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let from = CString::new(from.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let to = CString::new(to.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "target path contains NUL"))?;
    #[cfg(target_os = "macos")]
    let result = unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) };
    #[cfg(target_os = "linux")]
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn rename_no_replace(_from: &Path, _to: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace restore is only available on macOS and Linux",
    ))
}

fn create_restricted_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

#[cfg(unix)]
fn restrict_staging_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn restrict_staging_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    sync_file_durable(&File::open(path)?)
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use object_store::memory::InMemory;
    use object_store::path::Path as ObjectPath;
    use object_store::ObjectStoreExt;
    use sha2::{Digest, Sha256};
    use std::fs;
    use vidarax_core::timeline::{TimelineEvent, WalWriter};

    fn event(seq: u64, kind: &str, payload: String) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: "run-1".into(),
            stream_id: "stream-1".into(),
            pts_ms: seq * 100,
            kind: kind.into(),
            payload,
        }
    }

    #[tokio::test]
    async fn roundtrip_multiple_chunks_and_keyframes() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        let jpeg = b"\xff\xd8\xff\xd9";
        let hash = format!("{:x}", Sha256::digest(jpeg));
        let blob = data_dir.join(format!("keyframes/blobs/{}/{}.jpg", &hash[..2], hash));
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, jpeg).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        {
            let mut writer = WalWriter::open(&wal_path).unwrap();
            writer
                .append(&event(1, "run_created", "{}".into()))
                .unwrap();
            writer.append(&event(2, "keyframe_stored", format!(
                "{{\"image_ref\":\"keyframes/blobs/{}/{}.jpg\",\"image_sha256\":\"{}\",\"image_bytes\":4}}",
                &hash[..2], hash, hash
            ))).unwrap();
            writer
                .append(&event(3, "run_completed", "{}".into()))
                .unwrap();
        }
        let store = InMemory::new();
        let published = archive(
            &data_dir,
            &store,
            &ArchiveOptions {
                prefix: "test".into(),
                chunk_bytes: 100,
            },
        )
        .await
        .unwrap();
        assert!(published.manifest.wal_chunks.len() > 1);
        assert_eq!(published.manifest.event_count, 3);
        assert_eq!(published.manifest.event_coverage_seq, 3);
        assert_eq!(published.manifest.evidence_coverage_seq, 3);
        assert_eq!(published.manifest.keyframes.len(), 1);
        let target = root.path().join("restored");
        let recovered = restore(&target, &store, &published.manifest_key)
            .await
            .unwrap();
        assert_eq!(recovered.event_count, 3);
        assert_eq!(
            fs::read(target.join("timeline.wal")).unwrap(),
            fs::read(wal_path).unwrap()
        );
        assert_eq!(
            fs::read(target.join(format!("keyframes/blobs/{}/{}.jpg", &hash[..2], hash))).unwrap(),
            jpeg
        );
        assert_eq!(
            WalWriter::open(target.join("timeline.wal"))
                .unwrap()
                .read_all()
                .unwrap()
                .len(),
            3
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&target).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(target.join("timeline.wal"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(target.join(format!("keyframes/blobs/{}/{}.jpg", &hash[..2], hash)))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[tokio::test]
    async fn roundtrip_nested_jpeg_references_and_opaque_unrelated_payload() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        let mut expected_blobs = Vec::new();
        {
            let mut writer = WalWriter::open(&wal_path).unwrap();
            for (index, kind) in [
                vidarax_core::zone::RESTRICTED_ZONE_ACTIVITY_EVENT,
                "trigger.near_miss",
            ]
            .iter()
            .enumerate()
            {
                let jpeg = vec![0xff, 0xd8, index as u8, 0xff, 0xd9];
                let hash = sha256_hex(&jpeg);
                let image_ref = format!("keyframes/blobs/{}/{}.jpg", &hash[..2], hash);
                let blob = data_dir.join(&image_ref);
                fs::create_dir_all(blob.parent().unwrap()).unwrap();
                fs::write(blob, &jpeg).unwrap();
                let payload = serde_json::json!({
                    "evidence": {
                        "image_ref": image_ref,
                        "image_sha256": hash,
                        "image_bytes": jpeg.len(),
                        "image_media_type": "image/jpeg",
                    }
                });
                writer
                    .append(&event(index as u64 + 1, kind, payload.to_string()))
                    .unwrap();
                expected_blobs.push((image_ref, jpeg));
            }
            writer
                .append(&event(3, "observation", "an opaque legacy payload".into()))
                .unwrap();
        }
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        assert_eq!(published.manifest.keyframes.len(), 2);
        let target = root.path().join("restored");
        restore(&target, &store, &published.manifest_key)
            .await
            .unwrap();
        assert_eq!(
            fs::read(target.join("timeline.wal")).unwrap(),
            fs::read(wal_path).unwrap()
        );
        for (image_ref, jpeg) in expected_blobs {
            assert_eq!(fs::read(target.join(image_ref)).unwrap(), jpeg);
        }
    }

    #[tokio::test]
    async fn archive_rejects_nested_jpeg_metadata_without_a_hash() {
        for kind in [
            vidarax_core::zone::RESTRICTED_ZONE_ACTIVITY_EVENT,
            "trigger.near_miss",
        ] {
            let root = tempfile::tempdir().unwrap();
            WalWriter::open(root.path().join("timeline.wal"))
                .unwrap()
                .append(&event(1, kind, r#"{"evidence":{"image_bytes":4}}"#.into()))
                .unwrap();
            let store = InMemory::new();
            assert!(archive(root.path(), &store, &ArchiveOptions::default())
                .await
                .is_err());
            assert!(store.list(None).next().await.is_none());
        }
    }

    #[tokio::test]
    async fn oversized_manifest_is_rejected_before_any_upload() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        WalWriter::open(data_dir.join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let result = archive_with_manifest_limit(
            &data_dir,
            &store,
            &ArchiveOptions {
                prefix: "test".into(),
                chunk_bytes: 1,
            },
            128,
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("manifest exceeds"));
        assert!(store.list(None).next().await.is_none());
    }

    #[tokio::test]
    async fn archive_requires_offline_writer_and_all_evidence() {
        let root = tempfile::tempdir().unwrap();
        let wal_path = root.path().join("timeline.wal");
        let mut writer = WalWriter::open(&wal_path).unwrap();
        writer
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        assert!(archive(root.path(), &store, &ArchiveOptions::default())
            .await
            .is_err());
        drop(writer);
        writer = WalWriter::open(&wal_path).unwrap();
        writer.append(&event(2, "keyframe_stored", format!(
            "{{\"image_ref\":\"keyframes/blobs/aa/{}.jpg\",\"image_sha256\":\"{}\",\"image_bytes\":4}}",
            "a".repeat(64), "a".repeat(64)
        ))).unwrap();
        drop(writer);
        assert!(archive(root.path(), &store, &ArchiveOptions::default())
            .await
            .is_err());
        let objects = store.list(None).collect::<Vec<_>>().await;
        assert!(objects.iter().all(|object| {
            !object
                .as_ref()
                .unwrap()
                .location
                .to_string()
                .contains("/manifests/")
        }));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn archive_rejects_writer_on_wal_hard_link_alias() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        WalWriter::open(&wal_path)
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let alias = root.path().join("timeline-alias.wal");
        fs::hard_link(&wal_path, &alias).unwrap();
        let _online_writer = WalWriter::open(&alias).unwrap();

        let store = InMemory::new();
        let error = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("WAL inode already owned"));
        assert!(store.list(None).next().await.is_none());
    }

    #[tokio::test]
    async fn restore_refuses_existing_target() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        WalWriter::open(&wal_path)
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        let target = root.path().join("target");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("sentinel"), b"keep").unwrap();
        assert!(restore(&target, &store, &published.manifest_key)
            .await
            .is_err());
        assert_eq!(fs::read(target.join("sentinel")).unwrap(), b"keep");
    }

    #[tokio::test]
    async fn restore_accepts_bare_relative_target_name() {
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        WalWriter::open(data_dir.join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();

        let reserved = tempfile::Builder::new()
            .prefix(".vidarax-relative-restore-test-")
            .tempdir_in(".")
            .unwrap();
        let target = std::path::PathBuf::from(reserved.path().file_name().unwrap());
        reserved.close().unwrap();
        let _cleanup = Cleanup(target.clone());
        restore(&target, &store, &published.manifest_key)
            .await
            .unwrap();
        assert!(target.join("timeline.wal").is_file());
    }

    #[tokio::test]
    async fn archive_accepts_relative_data_directory() {
        let source = tempfile::Builder::new()
            .prefix(".vidarax-relative-archive-test-")
            .tempdir_in(".")
            .unwrap();
        WalWriter::open(source.path().join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let relative_dir = std::path::PathBuf::from(source.path().file_name().unwrap());
        let store = InMemory::new();
        let published = archive(&relative_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        assert_eq!(published.manifest.event_count, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restore_refuses_dangling_symlink_target() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        WalWriter::open(data_dir.join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        let target = root.path().join("target");
        symlink(root.path().join("missing"), &target).unwrap();
        assert!(restore(&target, &store, &published.manifest_key)
            .await
            .is_err());
        assert!(fs::symlink_metadata(&target)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn restore_refuses_symlink_parent() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        WalWriter::open(data_dir.join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        let real_parent = root.path().join("real-parent");
        fs::create_dir(&real_parent).unwrap();
        let alias = root.path().join("parent-alias");
        symlink(&real_parent, &alias).unwrap();
        let target = alias.join("restored");
        assert!(restore(&target, &store, &published.manifest_key)
            .await
            .is_err());
        assert!(!real_parent.join("restored").exists());
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn publishing_staging_never_replaces_an_existing_empty_directory() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("staging");
        let target = root.path().join("target");
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("timeline.wal"), b"staged").unwrap();
        fs::create_dir(&target).unwrap();

        assert!(rename_no_replace(&staging, &target).is_err());
        assert!(staging.join("timeline.wal").exists());
        assert_eq!(fs::read_dir(&target).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn restore_rejects_manifest_missing_referenced_keyframe() {
        for kind in [
            "keyframe_stored",
            vidarax_core::zone::RESTRICTED_ZONE_ACTIVITY_EVENT,
            "trigger.near_miss",
        ] {
            let root = tempfile::tempdir().unwrap();
            let data_dir = root.path().join("source");
            fs::create_dir(&data_dir).unwrap();
            let jpeg = b"\xff\xd8\xff\xd9";
            let hash = sha256_hex(jpeg);
            let image_ref = format!("keyframes/blobs/{}/{}.jpg", &hash[..2], hash);
            let blob = data_dir.join(&image_ref);
            fs::create_dir_all(blob.parent().unwrap()).unwrap();
            fs::write(blob, jpeg).unwrap();
            let reference = serde_json::json!({
                "image_ref": image_ref,
                "image_sha256": hash,
                "image_bytes": jpeg.len(),
            });
            let payload = if kind == "keyframe_stored" {
                reference
            } else {
                serde_json::json!({ "evidence": reference })
            };
            WalWriter::open(data_dir.join("timeline.wal"))
                .unwrap()
                .append(&event(1, kind, payload.to_string()))
                .unwrap();
            let store = InMemory::new();
            let published = archive(&data_dir, &store, &ArchiveOptions::default())
                .await
                .unwrap();
            let mut forged = published.manifest;
            forged.keyframes.clear();
            let bytes = serde_json::to_vec(&forged).unwrap();
            let forged_key = format!("test/manifests/{}.json", sha256_hex(&bytes));
            store
                .put(&ObjectPath::parse(&forged_key).unwrap(), bytes.into())
                .await
                .unwrap();

            let target = root.path().join("restored");
            assert!(
                restore(&target, &store, &forged_key).await.is_err(),
                "restore must reject the missing JPEG for {kind}"
            );
            assert!(
                !target.exists(),
                "failed restore must not expose a data directory"
            );
        }
    }

    #[tokio::test]
    async fn corrupt_remote_chunk_leaves_no_restore_target() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        WalWriter::open(data_dir.join("timeline.wal"))
            .unwrap()
            .append(&event(1, "run_created", "{}".into()))
            .unwrap();
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        let chunk_key = &published.manifest.wal_chunks[0].object.key;
        store
            .put(
                &ObjectPath::parse(chunk_key).unwrap(),
                b"wrong bytes".to_vec().into(),
            )
            .await
            .unwrap();
        let target = root.path().join("restored");
        assert!(restore(&target, &store, &published.manifest_key)
            .await
            .is_err());
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn archive_and_restore_wal_larger_than_64_mib() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        fs::create_dir(&data_dir).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        let payload = "x".repeat(9 * 1024 * 1024);
        {
            let mut writer = WalWriter::open(&wal_path).unwrap();
            for seq in 1..=8 {
                writer
                    .append(&event(seq, "observation", payload.clone()))
                    .unwrap();
            }
        }
        assert!(fs::metadata(&wal_path).unwrap().len() > 64 * 1024 * 1024);
        let store = InMemory::new();
        let published = archive(&data_dir, &store, &ArchiveOptions::default())
            .await
            .unwrap();
        assert_eq!(published.manifest.wal_chunks.len(), 8);
        let target = root.path().join("restored");
        let restored = restore(&target, &store, &published.manifest_key)
            .await
            .unwrap();
        assert_eq!(restored.event_count, 8);
        assert_eq!(
            fs::read(target.join("timeline.wal")).unwrap(),
            fs::read(wal_path).unwrap()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn archive_rejects_symlinked_keyframe_directory() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("source");
        let outside = root.path().join("outside");
        fs::create_dir(&data_dir).unwrap();
        fs::create_dir(&outside).unwrap();
        symlink(&outside, data_dir.join("keyframes")).unwrap();

        let jpeg = b"\xff\xd8\xff\xd9";
        let hash = format!("{:x}", Sha256::digest(jpeg));
        let outside_blob = outside.join(format!("blobs/{}/{}.jpg", &hash[..2], hash));
        fs::create_dir_all(outside_blob.parent().unwrap()).unwrap();
        fs::write(outside_blob, jpeg).unwrap();
        let wal_path = data_dir.join("timeline.wal");
        WalWriter::open(&wal_path).unwrap().append(&event(1, "keyframe_stored", format!(
            "{{\"image_ref\":\"keyframes/blobs/{}/{}.jpg\",\"image_sha256\":\"{}\",\"image_bytes\":4}}",
            &hash[..2], hash, hash
        ))).unwrap();

        let store = InMemory::new();
        let result = archive(&data_dir, &store, &ArchiveOptions::default()).await;
        assert!(
            result.is_err(),
            "archive must not follow keyframe directory symlinks"
        );
    }
}
