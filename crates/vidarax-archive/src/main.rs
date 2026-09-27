use std::error::Error;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use object_store::aws::{AmazonS3, AmazonS3Builder, S3ConditionalPut};
use vidarax_archive::{archive, restore, ArchiveOptions};

#[derive(Parser)]
#[command(
    name = "vidarax-archive",
    about = "Offline Vidarax WAL and keyframe archive"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Stop the API, then upload a complete local snapshot and publish its manifest.
    Archive {
        #[arg(long)]
        data_dir: PathBuf,
        #[command(flatten)]
        store: StoreArgs,
        #[arg(long, default_value = "vidarax")]
        prefix: String,
        /// Target WAL chunk size in MiB (1-64). One large record can exceed it.
        #[arg(long, default_value_t = 8)]
        chunk_mib: usize,
    },
    /// Restore a named manifest into a data directory that does not exist yet.
    Restore {
        #[arg(long)]
        target_dir: PathBuf,
        #[arg(long)]
        manifest_key: String,
        #[command(flatten)]
        store: StoreArgs,
    },
}

#[derive(Args)]
struct StoreArgs {
    #[arg(long)]
    bucket: String,
    #[arg(long)]
    region: String,
    /// Optional S3-compatible endpoint, such as a private MinIO address.
    #[arg(long)]
    endpoint: Option<String>,
    /// Permit HTTP only for a trusted local test endpoint.
    #[arg(long, default_value_t = false)]
    allow_http: bool,
}

fn open_store(args: &StoreArgs) -> Result<AmazonS3, Box<dyn Error + Send + Sync>> {
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(&args.bucket)
        .with_region(&args.region)
        .with_conditional_put(S3ConditionalPut::ETagMatch)
        .with_allow_http(args.allow_http);
    if let Some(endpoint) = &args.endpoint {
        builder = builder.with_endpoint(endpoint);
    }
    Ok(builder.build()?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Archive {
            data_dir,
            store,
            prefix,
            chunk_mib,
        } => {
            if !(1..=64).contains(&chunk_mib) {
                return Err("--chunk-mib must be between 1 and 64".into());
            }
            let store = open_store(&store)?;
            let result = archive(
                &data_dir,
                &store,
                &ArchiveOptions {
                    prefix,
                    chunk_bytes: chunk_mib * 1024 * 1024,
                },
            )
            .await?;
            println!(
                "{}",
                serde_json::json!({
                    "manifest_key": result.manifest_key,
                    "events": result.manifest.event_count,
                    "event_coverage_seq": result.manifest.event_coverage_seq,
                    "evidence_coverage_seq": result.manifest.evidence_coverage_seq,
                    "wal_chunks": result.manifest.wal_chunks.len(),
                    "keyframes": result.manifest.keyframes.len(),
                })
            );
        }
        Command::Restore {
            target_dir,
            manifest_key,
            store,
        } => {
            let store = open_store(&store)?;
            let manifest = restore(&target_dir, &store, &manifest_key).await?;
            println!(
                "{}",
                serde_json::json!({
                    "target_dir": target_dir,
                    "events": manifest.event_count,
                    "event_coverage_seq": manifest.event_coverage_seq,
                    "evidence_coverage_seq": manifest.evidence_coverage_seq,
                })
            );
        }
    }
    Ok(())
}
