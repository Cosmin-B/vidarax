#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::Stdio;

use vidarax_contracts::lifecycle::StreamState;
use vidarax_contracts::models::REQUIRED_MODELS;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .with_target(false)
        .without_time()
        .init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("run-create") => {
            println!(r#"{{"run_id":"run-local-00000001","status":"pending","mode":"balanced"}}"#);
        }
        Some("health") => {
            println!(r#"{{"status":"ok"}}"#);
        }
        Some("models") => {
            for model in REQUIRED_MODELS {
                println!("{model}");
            }
        }
        Some("states") => {
            let states = [
                StreamState::Pending,
                StreamState::Processing,
                StreamState::Completed,
                StreamState::Failed,
                StreamState::Cancelled,
                StreamState::Expired,
            ];
            for state in states {
                println!("{state:?}\tterminal={}", state.is_terminal());
            }
        }
        Some("distill") => {
            let sub = args.next();
            let opts = DistillOpts::parse(args);
            match sub.as_deref() {
                Some("status") => cmd_distill_status(opts).await,
                Some("train") => cmd_distill_train(opts).await,
                Some("export") => cmd_distill_export(opts).await,
                Some("deploy") => cmd_distill_deploy(opts).await,
                _ => {
                    eprintln!("Usage:");
                    eprintln!("  vidarax-cli distill status  --tenant-id <id> [--data-dir <dir>]");
                    eprintln!("  vidarax-cli distill train   --tenant-id <id> [--data-dir <dir>] [--mlx] [--resume]");
                    eprintln!("  vidarax-cli distill export  --tenant-id <id> [--data-dir <dir>] [--format gguf|onnx|mlx]");
                    eprintln!("  vidarax-cli distill deploy  --tenant-id <id> [--data-dir <dir>]");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  vidarax-cli run-create");
            eprintln!("  vidarax-cli health");
            eprintln!("  vidarax-cli models");
            eprintln!("  vidarax-cli states");
            eprintln!("  vidarax-cli distill <status|train|export|deploy> ...");
        }
    }
}

// ─── Options ──────────────────────────────────────────────────────────────────

struct DistillOpts {
    tenant_id: Option<String>,
    data_dir: PathBuf,
    mlx: bool,
    resume: bool,
    format: String,
}

impl DistillOpts {
    fn parse(args: impl Iterator<Item = String>) -> Self {
        let mut opts = Self {
            tenant_id: None,
            data_dir: PathBuf::from(".vidarax-data"),
            mlx: false,
            resume: false,
            format: "gguf".to_string(),
        };
        let args: Vec<String> = args.collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--tenant-id" if i + 1 < args.len() => {
                    opts.tenant_id = Some(args[i + 1].clone());
                    i += 2;
                }
                "--data-dir" if i + 1 < args.len() => {
                    opts.data_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                }
                "--mlx" => {
                    opts.mlx = true;
                    i += 1;
                }
                "--resume" => {
                    opts.resume = true;
                    i += 1;
                }
                "--format" if i + 1 < args.len() => {
                    opts.format = args[i + 1].clone();
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }
        opts
    }

    fn require_tenant_id(&self) -> &str {
        self.tenant_id.as_deref().unwrap_or_else(|| {
            eprintln!("error: --tenant-id is required");
            std::process::exit(1);
        })
    }
}

// ─── distill status ───────────────────────────────────────────────────────────

/// Show the number of stored training pairs and database location.
async fn cmd_distill_status(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    match vidarax_core::training_data::TrainingStore::new(&opts.data_dir) {
        Ok(store) => {
            let count = store.pair_count(tenant_id).unwrap_or(0);
            println!("tenant_id : {tenant_id}");
            println!("pairs     : {count}");
            println!("data_dir  : {}", opts.data_dir.display());
        }
        Err(e) => {
            eprintln!("error opening training store: {e}");
            std::process::exit(1);
        }
    }
}

// ─── distill train ────────────────────────────────────────────────────────────

/// Spawn the Python training script and stream its output.
async fn cmd_distill_train(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    // Export a fresh JSONL snapshot before kicking off training.
    let jsonl_path = opts.data_dir.join(format!("{tenant_id}-training.jsonl"));
    match vidarax_core::training_data::TrainingStore::new(&opts.data_dir) {
        Ok(store) => {
            match store.export_training_jsonl(tenant_id, &jsonl_path) {
                Ok(n) => tracing::info!(count = n, path = %jsonl_path.display(), "exported training pairs"),
                Err(e) => {
                    eprintln!("error exporting training data: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("error opening training store: {e}");
            std::process::exit(1);
        }
    }

    let script = if opts.mlx {
        "scripts/train_specialist_mlx.py"
    } else {
        "scripts/train_specialist.py"
    };

    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(script)
        .arg("--tenant-id")
        .arg(tenant_id)
        .arg("--data-path")
        .arg(&jsonl_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if opts.resume {
        cmd.arg("--resume");
    }

    tracing::info!(script, tenant_id, "starting training subprocess");
    run_subprocess(cmd).await;
}

// ─── distill export ───────────────────────────────────────────────────────────

/// Spawn the Python export script to convert the fine-tuned model.
async fn cmd_distill_export(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    let format = opts.format.as_str();
    if !matches!(format, "gguf" | "onnx" | "mlx") {
        eprintln!("error: --format must be one of: gguf, onnx, mlx");
        std::process::exit(1);
    }

    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg("scripts/export_specialist.py")
        .arg("--tenant-id")
        .arg(tenant_id)
        .arg("--format")
        .arg(format)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    tracing::info!(tenant_id, format, "starting export subprocess");
    run_subprocess(cmd).await;
}

// ─── distill deploy ───────────────────────────────────────────────────────────

/// Write a `.reload-specialist` sentinel file that the running API polls to
/// trigger a hot-reload of the specialist model weights.
async fn cmd_distill_deploy(opts: DistillOpts) {
    let tenant_id = opts.require_tenant_id();

    let sentinel = opts.data_dir.join(".reload-specialist");
    match std::fs::write(&sentinel, tenant_id) {
        Ok(()) => {
            tracing::info!(
                path = %sentinel.display(),
                tenant_id,
                "reload sentinel written — running API will pick this up"
            );
            println!("sentinel written: {}", sentinel.display());
        }
        Err(e) => {
            eprintln!("error writing sentinel: {e}");
            std::process::exit(1);
        }
    }
}

// ─── Subprocess runner ────────────────────────────────────────────────────────

/// Spawn the command, stream its stdout/stderr via tracing, then wait for exit.
async fn run_subprocess(mut cmd: tokio::process::Command) {
    use tokio::io::AsyncBufReadExt as _;

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn subprocess: {e}");
            std::process::exit(1);
        }
    };

    // Stream stdout
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("[subprocess] {}", line);
            }
        });
    }

    // Stream stderr
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::warn!("[subprocess] {}", line);
            }
        });
    }

    match child.wait().await {
        Ok(status) if status.success() => {
            tracing::info!("subprocess exited successfully");
        }
        Ok(status) => {
            eprintln!("subprocess exited with: {status}");
            std::process::exit(status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("subprocess wait error: {e}");
            std::process::exit(1);
        }
    }
}
