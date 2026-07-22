use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::Semaphore;
use vidarax_core::provider::{
    HttpTransport, InferenceProvider, InferenceRequest, OpenAiCompatProvider, ProviderKind,
};

#[derive(Debug, Serialize)]
struct BenchStats {
    throughput_rps: f64,
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Debug, Serialize)]
struct BenchResult {
    blocking_spawn: BenchStats,
    async_reqwest: BenchStats,
    recommendation: &'static str,
}

#[tokio::main]
async fn main() {
    let server = spawn_mock_server();
    let base_url = format!("http://{}", server.addr);
    let request = InferenceRequest {
        model: Arc::from("Qwen/Qwen3-VL-2B-Instruct"),
        prompt: Arc::from("benchmark"),
        input_images: Vec::new(),
        input_videos: Vec::new(),
        max_tokens: 16,
        temperature: 0.0,
        timeout_ms: 10_000,
        allow_fallback: true,
        guided_json: None,
        scheduling: Default::default(),
    };

    let requests = 200usize;
    let concurrency = 16usize;
    let blocking =
        bench_blocking_spawn(requests, concurrency, base_url.clone(), request.clone()).await;
    let async_stats = bench_async_reqwest(requests, concurrency, base_url.clone()).await;

    let recommendation = if async_stats.throughput_rps > (blocking.throughput_rps * 1.15)
        && async_stats.p95_ms < (blocking.p95_ms * 0.9)
    {
        "migrate_to_async_transport"
    } else {
        "keep_blocking_spawn_path_for_now"
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&BenchResult {
            blocking_spawn: blocking,
            async_reqwest: async_stats,
            recommendation,
        })
        .unwrap()
    );
    drop(server);
}

async fn bench_blocking_spawn(
    requests: usize,
    concurrency: usize,
    base_url: String,
    request: InferenceRequest,
) -> BenchStats {
    // Build the provider once so all tasks share the same connection pool.
    let transport = HttpTransport::new(&base_url).unwrap();
    let router = Arc::new(OpenAiCompatProvider::new(transport, ProviderKind::Vllm));
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut tasks = Vec::with_capacity(requests);
    let start = Instant::now();

    for _ in 0..requests {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let router = Arc::clone(&router);
        let request = request.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            let _ = tokio::task::spawn_blocking(move || router.infer(&request))
                .await
                .unwrap()
                .unwrap();
            started.elapsed().as_secs_f64() * 1000.0
        }));
    }

    let mut samples = Vec::with_capacity(requests);
    for task in tasks {
        samples.push(task.await.unwrap());
    }
    summarize(samples, start.elapsed().as_secs_f64())
}

async fn bench_async_reqwest(requests: usize, concurrency: usize, base_url: String) -> BenchStats {
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let client = Client::builder()
        .pool_max_idle_per_host(16)
        .build()
        .unwrap();
    let payload = serde_json::json!({
        "model":"Qwen/Qwen3-VL-2B-Instruct",
        "messages":[{"role":"user","content":"benchmark"}],
        "max_tokens": 16,
        "temperature": 0.0
    });

    let mut tasks = Vec::with_capacity(requests);
    let start = Instant::now();
    for _ in 0..requests {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let base_url = base_url.clone();
        let payload = payload.clone();
        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            let response = client
                .post(format!("{base_url}/v1/chat/completions"))
                .header("content-type", "application/json")
                .json(&payload)
                .send()
                .await
                .unwrap();
            let text = response.text().await.unwrap();
            let _: Value = serde_json::from_str(&text).unwrap();
            started.elapsed().as_secs_f64() * 1000.0
        }));
    }

    let mut samples = Vec::with_capacity(requests);
    for task in tasks {
        samples.push(task.await.unwrap());
    }
    summarize(samples, start.elapsed().as_secs_f64())
}

fn summarize(mut samples_ms: Vec<f64>, elapsed_secs: f64) -> BenchStats {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    BenchStats {
        throughput_rps: (samples_ms.len() as f64) / elapsed_secs.max(0.000_001),
        p50_ms: percentile(&samples_ms, 50),
        p95_ms: percentile(&samples_ms, 95),
    }
}

fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * pct) / 100;
    sorted[idx]
}

struct MockServer {
    addr: String,
    _thread: thread::JoinHandle<()>,
}

fn spawn_mock_server() -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let thread = thread::spawn(move || loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
                let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(_) => break,
        }
    });

    MockServer {
        addr,
        _thread: thread,
    }
}
