use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde::Serialize;
use tower::ServiceExt;
use vidarax_api::{app_router, AppState};

#[derive(Debug, Serialize)]
struct LatencyStats {
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Serialize)]
struct ApiPathBenchResult {
    iterations: usize,
    warmup: usize,
    elapsed_sec: f64,
    workflows_per_sec: f64,
    create_run: LatencyStats,
    ingest_run: LatencyStats,
    analyze_run: LatencyStats,
    query_run: LatencyStats,
    workflow_total: LatencyStats,
}

#[tokio::main]
async fn main() {
    let iterations = std::env::var("VIDARAX_API_BENCH_ITERATIONS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(200)
        .clamp(10, 20_000);
    let warmup = std::env::var("VIDARAX_API_BENCH_WARMUP")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(20)
        .clamp(0, 5_000);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let wal_path = std::env::temp_dir().join(format!("vidarax-api-bench-{nanos}.wal"));
    let state = AppState::with_wal_for_tests(wal_path);
    let app = app_router(state);

    let mut create_us = Vec::with_capacity(iterations);
    let mut ingest_us = Vec::with_capacity(iterations);
    let mut analyze_us = Vec::with_capacity(iterations);
    let mut query_us = Vec::with_capacity(iterations);
    let mut workflow_us = Vec::with_capacity(iterations);

    let total_start = Instant::now();
    for i in 0..(warmup + iterations) {
        let workflow_start = Instant::now();

        let create_start = Instant::now();
        let create_req = Request::builder()
            .uri("/v1/runs")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"balanced"}"#))
            .expect("valid create request");
        let (create_status, create_body) = execute(app.clone(), create_req).await;
        ensure_status("create", create_status, &create_body);
        let run_id = serde_json::from_slice::<serde_json::Value>(&create_body)
            .ok()
            .and_then(|v| {
                v.get("run_id")
                    .and_then(|x| x.as_str())
                    .map(ToString::to_string)
            })
            .expect("create response must contain run_id");
        let create_elapsed = create_start.elapsed().as_micros() as f64;

        let ingest_start = Instant::now();
        let ingest_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/ingest"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"frame_index":1}"#))
            .expect("valid ingest request");
        let (ingest_status, ingest_body) = execute(app.clone(), ingest_req).await;
        ensure_status("ingest", ingest_status, &ingest_body);
        let ingest_elapsed = ingest_start.elapsed().as_micros() as f64;

        let analyze_start = Instant::now();
        let analyze_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/analyze"))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"Qwen/Qwen3-VL-4B-Instruct",
                    "frames":[
                        {"frame_index":0,"pts_ms":0,"perceptual_hash":1,"luma_mean":0.2,"flicker_score":0.0,"ghosting_score":0.0,"noise_variance_score":0.0},
                        {"frame_index":1,"pts_ms":33,"perceptual_hash":2,"luma_mean":0.3,"flicker_score":0.1,"ghosting_score":0.0,"noise_variance_score":0.0}
                    ]
                }"#,
            ))
            .expect("valid analyze request");
        let (analyze_status, analyze_body) = execute(app.clone(), analyze_req).await;
        ensure_status("analyze", analyze_status, &analyze_body);
        let analyze_elapsed = analyze_start.elapsed().as_micros() as f64;

        let query_start = Instant::now();
        let query_req = Request::builder()
            .uri("/v1/query")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(format!(r#"{{"run_id":"{run_id}"}}"#)))
            .expect("valid query request");
        let (query_status, query_body) = execute(app.clone(), query_req).await;
        ensure_status("query", query_status, &query_body);
        let query_elapsed = query_start.elapsed().as_micros() as f64;

        let stop_req = Request::builder()
            .uri(format!("/v1/runs/{run_id}/stop"))
            .method("POST")
            .body(Body::empty())
            .expect("valid stop request");
        let (stop_status, stop_body) = execute(app.clone(), stop_req).await;
        ensure_status("stop", stop_status, &stop_body);

        let workflow_elapsed = workflow_start.elapsed().as_micros() as f64;

        if i >= warmup {
            create_us.push(create_elapsed);
            ingest_us.push(ingest_elapsed);
            analyze_us.push(analyze_elapsed);
            query_us.push(query_elapsed);
            workflow_us.push(workflow_elapsed);
        }
    }
    let elapsed_sec = total_start.elapsed().as_secs_f64();

    let result = ApiPathBenchResult {
        iterations,
        warmup,
        elapsed_sec,
        workflows_per_sec: iterations as f64 / elapsed_sec.max(0.000_001),
        create_run: summarize_ms(create_us),
        ingest_run: summarize_ms(ingest_us),
        analyze_run: summarize_ms(analyze_us),
        query_run: summarize_ms(query_us),
        workflow_total: summarize_ms(workflow_us),
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&result).expect("bench result should serialize")
    );
}

async fn execute(app: axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = app.oneshot(req).await.expect("request should execute");
    let status = resp.status();
    let body = resp
        .into_body()
        .collect()
        .await
        .expect("response body should collect")
        .to_bytes()
        .to_vec();
    (status, body)
}

fn ensure_status(step: &str, status: StatusCode, body: &[u8]) {
    if status != StatusCode::OK {
        let body = std::str::from_utf8(body).unwrap_or("<non-utf8>");
        panic!("{step} failed with {status}: {body}");
    }
}

fn summarize_ms(mut samples_us: Vec<f64>) -> LatencyStats {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max_ms = samples_us.last().copied().unwrap_or(0.0) / 1000.0;
    LatencyStats {
        p50_ms: percentile(&samples_us, 0.50) / 1000.0,
        p95_ms: percentile(&samples_us, 0.95) / 1000.0,
        p99_ms: percentile(&samples_us, 0.99) / 1000.0,
        max_ms,
    }
}

fn percentile(samples_us: &[f64], p: f64) -> f64 {
    if samples_us.is_empty() {
        return 0.0;
    }
    let idx = ((samples_us.len() - 1) as f64 * p).round() as usize;
    samples_us[idx.min(samples_us.len() - 1)]
}
