//! Stateless trigger compilation, validation, and deterministic replay.

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;
use vidarax_contracts::triggers::{
    compile_trigger as compile_source, TriggerCompileRequest, TriggerEvaluateRequest,
    TriggerProgram,
};
use vidarax_core::trigger::TriggerVm;

use crate::models::FieldError;
use crate::response::{ok, validation_error, ApiResponse};
use crate::state::AppState;

const MAX_EVALUATION_SAMPLES: usize = 10_000;

pub(crate) async fn compile_trigger(
    State(state): State<AppState>,
    Json(request): Json<TriggerCompileRequest>,
) -> impl IntoResponse {
    if request.source.len() > 64 * 1024 {
        return invalid(&state, "source", "trigger source exceeds 64 KiB");
    }
    let program = match compile_source(&request.source) {
        Ok(program) => program,
        Err(error) => return invalid(&state, "source", error.to_string()),
    };
    ok(json!({
        "request_id": state.next_request_id(),
        "instruction_count": program.instructions.len(),
        "state_slots": program.state_slots(),
        "program": program,
    }))
}

pub(crate) async fn validate_trigger(
    State(state): State<AppState>,
    Json(program): Json<TriggerProgram>,
) -> impl IntoResponse {
    if let Err(error) = program.validate() {
        return invalid(&state, "program", error.to_string());
    }
    ok(json!({
        "request_id": state.next_request_id(),
        "valid": true,
        "isa_version": program.isa_version,
        "program_id": program.program_id,
        "program_version": program.version,
        "instruction_count": program.instructions.len(),
        "state_slots": program.state_slots(),
    }))
}

pub(crate) async fn evaluate_trigger(
    State(state): State<AppState>,
    Json(request): Json<TriggerEvaluateRequest>,
) -> impl IntoResponse {
    if request.samples.is_empty() || request.samples.len() > MAX_EVALUATION_SAMPLES {
        return invalid(
            &state,
            "samples",
            format!("sample count must be in [1, {MAX_EVALUATION_SAMPLES}]"),
        );
    }
    if request
        .samples
        .windows(2)
        .any(|pair| pair[1].pts_ms < pair[0].pts_ms)
    {
        return invalid(&state, "samples", "sample timestamps must be monotonic");
    }
    let program = Arc::new(request.program);
    let mut vm = match TriggerVm::try_new(Arc::clone(&program)) {
        Ok(vm) => vm,
        Err(error) => return invalid(&state, "program", error),
    };
    let results = request
        .samples
        .iter()
        .map(|sample| {
            let evaluation = vm.evaluate(sample);
            let actions = evaluation
                .actions(&program)
                .map(|action| serde_json::to_value(action).expect("trigger action serializes"))
                .collect::<Vec<_>>();
            json!({
                "pts_ms": sample.pts_ms,
                "fired": evaluation.fired(),
                "missing_signal": evaluation.missing_signal,
                "actions": actions,
            })
        })
        .collect::<Vec<_>>();
    ok(json!({
        "request_id": state.next_request_id(),
        "program_id": program.program_id,
        "program_version": program.version,
        "results": results,
    }))
}

fn invalid(state: &AppState, field: &'static str, message: impl Into<String>) -> ApiResponse {
    validation_error(
        state,
        "invalid trigger program",
        vec![FieldError {
            field,
            message: message.into(),
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn state() -> AppState {
        AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-trigger-api-{}-{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        )))
    }

    #[tokio::test]
    async fn compile_validate_and_replay_trigger_through_http() {
        let app = crate::app_router(state());
        let source = "trigger zone version 1\nwhen motion_score >= 0.4 for 2 frames\nedge rising\nemit zone_entered\ncapture keyframe\nnotify webhook\nend";
        let response = app
            .clone()
            .oneshot(
                Request::post("/v1/triggers/compile")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"source": source}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        let program = body["program"].clone();

        let validated = app
            .clone()
            .oneshot(
                Request::post("/v1/triggers/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(program.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(validated.status(), StatusCode::OK);

        let evaluated = app
            .oneshot(
                Request::post("/v1/triggers/evaluate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "program": program,
                            "samples": [
                                {"pts_ms": 0, "motion_score": 0.8},
                                {"pts_ms": 100, "motion_score": 0.8},
                                {"pts_ms": 200, "motion_score": 0.8}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(evaluated.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&evaluated.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["results"][0]["fired"], false);
        assert_eq!(body["results"][1]["fired"], true);
        assert_eq!(body["results"][2]["fired"], false);
        assert_eq!(body["results"][1]["actions"].as_array().unwrap().len(), 3);
    }
}
