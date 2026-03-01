use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use crate::models::FieldError;
use crate::state::AppState;

pub type ApiResponse = (StatusCode, Json<Value>);

pub fn ok(value: Value) -> ApiResponse {
    (StatusCode::OK, Json(value))
}

pub fn validation_error(
    state: &AppState,
    message: impl Into<String>,
    details: Vec<FieldError>,
) -> ApiResponse {
    structured_error(
        state.next_request_id(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "validation_error",
        message,
        details,
    )
}

pub fn not_found_error(
    state: &AppState,
    message: impl Into<String>,
    details: Vec<FieldError>,
) -> ApiResponse {
    structured_error(
        state.next_request_id(),
        StatusCode::NOT_FOUND,
        "not_found",
        message,
        details,
    )
}

pub fn conflict_error(
    state: &AppState,
    message: impl Into<String>,
    details: Vec<FieldError>,
) -> ApiResponse {
    structured_error(
        state.next_request_id(),
        StatusCode::CONFLICT,
        "conflict",
        message,
        details,
    )
}

pub fn internal_error(state: &AppState, message: impl Into<String>) -> ApiResponse {
    let message = message.into();
    let request_id = state.next_request_id();
    tracing::error!(request_id, message, "vidarax-api internal error");
    structured_error(
        request_id,
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "internal server error",
        Vec::new(),
    )
}

fn structured_error(
    request_id: String,
    status: StatusCode,
    code: &'static str,
    message: impl Into<String>,
    details: Vec<FieldError>,
) -> ApiResponse {
    (
        status,
        Json(json!({
            "error": {
                "code": code,
                "message": message.into(),
                "request_id": request_id,
                "details": details
            }
        })),
    )
}
