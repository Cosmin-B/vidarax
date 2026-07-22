//! WAL-native policy revisions and deployment state.
//!
//! Policy state is reconstructed from immutable timeline events. The live
//! media path never reads this module and therefore pays no allocation or
//! synchronization cost for control-plane operations.

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use vidarax_core::timeline::TimelineEvent;

use crate::ids::validate_run_id;
use crate::models::FieldError;
use crate::response::{
    conflict_error, internal_error, not_found_error, ok, service_unavailable, validation_error,
    ApiResponse,
};
use crate::state::AppState;

const POLICY_CREATED: &str = "policy_revision_created";
const POLICY_DEPLOY_REQUESTED: &str = "policy_deployment_requested";
const POLICY_DEPLOY_ACKNOWLEDGED: &str = "policy_deployment_acknowledged";
const POLICY_DEPLOY_REJECTED: &str = "policy_deployment_rejected";
const POLICY_ROLLBACK_REQUESTED: &str = "policy_rollback_requested";
const POLICY_ROLLBACK_ACKNOWLEDGED: &str = "policy_rollback_acknowledged";
const POLICY_ROLLBACK_REJECTED: &str = "policy_rollback_rejected";
const POLICY_REPLAY_EVALUATED: &str = "policy_replay_evaluated";
const RESTRICTED_ZONE_EVENT: &str = "restricted_zone_activity_entered";
const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_SCHEMA_BYTES: usize = 64 * 1024;
const MAX_ID_BYTES: usize = 128;
const COMMAND_ACK_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NormalizedRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

/// Wire-compatible with `vidarax_core::zone::RestrictedZonePolicy`. The API
/// crate keeps the request type local so it can validate and persist policy
/// revisions without coupling its public contract to the worker implementation.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RestrictedZoneParameters {
    policy_id: String,
    policy_version: u64,
    device_id: String,
    region: NormalizedRect,
    enter_motion_score: f32,
    exit_motion_score: f32,
    enter_after_frames: u16,
    exit_after_frames: u16,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PolicyParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    restricted_zone: Option<RestrictedZoneParameters>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreatePolicyRequest {
    parent_revision: Option<u64>,
    prompt: Option<String>,
    output_schema: Option<Value>,
    parameters: Option<PolicyParameters>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PolicyStatus {
    Draft,
    Shadow,
    Canary,
    Active,
    Retired,
    RolledBack,
}

impl PolicyStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Shadow => "shadow",
            Self::Canary => "canary",
            Self::Active => "active",
            Self::Retired => "retired",
            Self::RolledBack => "rolled_back",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivatePolicyRequest {
    stage: PolicyStatus,
    expected_generation: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RollbackPolicyRequest {
    expected_generation: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReplayPolicyRequest {
    from_seq: Option<u64>,
    to_seq: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyRevision {
    revision: u64,
    parent_revision: Option<u64>,
    status: PolicyStatus,
    prompt: Option<String>,
    output_schema: Option<Value>,
    parameters: PolicyParameters,
    created_at_ms: u64,
    updated_at_ms: u64,
    effective_generation: Option<u64>,
    effective_on_current_generation: bool,
    deferred_fields: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LiveApplication {
    session_id: Option<String>,
    generation: Option<u64>,
    prompt_acknowledged: bool,
    effective_on_current_generation: bool,
    deferred_fields: Vec<String>,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct ReplayEvaluation {
    revision: u64,
    source: &'static str,
    comparison: &'static str,
    from_seq: u64,
    to_seq: Option<u64>,
    candidate_events: usize,
    accepted_events: usize,
    rejected_events: usize,
    events_without_score: usize,
    threshold: Option<f32>,
    limitation: &'static str,
}

#[tracing::instrument(name = "api.create_policy", skip_all, fields(run_id))]
pub(crate) async fn create_policy(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Json(mut request): Json<CreatePolicyRequest>,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    if let Some(error) = validate_policy_request(&state, &request) {
        return error;
    }

    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read policy events: {err}")),
    };
    let revisions = reconstruct_policies(&events);
    let current_revision = revisions.last().map(|policy| policy.revision);
    if request.parent_revision != current_revision {
        return conflict_error(
            &state,
            "policy parent is stale",
            vec![field_error(
                "parent_revision",
                format!(
                    "expected {}, received {}",
                    display_revision(current_revision),
                    display_revision(request.parent_revision)
                ),
            )],
        );
    }

    if let Some(prompt) = request.prompt.as_mut() {
        *prompt = prompt.trim().to_string();
    }
    let event = match state
        .append_run_event_async(
            &run_id,
            POLICY_CREATED,
            json!({
                "parent_revision": request.parent_revision,
                "prompt": request.prompt,
                "output_schema": request.output_schema,
                "parameters": request.parameters.unwrap_or_default(),
            }),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => {
            return internal_error(&state, format!("failed to append policy revision: {err}"))
        }
    };

    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => {
            return internal_error(&state, format!("failed to reload policy revision: {err}"))
        }
    };
    let Some(policy) = reconstruct_policies(&events)
        .into_iter()
        .find(|policy| policy.revision == event.seq)
    else {
        return internal_error(
            &state,
            "created policy revision was not readable from the WAL",
        );
    };

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "policy": policy,
    }))
}

#[tracing::instrument(name = "api.list_policies", skip_all, fields(run_id))]
pub(crate) async fn list_policies(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read policies: {err}")),
    };
    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "policies": reconstruct_policies(&events),
    }))
}

#[tracing::instrument(name = "api.get_policy", skip_all, fields(run_id, revision))]
pub(crate) async fn get_policy(
    State(state): State<AppState>,
    Path((run_id, revision)): Path<(String, u64)>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read policy: {err}")),
    };
    let Some(policy) = reconstruct_policies(&events)
        .into_iter()
        .find(|policy| policy.revision == revision)
    else {
        return policy_not_found(&state, revision);
    };
    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "policy": policy,
    }))
}

#[tracing::instrument(name = "api.activate_policy", skip_all, fields(run_id, revision))]
pub(crate) async fn activate_policy(
    State(state): State<AppState>,
    Path((run_id, revision)): Path<(String, u64)>,
    headers: HeaderMap,
    Json(request): Json<ActivatePolicyRequest>,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    if !matches!(
        request.stage,
        PolicyStatus::Shadow | PolicyStatus::Canary | PolicyStatus::Active
    ) {
        return validation_error(
            &state,
            "invalid policy deployment stage",
            vec![field_error(
                "stage",
                "stage must be one of: shadow, canary, active".to_string(),
            )],
        );
    }

    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read policy: {err}")),
    };
    let revisions = reconstruct_policies(&events);
    let Some(policy) = revisions
        .iter()
        .find(|policy| policy.revision == revision)
        .cloned()
    else {
        return policy_not_found(&state, revision);
    };
    if !valid_promotion(policy.status, request.stage) {
        return conflict_error(
            &state,
            "policy stage transition is not allowed",
            vec![field_error(
                "stage",
                format!(
                    "cannot move revision {revision} from {} to {}",
                    policy.status.as_str(),
                    request.stage.as_str()
                ),
            )],
        );
    }

    let request_event = match state
        .append_run_event_async(
            &run_id,
            POLICY_DEPLOY_REQUESTED,
            json!({
                "revision": revision,
                "from_status": policy.status,
                "to_status": request.stage,
                "expected_generation": request.expected_generation,
            }),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => {
            return internal_error(
                &state,
                format!("failed to append deployment request: {err}"),
            )
        }
    };

    let application = match apply_policy(
        &state,
        &run_id,
        &policy,
        request.stage == PolicyStatus::Active,
        request.expected_generation,
    )
    .await
    {
        Ok(application) => application,
        Err(error) => {
            let _ = state
                .append_run_event_async(
                    &run_id,
                    POLICY_DEPLOY_REJECTED,
                    json!({
                        "revision": revision,
                        "request_seq": request_event.seq,
                        "to_status": request.stage,
                        "reason": error.reason(),
                    }),
                )
                .await;
            return error.into_response(&state);
        }
    };

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            POLICY_DEPLOY_ACKNOWLEDGED,
            json!({
                "revision": revision,
                "request_seq": request_event.seq,
                "status": request.stage,
                "application": application,
            }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append deployment acknowledgement: {err}"),
        );
    }

    policy_response_after_transition(&state, &run_id, revision, application).await
}

#[tracing::instrument(name = "api.rollback_policy", skip_all, fields(run_id, revision))]
pub(crate) async fn rollback_policy(
    State(state): State<AppState>,
    Path((run_id, revision)): Path<(String, u64)>,
    headers: HeaderMap,
    Json(request): Json<RollbackPolicyRequest>,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read policies: {err}")),
    };
    let policies = reconstruct_policies(&events);
    let Some(target) = policies
        .iter()
        .find(|policy| policy.revision == revision)
        .cloned()
    else {
        return policy_not_found(&state, revision);
    };
    let Some(current) = policies
        .iter()
        .find(|policy| policy.status == PolicyStatus::Active)
        .cloned()
    else {
        return conflict_error(
            &state,
            "no active policy exists to roll back",
            vec![field_error(
                "revision",
                "activate a policy before requesting rollback".to_string(),
            )],
        );
    };
    if current.revision == revision {
        return conflict_error(
            &state,
            "rollback target is already active",
            vec![field_error(
                "revision",
                format!("revision {revision} is already active"),
            )],
        );
    }
    if !matches!(
        target.status,
        PolicyStatus::Retired | PolicyStatus::RolledBack
    ) {
        return conflict_error(
            &state,
            "rollback target was never an active policy",
            vec![field_error(
                "revision",
                format!("revision {revision} has status {}", target.status.as_str()),
            )],
        );
    }

    let request_event = match state
        .append_run_event_async(
            &run_id,
            POLICY_ROLLBACK_REQUESTED,
            json!({
                "revision": revision,
                "from_revision": current.revision,
                "expected_generation": request.expected_generation,
            }),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => {
            return internal_error(&state, format!("failed to append rollback request: {err}"))
        }
    };

    let application =
        match apply_policy(&state, &run_id, &target, true, request.expected_generation).await {
            Ok(application) => application,
            Err(error) => {
                let _ = state
                    .append_run_event_async(
                        &run_id,
                        POLICY_ROLLBACK_REJECTED,
                        json!({
                            "revision": revision,
                            "from_revision": current.revision,
                            "request_seq": request_event.seq,
                            "reason": error.reason(),
                        }),
                    )
                    .await;
                return error.into_response(&state);
            }
        };

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            POLICY_ROLLBACK_ACKNOWLEDGED,
            json!({
                "revision": revision,
                "from_revision": current.revision,
                "request_seq": request_event.seq,
                "status": PolicyStatus::Active,
                "application": application,
            }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append rollback acknowledgement: {err}"),
        );
    }

    policy_response_after_transition(&state, &run_id, revision, application).await
}

#[tracing::instrument(name = "api.replay_policy", skip_all, fields(run_id, revision))]
pub(crate) async fn replay_policy(
    State(state): State<AppState>,
    Path((run_id, revision)): Path<(String, u64)>,
    headers: HeaderMap,
    Json(request): Json<ReplayPolicyRequest>,
) -> impl IntoResponse {
    if let Err(error) = authorize_run(&state, &headers, &run_id) {
        return error;
    }
    if request
        .to_seq
        .is_some_and(|to_seq| to_seq < request.from_seq.unwrap_or(0))
    {
        return validation_error(
            &state,
            "invalid replay range",
            vec![field_error(
                "to_seq",
                "to_seq must be greater than or equal to from_seq".to_string(),
            )],
        );
    }
    let events = match state.read_run_events_async(&run_id).await {
        Ok(events) => events,
        Err(err) => {
            return internal_error(&state, format!("failed to read replay evidence: {err}"))
        }
    };
    let Some(policy) = reconstruct_policies(&events)
        .into_iter()
        .find(|policy| policy.revision == revision)
    else {
        return policy_not_found(&state, revision);
    };
    let evaluation = evaluate_replay(&policy, &events, &request);
    let event = match state
        .append_run_event_async(
            &run_id,
            POLICY_REPLAY_EVALUATED,
            serde_json::to_value(&evaluation).unwrap_or_else(|_| json!({ "revision": revision })),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => {
            return internal_error(&state, format!("failed to append replay evaluation: {err}"))
        }
    };
    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "evaluation_id": event.seq,
        "evaluation": evaluation,
    }))
}

async fn policy_response_after_transition(
    state: &AppState,
    run_id: &str,
    revision: u64,
    application: LiveApplication,
) -> ApiResponse {
    let events = match state.read_run_events_async(run_id).await {
        Ok(events) => events,
        Err(err) => return internal_error(state, format!("failed to reload policy state: {err}")),
    };
    let Some(policy) = reconstruct_policies(&events)
        .into_iter()
        .find(|policy| policy.revision == revision)
    else {
        return internal_error(
            state,
            "acknowledged policy revision was not readable from the WAL",
        );
    };
    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "policy": policy,
        "application": application,
    }))
}

#[derive(Debug)]
enum DeploymentError {
    ExpectedGenerationRequired,
    StaleGeneration { expected: u64, current: u64 },
    GenerationClosed(String),
    AcknowledgementTimeout,
}

impl DeploymentError {
    fn reason(&self) -> String {
        match self {
            Self::ExpectedGenerationRequired => "expected_generation_required".to_string(),
            Self::StaleGeneration { expected, current } => {
                format!("stale_generation: expected {expected}, current {current}")
            }
            Self::GenerationClosed(message) => format!("generation_closed: {message}"),
            Self::AcknowledgementTimeout => "acknowledgement_timeout".to_string(),
        }
    }

    fn into_response(self, state: &AppState) -> ApiResponse {
        match self {
            Self::ExpectedGenerationRequired => conflict_error(
                state,
                "live policy activation requires expected_generation",
                vec![field_error(
                    "expected_generation",
                    "read the live generation and retry with that exact value".to_string(),
                )],
            ),
            Self::StaleGeneration { expected, current } => conflict_error(
                state,
                "policy command targeted a stale generation",
                vec![field_error(
                    "expected_generation",
                    format!("expected generation {expected}, current generation is {current}"),
                )],
            ),
            Self::GenerationClosed(message) => conflict_error(
                state,
                "pipeline generation rejected policy command",
                vec![field_error("expected_generation", message)],
            ),
            Self::AcknowledgementTimeout => service_unavailable(
                state,
                "policy_acknowledgement_timeout",
                "the live generation did not acknowledge the policy command within two seconds",
            ),
        }
    }
}

async fn apply_policy(
    state: &AppState,
    run_id: &str,
    policy: &PolicyRevision,
    apply_live: bool,
    expected_generation: Option<u64>,
) -> Result<LiveApplication, DeploymentError> {
    let Some((session_id, session)) = state.live_session_for_run(run_id) else {
        return Ok(LiveApplication {
            session_id: None,
            generation: None,
            prompt_acknowledged: false,
            effective_on_current_generation: false,
            deferred_fields: deferred_fields(policy),
            note: "stored durably; applies when a new pipeline generation starts",
        });
    };
    let generation = session.generation().get();
    if !apply_live {
        return Ok(LiveApplication {
            session_id: Some(session_id),
            generation: Some(generation),
            prompt_acknowledged: false,
            effective_on_current_generation: false,
            deferred_fields: deferred_fields(policy),
            note: "shadow and canary stages are replay-only in this release",
        });
    }
    let expected = expected_generation.ok_or(DeploymentError::ExpectedGenerationRequired)?;
    if expected != generation {
        return Err(DeploymentError::StaleGeneration {
            expected,
            current: generation,
        });
    }

    let mut prompt_acknowledged = false;
    if let Some(prompt) = &policy.prompt {
        let update = session.update_config(
            prompt.clone(),
            policy.output_schema.as_ref().map(Value::to_string),
        );
        match tokio::time::timeout(COMMAND_ACK_TIMEOUT, update).await {
            Ok(Ok(())) => prompt_acknowledged = true,
            Ok(Err(err)) => return Err(DeploymentError::GenerationClosed(err.to_string())),
            Err(_) => return Err(DeploymentError::AcknowledgementTimeout),
        }
    }

    let deferred_fields = deferred_fields(policy);
    let effective_on_current_generation = prompt_acknowledged && deferred_fields.is_empty();
    Ok(LiveApplication {
        session_id: Some(session_id),
        generation: Some(generation),
        prompt_acknowledged,
        effective_on_current_generation,
        deferred_fields,
        note: if effective_on_current_generation {
            "live generation acknowledged the complete policy"
        } else {
            "prompt was acknowledged; static detector parameters require a new generation"
        },
    })
}

fn deferred_fields(policy: &PolicyRevision) -> Vec<String> {
    policy
        .parameters
        .restricted_zone
        .as_ref()
        .map(|_| vec!["parameters.restricted_zone".to_string()])
        .unwrap_or_default()
}

fn evaluate_replay(
    policy: &PolicyRevision,
    events: &[TimelineEvent],
    request: &ReplayPolicyRequest,
) -> ReplayEvaluation {
    let from_seq = request.from_seq.unwrap_or(0);
    let threshold = policy
        .parameters
        .restricted_zone
        .as_ref()
        .map(|parameters| parameters.enter_motion_score);
    let mut evaluation = ReplayEvaluation {
        revision: policy.revision,
        source: "local_wal",
        comparison: "persisted restricted_zone_activity_entered confidence against enter_motion_score",
        from_seq,
        to_seq: request.to_seq,
        candidate_events: 0,
        accepted_events: 0,
        rejected_events: 0,
        events_without_score: 0,
        threshold,
        limitation: "replay evaluates persisted candidates only; it cannot discover events that the original pipeline did not emit",
    };
    for event in events.iter().filter(|event| {
        event.kind == RESTRICTED_ZONE_EVENT
            && event.seq >= from_seq
            && request.to_seq.is_none_or(|to_seq| event.seq <= to_seq)
    }) {
        evaluation.candidate_events += 1;
        let payload = parse_payload(event);
        let score = payload
            .get("confidence")
            .and_then(Value::as_f64)
            .or_else(|| {
                payload
                    .pointer("/assertion/trigger/score")
                    .and_then(Value::as_f64)
            });
        match (score, threshold) {
            (Some(score), Some(threshold)) if score >= f64::from(threshold) => {
                evaluation.accepted_events += 1;
            }
            (Some(_), Some(_)) => evaluation.rejected_events += 1,
            _ => evaluation.events_without_score += 1,
        }
    }
    evaluation
}

fn reconstruct_policies(events: &[TimelineEvent]) -> Vec<PolicyRevision> {
    let mut policies = Vec::<PolicyRevision>::new();
    for event in events {
        let payload = parse_payload(event);
        match event.kind.as_str() {
            POLICY_CREATED => {
                let parameters = payload
                    .get("parameters")
                    .cloned()
                    .and_then(|value| serde_json::from_value(value).ok())
                    .unwrap_or_default();
                policies.push(PolicyRevision {
                    revision: event.seq,
                    parent_revision: payload.get("parent_revision").and_then(Value::as_u64),
                    status: PolicyStatus::Draft,
                    prompt: payload
                        .get("prompt")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    output_schema: payload
                        .get("output_schema")
                        .filter(|v| !v.is_null())
                        .cloned(),
                    parameters,
                    created_at_ms: event.pts_ms,
                    updated_at_ms: event.pts_ms,
                    effective_generation: None,
                    effective_on_current_generation: false,
                    deferred_fields: Vec::new(),
                });
            }
            POLICY_DEPLOY_ACKNOWLEDGED => {
                let Some(revision) = payload.get("revision").and_then(Value::as_u64) else {
                    continue;
                };
                let Some(status) = payload
                    .get("status")
                    .cloned()
                    .and_then(|value| serde_json::from_value(value).ok())
                else {
                    continue;
                };
                if status == PolicyStatus::Active {
                    retire_active_policy(&mut policies, revision, event.pts_ms);
                }
                if let Some(policy) = policies.iter_mut().find(|p| p.revision == revision) {
                    apply_acknowledgement(policy, status, &payload, event.pts_ms);
                }
            }
            POLICY_ROLLBACK_ACKNOWLEDGED => {
                let Some(revision) = payload.get("revision").and_then(Value::as_u64) else {
                    continue;
                };
                if let Some(from_revision) = payload.get("from_revision").and_then(Value::as_u64) {
                    if let Some(previous) = policies
                        .iter_mut()
                        .find(|policy| policy.revision == from_revision)
                    {
                        previous.status = PolicyStatus::RolledBack;
                        previous.updated_at_ms = event.pts_ms;
                        previous.effective_on_current_generation = false;
                    }
                }
                retire_active_policy(&mut policies, revision, event.pts_ms);
                if let Some(policy) = policies.iter_mut().find(|p| p.revision == revision) {
                    apply_acknowledgement(policy, PolicyStatus::Active, &payload, event.pts_ms);
                }
            }
            _ => {}
        }
    }
    policies.sort_by_key(|policy| policy.revision);
    policies
}

fn retire_active_policy(policies: &mut [PolicyRevision], except: u64, timestamp_ms: u64) {
    for policy in policies
        .iter_mut()
        .filter(|policy| policy.revision != except && policy.status == PolicyStatus::Active)
    {
        policy.status = PolicyStatus::Retired;
        policy.updated_at_ms = timestamp_ms;
        policy.effective_on_current_generation = false;
    }
}

fn apply_acknowledgement(
    policy: &mut PolicyRevision,
    status: PolicyStatus,
    payload: &Value,
    timestamp_ms: u64,
) {
    policy.status = status;
    policy.updated_at_ms = timestamp_ms;
    let application = payload.get("application").unwrap_or(&Value::Null);
    policy.effective_generation = application.get("generation").and_then(Value::as_u64);
    policy.effective_on_current_generation = application
        .get("effective_on_current_generation")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    policy.deferred_fields = application
        .get("deferred_fields")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect();
}

fn valid_promotion(from: PolicyStatus, to: PolicyStatus) -> bool {
    matches!(
        (from, to),
        (PolicyStatus::Draft, PolicyStatus::Shadow)
            | (PolicyStatus::Shadow, PolicyStatus::Canary)
            | (PolicyStatus::Canary, PolicyStatus::Active)
    )
}

fn validate_policy_request(state: &AppState, request: &CreatePolicyRequest) -> Option<ApiResponse> {
    let mut errors = Vec::new();
    if request.prompt.is_none()
        && request
            .parameters
            .as_ref()
            .and_then(|parameters| parameters.restricted_zone.as_ref())
            .is_none()
    {
        errors.push(field_error(
            "prompt",
            "a prompt or restricted-zone parameters are required".to_string(),
        ));
    }
    if let Some(prompt) = request.prompt.as_deref() {
        if prompt.trim().is_empty() {
            errors.push(field_error(
                "prompt",
                "prompt must not be empty".to_string(),
            ));
        } else if prompt.len() > MAX_PROMPT_BYTES {
            errors.push(field_error(
                "prompt",
                format!("prompt must be at most {MAX_PROMPT_BYTES} bytes"),
            ));
        }
    }
    if let Some(schema) = &request.output_schema {
        if !schema.is_object() {
            errors.push(field_error(
                "output_schema",
                "output_schema must be a JSON object".to_string(),
            ));
        } else if schema.to_string().len() > MAX_SCHEMA_BYTES {
            errors.push(field_error(
                "output_schema",
                format!("output_schema must be at most {MAX_SCHEMA_BYTES} bytes"),
            ));
        }
    }
    if let Some(zone) = request
        .parameters
        .as_ref()
        .and_then(|parameters| parameters.restricted_zone.as_ref())
    {
        validate_zone(zone, &mut errors);
    }
    (!errors.is_empty()).then(|| validation_error(state, "invalid policy revision", errors))
}

fn validate_zone(zone: &RestrictedZoneParameters, errors: &mut Vec<FieldError>) {
    for (field, value) in [
        (
            "parameters.restricted_zone.policy_id",
            zone.policy_id.as_str(),
        ),
        (
            "parameters.restricted_zone.device_id",
            zone.device_id.as_str(),
        ),
    ] {
        if value.trim().is_empty() || value.len() > MAX_ID_BYTES {
            errors.push(field_error(
                field,
                format!("must contain 1 to {MAX_ID_BYTES} bytes"),
            ));
        }
    }
    let rect = &zone.region;
    if ![rect.x, rect.y, rect.width, rect.height]
        .into_iter()
        .all(f32::is_finite)
        || rect.x < 0.0
        || rect.y < 0.0
        || rect.width <= 0.0
        || rect.height <= 0.0
        || rect.x + rect.width > 1.0
        || rect.y + rect.height > 1.0
    {
        errors.push(field_error(
            "parameters.restricted_zone.region",
            "region must be a finite, positive rectangle inside normalized image coordinates"
                .to_string(),
        ));
    }
    if !zone.enter_motion_score.is_finite()
        || !zone.exit_motion_score.is_finite()
        || !(0.0..=1.0).contains(&zone.enter_motion_score)
        || !(0.0..=1.0).contains(&zone.exit_motion_score)
        || zone.enter_motion_score <= zone.exit_motion_score
    {
        errors.push(field_error(
            "parameters.restricted_zone.enter_motion_score",
            "enter_motion_score must be in [0,1] and greater than exit_motion_score".to_string(),
        ));
    }
    for (field, value) in [
        (
            "parameters.restricted_zone.enter_after_frames",
            zone.enter_after_frames,
        ),
        (
            "parameters.restricted_zone.exit_after_frames",
            zone.exit_after_frames,
        ),
    ] {
        if !(1..=300).contains(&value) {
            errors.push(field_error(field, "must be between 1 and 300".to_string()));
        }
    }
}

fn authorize_run(state: &AppState, headers: &HeaderMap, run_id: &str) -> Result<(), ApiResponse> {
    if !validate_run_id(run_id) {
        return Err(validation_error(
            state,
            "invalid policy request",
            vec![field_error(
                "run_id",
                "run_id must match run-<16 or 32 hex chars>".to_string(),
            )],
        ));
    }
    let Some(snapshot) = state.run_runtime_snapshot(run_id, now_epoch_ms()) else {
        return Err(not_found_error(
            state,
            "run not found",
            vec![field_error("run_id", run_id.to_string())],
        ));
    };
    let principal = state.security_policy().principal_key_from_headers(headers);
    if snapshot.principal_key != principal {
        return Err(not_found_error(
            state,
            "run not found",
            vec![field_error("run_id", run_id.to_string())],
        ));
    }
    Ok(())
}

fn policy_not_found(state: &AppState, revision: u64) -> ApiResponse {
    not_found_error(
        state,
        "policy revision not found",
        vec![field_error("revision", revision.to_string())],
    )
}

fn field_error(field: &'static str, message: String) -> FieldError {
    FieldError { field, message }
}

fn parse_payload(event: &TimelineEvent) -> Value {
    serde_json::from_str(&event.payload).unwrap_or(Value::Null)
}

fn display_revision(revision: Option<u64>) -> String {
    revision.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vidarax_core::webrtc::runtime::PipelineGeneration;
    use vidarax_core::webrtc::session::WebRtcSession;

    fn event(seq: u64, kind: &str, payload: Value) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: "run-0000000000000001".to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms: seq * 10,
            kind: kind.to_string(),
            payload: payload.to_string(),
        }
    }

    fn zone_parameters(threshold: f32) -> PolicyParameters {
        PolicyParameters {
            restricted_zone: Some(RestrictedZoneParameters {
                policy_id: "loading-bay".to_string(),
                policy_version: 1,
                device_id: "camera-1".to_string(),
                region: NormalizedRect {
                    x: 0.1,
                    y: 0.2,
                    width: 0.3,
                    height: 0.4,
                },
                enter_motion_score: threshold,
                exit_motion_score: 0.2,
                enter_after_frames: 2,
                exit_after_frames: 3,
            }),
        }
    }

    #[test]
    fn reconstruction_retires_previous_active_revision() {
        let events = vec![
            event(
                1,
                POLICY_CREATED,
                json!({"parent_revision": null, "prompt": "one", "parameters": {}}),
            ),
            event(
                2,
                POLICY_DEPLOY_ACKNOWLEDGED,
                json!({"revision": 1, "status": "shadow", "application": {}}),
            ),
            event(
                3,
                POLICY_DEPLOY_ACKNOWLEDGED,
                json!({"revision": 1, "status": "canary", "application": {}}),
            ),
            event(
                4,
                POLICY_DEPLOY_ACKNOWLEDGED,
                json!({"revision": 1, "status": "active", "application": {}}),
            ),
            event(
                5,
                POLICY_CREATED,
                json!({"parent_revision": 1, "prompt": "two", "parameters": {}}),
            ),
            event(
                6,
                POLICY_DEPLOY_ACKNOWLEDGED,
                json!({"revision": 5, "status": "active", "application": {"generation": 9}}),
            ),
        ];
        let policies = reconstruct_policies(&events);
        assert_eq!(policies[0].status, PolicyStatus::Retired);
        assert_eq!(policies[1].status, PolicyStatus::Active);
        assert_eq!(policies[1].effective_generation, Some(9));
    }

    #[test]
    fn replay_reports_only_the_persisted_candidates_it_compares() {
        let policy = PolicyRevision {
            revision: 1,
            parent_revision: None,
            status: PolicyStatus::Draft,
            prompt: None,
            output_schema: None,
            parameters: zone_parameters(0.6),
            created_at_ms: 0,
            updated_at_ms: 0,
            effective_generation: None,
            effective_on_current_generation: false,
            deferred_fields: vec![],
        };
        let events = vec![
            event(2, RESTRICTED_ZONE_EVENT, json!({"confidence": 0.75})),
            event(
                3,
                RESTRICTED_ZONE_EVENT,
                json!({"assertion":{"trigger":{"score":0.4}}}),
            ),
            event(4, RESTRICTED_ZONE_EVENT, json!({})),
            event(5, "keyframe_stored", json!({"confidence": 1.0})),
        ];
        let evaluation = evaluate_replay(&policy, &events, &ReplayPolicyRequest::default());
        assert_eq!(evaluation.candidate_events, 3);
        assert_eq!(evaluation.accepted_events, 1);
        assert_eq!(evaluation.rejected_events, 1);
        assert_eq!(evaluation.events_without_score, 1);
        assert!(evaluation.limitation.contains("cannot discover"));
    }

    #[test]
    fn restricted_zone_parameters_match_worker_invariants() {
        let mut errors = Vec::new();
        validate_zone(
            zone_parameters(0.6).restricted_zone.as_ref().unwrap(),
            &mut errors,
        );
        assert!(errors.is_empty());

        let mut invalid = zone_parameters(0.1).restricted_zone.unwrap();
        invalid.region.width = 2.0;
        let mut errors = Vec::new();
        validate_zone(&invalid, &mut errors);
        assert_eq!(errors.len(), 2);
    }

    #[tokio::test]
    async fn live_policy_rejects_a_stale_generation_before_sending() {
        let state = AppState::with_wal_for_tests(std::env::temp_dir().join(format!(
            "vidarax-policy-stale-{}-{}.wal",
            std::process::id(),
            now_epoch_ms()
        )));
        let run_id = "run-0000000000000001";
        state
            .append_run_event(run_id, "run_created", json!({"principal_key": "public"}))
            .unwrap();
        let (session, _commands) =
            WebRtcSession::new_for_tests_with_generation(PipelineGeneration::new(7));
        assert!(state.insert_session(
            "sess-policy".to_string(),
            "public".to_string(),
            Arc::from(run_id),
            Arc::new(session),
        ));
        let policy = PolicyRevision {
            revision: 2,
            parent_revision: None,
            status: PolicyStatus::Canary,
            prompt: Some("watch the bay".to_string()),
            output_schema: None,
            parameters: PolicyParameters::default(),
            created_at_ms: 0,
            updated_at_ms: 0,
            effective_generation: None,
            effective_on_current_generation: false,
            deferred_fields: vec![],
        };

        let result = apply_policy(&state, run_id, &policy, true, Some(6)).await;
        assert!(matches!(
            result,
            Err(DeploymentError::StaleGeneration {
                expected: 6,
                current: 7
            })
        ));
    }
}
