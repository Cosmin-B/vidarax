use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

pub const SESSION_COMMAND_QUEUE_CAPACITY: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineGeneration(u64);

impl PipelineGeneration {
    pub const INITIAL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Decode,
    Analysis,
    ClipAccumulator,
    Vlm,
    EventWriter,
}

impl PipelineStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Analysis => "analysis",
            Self::ClipAccumulator => "clip_accumulator",
            Self::Vlm => "vlm",
            Self::EventWriter => "event_writer",
        }
    }

    pub const ALL: [Self; 5] = [
        Self::Decode,
        Self::Analysis,
        Self::ClipAccumulator,
        Self::Vlm,
        Self::EventWriter,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Decode => 0,
            Self::Analysis => 1,
            Self::ClipAccumulator => 2,
            Self::Vlm => 3,
            Self::EventWriter => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineHealth {
    Starting,
    Healthy,
    Faulted(PipelineFault),
    Stopping,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineFault {
    pub stage: PipelineStage,
    pub reason: PipelineFaultReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineFaultReason {
    UnexpectedExit,
    Panic,
    SpawnFailure,
    JoinDeadline,
}

impl PipelineFaultReason {
    pub const ALL: [Self; 4] = [
        Self::UnexpectedExit,
        Self::Panic,
        Self::SpawnFailure,
        Self::JoinDeadline,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnexpectedExit => "unexpected_exit",
            Self::Panic => "panic",
            Self::SpawnFailure => "spawn_failure",
            Self::JoinDeadline => "join_deadline",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::UnexpectedExit => 0,
            Self::Panic => 1,
            Self::SpawnFailure => 2,
            Self::JoinDeadline => 3,
        }
    }
}

#[derive(Debug)]
pub enum SessionCommand {
    UpdateConfig {
        generation: PipelineGeneration,
        prompt: Arc<str>,
        guided_json: Option<Arc<str>>,
        accepted: oneshot::Sender<Result<(), SessionControlError>>,
        decision: Arc<ConfigUpdateDecision>,
    },
}

/// Serializes cancellation with the synchronous configuration replacement.
#[derive(Debug)]
pub struct ConfigUpdateDecision(Mutex<ConfigUpdateState>);

#[derive(Debug)]
enum ConfigUpdateState {
    Pending,
    Completed(Result<(), SessionControlError>),
    Cancelled,
}

impl ConfigUpdateDecision {
    fn cancel_or_result(&self) -> Result<(), SessionControlError> {
        let mut state = self.0.lock().unwrap_or_else(|err| err.into_inner());
        match *state {
            ConfigUpdateState::Completed(result) => result,
            ConfigUpdateState::Pending | ConfigUpdateState::Cancelled => {
                *state = ConfigUpdateState::Cancelled;
                Err(SessionControlError::TimedOut)
            }
        }
    }
}

struct CancelConfigUpdate(Arc<ConfigUpdateDecision>);

impl Drop for CancelConfigUpdate {
    fn drop(&mut self) {
        let _ = self.0.cancel_or_result();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionControlError {
    Closed,
    TimedOut,
    StaleGeneration {
        expected: PipelineGeneration,
        received: PipelineGeneration,
    },
}

impl fmt::Display for SessionControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("pipeline generation is closed"),
            Self::TimedOut => f.write_str("configuration update was cancelled at its deadline"),
            Self::StaleGeneration { expected, received } => write!(
                f,
                "stale pipeline generation: expected {}, received {}",
                expected.get(),
                received.get()
            ),
        }
    }
}

impl std::error::Error for SessionControlError {}

#[derive(Clone)]
pub struct SessionControl {
    generation: PipelineGeneration,
    commands: mpsc::Sender<SessionCommand>,
    stopping: Arc<AtomicBool>,
}

impl SessionControl {
    pub fn channel(generation: PipelineGeneration) -> (Self, mpsc::Receiver<SessionCommand>) {
        let (commands, receiver) = mpsc::channel(SESSION_COMMAND_QUEUE_CAPACITY);
        (
            Self {
                generation,
                commands,
                stopping: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    pub const fn generation(&self) -> PipelineGeneration {
        self.generation
    }

    pub fn stopping_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stopping)
    }

    pub fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }

    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    pub async fn update_config(
        &self,
        prompt: Arc<str>,
        guided_json: Option<Arc<str>>,
    ) -> Result<(), SessionControlError> {
        self.update_config_inner(prompt, guided_json, None).await
    }

    /// Cancel at the deadline only if the worker has not replaced the configuration.
    pub async fn update_config_with_timeout(
        &self,
        prompt: Arc<str>,
        guided_json: Option<Arc<str>>,
        timeout: Duration,
    ) -> Result<(), SessionControlError> {
        self.update_config_inner(prompt, guided_json, Some(timeout))
            .await
    }

    async fn update_config_inner(
        &self,
        prompt: Arc<str>,
        guided_json: Option<Arc<str>>,
        timeout: Option<Duration>,
    ) -> Result<(), SessionControlError> {
        if self.is_stopping() {
            return Err(SessionControlError::Closed);
        }
        let decision = Arc::new(ConfigUpdateDecision(Mutex::new(ConfigUpdateState::Pending)));
        let cancellation = CancelConfigUpdate(Arc::clone(&decision));
        let (accepted, response) = oneshot::channel();
        let update = async {
            self.commands
                .send(SessionCommand::UpdateConfig {
                    generation: self.generation,
                    prompt,
                    guided_json,
                    accepted,
                    decision: Arc::clone(&decision),
                })
                .await
                .map_err(|_| SessionControlError::Closed)?;
            response.await.map_err(|_| SessionControlError::Closed)?
        };
        let result = match timeout {
            Some(timeout) => match tokio::time::timeout(timeout, update).await {
                Ok(result) => result,
                Err(_) => decision.cancel_or_result(),
            },
            None => update.await,
        };
        drop(cancellation);
        result
    }
}

pub fn apply_pending_session_commands(
    receiver: &mut mpsc::Receiver<SessionCommand>,
    generation: PipelineGeneration,
    prompt: &mut Arc<str>,
    guided_json: &mut Option<Arc<str>>,
) {
    // A producer may refill the bounded queue while we drain it. Leave time
    // for media work after at most one queue capacity of configuration updates.
    for _ in 0..SESSION_COMMAND_QUEUE_CAPACITY {
        let Ok(command) = receiver.try_recv() else {
            break;
        };
        match command {
            SessionCommand::UpdateConfig {
                generation: received,
                prompt: next_prompt,
                guided_json: next_guided_json,
                accepted,
                decision,
            } => {
                apply_config_command(
                    received,
                    generation,
                    next_prompt,
                    next_guided_json,
                    accepted,
                    decision,
                    prompt,
                    guided_json,
                    || {},
                    || {},
                );
            }
        }
    }
}

// The hooks bracket acquisition of the cancellation/application gate.
// Production uses empty closures; tests can synchronize competing threads here.
#[allow(clippy::too_many_arguments)]
fn apply_config_command(
    received: PipelineGeneration,
    generation: PipelineGeneration,
    next_prompt: Arc<str>,
    next_guided_json: Option<Arc<str>>,
    accepted: oneshot::Sender<Result<(), SessionControlError>>,
    decision: Arc<ConfigUpdateDecision>,
    prompt: &mut Arc<str>,
    guided_json: &mut Option<Arc<str>>,
    before_decision: impl FnOnce(),
    after_claim: impl FnOnce(),
) {
    if accepted.is_closed() {
        return;
    }
    before_decision();
    let mut state = decision.0.lock().unwrap_or_else(|err| err.into_inner());
    if !matches!(*state, ConfigUpdateState::Pending) {
        return;
    }
    after_claim();
    let result = if received != generation {
        Err(SessionControlError::StaleGeneration {
            expected: generation,
            received,
        })
    } else {
        *prompt = next_prompt;
        *guided_json = next_guided_json;
        Ok(())
    };
    *state = ConfigUpdateState::Completed(result);
    // Sending and publishing the result share the same gate as cancellation.
    let _ = accepted.send(result);
}

pub struct StageHandle {
    stage: PipelineStage,
    handle: JoinHandle<()>,
}

impl StageHandle {
    pub fn new(stage: PipelineStage, handle: JoinHandle<()>) -> Self {
        Self { stage, handle }
    }

    pub const fn stage(&self) -> PipelineStage {
        self.stage
    }

    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    pub(crate) fn join(self) {
        let _ = self.handle.join();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineShutdown {
    Clean,
    Faulted(PipelineFault),
    JoinDeadline {
        fault: Option<PipelineFault>,
        overrun: PipelineFault,
        /// Workers that were still running at the deadline. Their OS threads
        /// keep running detached and keep their memory until process exit.
        detached: u32,
    },
}

/// VLM request timeouts for one work item. Shared here so the join deadline
/// below can be derived from them instead of drifting apart.
pub const KEYFRAME_FIRST_PASS_TIMEOUT_MS: u64 = 5_000;
pub const KEYFRAME_SECOND_PASS_TIMEOUT_MS: u64 = 10_000;
pub const CLIP_FIRST_PASS_TIMEOUT_MS: u64 = 15_000;
pub const CLIP_SECOND_PASS_TIMEOUT_MS: u64 = 20_000;

/// Inputs for deriving the supervisor join deadline from configuration.
pub struct JoinDeadlineInputs {
    /// Serial inference attempts one call can make (provider fallback chain).
    pub max_serial_inference_attempts: u64,
    /// Admission wait before a pass may run (AdmissionLimits::wait_timeout).
    pub admission_wait_ms: u64,
    /// End-to-end sidecar embedding exchange timeout.
    pub novelty_embedding_timeout_ms: u64,
}

/// Join deadline for generation teardown, derived from the work a healthy
/// worker can legitimately be inside when stop is raised: an admission wait
/// before each of the two tiered passes, the serial fallback attempts of one
/// tiered call, and one end-to-end sidecar exchange. The serial inference
/// allowance is conservative; providers also share the request deadline.
/// All arithmetic saturates and the result is capped at 24 hours.
pub fn supervise_join_deadline_from(inputs: &JoinDeadlineInputs) -> Duration {
    const CAP_MS: u64 = 86_400_000;
    let per_attempt = CLIP_FIRST_PASS_TIMEOUT_MS + CLIP_SECOND_PASS_TIMEOUT_MS;
    let inference = inputs
        .max_serial_inference_attempts
        .max(1)
        .saturating_mul(per_attempt);
    let admission = inputs.admission_wait_ms.saturating_mul(2);
    let novelty = inputs.novelty_embedding_timeout_ms;
    let total = inference
        .saturating_add(admission)
        .saturating_add(novelty)
        .saturating_add(5_000);
    Duration::from_millis(total.min(CAP_MS))
}

/// Single-backend deadline with no admission or novelty allowance. Prefer
/// supervise_join_deadline_from with real configuration values.
pub fn supervise_join_deadline() -> Duration {
    supervise_join_deadline_from(&JoinDeadlineInputs {
        max_serial_inference_attempts: 1,
        admission_wait_ms: 0,
        novelty_embedding_timeout_ms: 0,
    })
}

#[derive(Debug)]
pub struct PipelineStartError {
    pub fault: PipelineFault,
    pub join_deadline: Option<PipelineFault>,
    /// Workers still running when the startup abort hit its deadline. They
    /// keep running detached and keep their memory.
    pub detached: u32,
    source: io::Error,
}

#[derive(Debug)]
pub struct StageSpawnError {
    pub stage: PipelineStage,
    source: io::Error,
}

impl StageSpawnError {
    pub fn new(stage: PipelineStage, source: io::Error) -> Self {
        Self { stage, source }
    }

    pub fn into_parts(self) -> (PipelineStage, io::Error) {
        (self.stage, self.source)
    }
}

impl PipelineStartError {
    pub fn new(
        stage: PipelineStage,
        source: io::Error,
        join_deadline: Option<PipelineFault>,
        detached: u32,
    ) -> Self {
        Self {
            fault: PipelineFault {
                stage,
                reason: PipelineFaultReason::SpawnFailure,
            },
            join_deadline,
            detached,
            source,
        }
    }
}

impl fmt::Display for PipelineStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} worker failed to start: {}",
            self.fault.stage.as_str(),
            self.source
        )?;
        if let Some(deadline) = self.join_deadline {
            write!(
                f,
                "; {} worker exceeded startup rollback deadline",
                deadline.stage.as_str()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for PipelineStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub struct PipelineRuntime {
    generation: PipelineGeneration,
    health: PipelineHealth,
    stopping: Arc<AtomicBool>,
    workers: Vec<StageHandle>,
}

impl PipelineRuntime {
    pub fn new(generation: PipelineGeneration, stopping: Arc<AtomicBool>) -> Self {
        Self {
            generation,
            health: PipelineHealth::Starting,
            stopping,
            workers: Vec::new(),
        }
    }

    pub const fn generation(&self) -> PipelineGeneration {
        self.generation
    }

    pub const fn health(&self) -> PipelineHealth {
        self.health
    }

    pub fn mark_healthy(&mut self) {
        if self.health == PipelineHealth::Starting {
            self.health = PipelineHealth::Healthy;
        }
    }

    pub fn push(&mut self, worker: StageHandle) {
        self.workers.push(worker);
    }

    pub fn extend(&mut self, workers: impl IntoIterator<Item = StageHandle>) {
        self.workers.extend(workers);
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }

    /// Stop and bounded-join workers that were already created when a later
    /// pipeline stage failed to start. This prevents partial startup from
    /// silently detaching the successfully spawned prefix.
    pub fn abort_startup(&mut self, join_deadline: Duration) -> Option<PipelineFault> {
        self.health = PipelineHealth::Stopping;
        self.stop();
        let deadline = Instant::now() + join_deadline;
        while !self.workers.is_empty() {
            if let Some(index) = self.workers.iter().position(StageHandle::is_finished) {
                let worker = self.workers.swap_remove(index);
                let _ = worker.handle.join();
                continue;
            }
            if Instant::now() >= deadline {
                return Some(PipelineFault {
                    stage: self.workers[0].stage,
                    reason: PipelineFaultReason::JoinDeadline,
                });
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        self.health = PipelineHealth::Stopped;
        None
    }

    /// Observe the first exit, fault a live generation, and wait a bounded time
    /// for the remaining workers. OS threads cannot be force-killed safely; a
    /// deadline therefore records and detaches a stuck worker after stop was
    /// raised, while the session peer is closed by `on_fault`.
    pub fn supervise(
        mut self,
        join_deadline: Duration,
        on_fault: impl FnOnce(PipelineFault),
    ) -> PipelineShutdown {
        let mut on_fault = Some(on_fault);
        let mut first_fault = None;
        if self.health == PipelineHealth::Starting {
            self.health = PipelineHealth::Healthy;
        }
        if self.stopping() {
            self.health = PipelineHealth::Stopping;
        }
        let mut stop_deadline = self.stopping().then(|| Instant::now() + join_deadline);

        loop {
            if self.workers.is_empty() {
                self.health = PipelineHealth::Stopped;
                return first_fault.map_or(PipelineShutdown::Clean, PipelineShutdown::Faulted);
            }

            if let Some(index) = self.workers.iter().position(StageHandle::is_finished) {
                let worker = self.workers.swap_remove(index);
                let stage = worker.stage;
                let panicked = worker.handle.join().is_err();
                if !self.stopping() && first_fault.is_none() {
                    let fault = PipelineFault {
                        stage,
                        reason: if panicked {
                            PipelineFaultReason::Panic
                        } else {
                            PipelineFaultReason::UnexpectedExit
                        },
                    };
                    first_fault = Some(fault);
                    self.health = PipelineHealth::Faulted(fault);
                    self.stop();
                    if let Some(callback) = on_fault.take() {
                        callback(fault);
                    }
                    stop_deadline = Some(Instant::now() + join_deadline);
                }
                continue;
            }

            if self.stopping() && stop_deadline.is_none() {
                self.health = PipelineHealth::Stopping;
                stop_deadline = Some(Instant::now() + join_deadline);
            }

            if stop_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                let stage = self.workers[0].stage;
                self.health = PipelineHealth::Faulted(PipelineFault {
                    stage,
                    reason: PipelineFaultReason::JoinDeadline,
                });
                return PipelineShutdown::JoinDeadline {
                    fault: first_fault,
                    overrun: PipelineFault {
                        stage,
                        reason: PipelineFaultReason::JoinDeadline,
                    },
                    detached: self.workers.len() as u32,
                };
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for PipelineRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn join_deadline_includes_admission_and_novelty_budgets() {
        let d = super::supervise_join_deadline_from(&super::JoinDeadlineInputs {
            max_serial_inference_attempts: 2,
            admission_wait_ms: 120_000,
            novelty_embedding_timeout_ms: 1_500,
        });
        let expected = 2 * (super::CLIP_FIRST_PASS_TIMEOUT_MS + super::CLIP_SECOND_PASS_TIMEOUT_MS)
            + 2 * 120_000
            + 1_500
            + 5_000;
        assert_eq!(d, std::time::Duration::from_millis(expected));
    }

    #[test]
    fn join_deadline_saturates_instead_of_overflowing() {
        let d = super::supervise_join_deadline_from(&super::JoinDeadlineInputs {
            max_serial_inference_attempts: u64::MAX,
            admission_wait_ms: u64::MAX,
            novelty_embedding_timeout_ms: u64::MAX,
        });
        assert_eq!(d, std::time::Duration::from_millis(86_400_000));
    }

    use super::{
        apply_pending_session_commands, PipelineFaultReason, PipelineGeneration, PipelineRuntime,
        PipelineShutdown, PipelineStage, SessionControl, SessionControlError, StageHandle,
    };
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn update_acknowledges_only_after_generation_accepts_command() {
        let generation = PipelineGeneration::new(7);
        let (control, mut receiver) = SessionControl::channel(generation);
        let update = tokio::spawn(async move {
            control
                .update_config(Arc::from("next"), Some(Arc::from("{}")))
                .await
        });

        tokio::task::yield_now().await;
        assert!(!update.is_finished());

        let mut prompt: Arc<str> = Arc::from("old");
        let mut schema = None;
        apply_pending_session_commands(&mut receiver, generation, &mut prompt, &mut schema);
        assert_eq!(prompt.as_ref(), "next");
        assert_eq!(schema.as_deref(), Some("{}"));
        assert_eq!(update.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn stopped_control_rejects_updates() {
        let (control, _receiver) = SessionControl::channel(PipelineGeneration::new(2));
        control.stop();
        assert_eq!(
            control.update_config(Arc::from("next"), None).await,
            Err(SessionControlError::Closed)
        );
    }

    #[tokio::test]
    async fn cancelled_acknowledgement_does_not_apply_command_later() {
        let generation = PipelineGeneration::new(3);
        let (control, mut receiver) = SessionControl::channel(generation);
        let update =
            tokio::spawn(async move { control.update_config(Arc::from("late"), None).await });
        tokio::task::yield_now().await;
        update.abort();
        let _ = update.await;

        let mut prompt: Arc<str> = Arc::from("current");
        let mut schema = None;
        apply_pending_session_commands(&mut receiver, generation, &mut prompt, &mut schema);
        assert_eq!(prompt.as_ref(), "current");
    }

    #[tokio::test]
    async fn deadline_cancels_before_worker_claim() {
        let generation = PipelineGeneration::new(3);
        let (control, mut receiver) = SessionControl::channel(generation);
        let update = tokio::spawn(async move {
            control
                .update_config_with_timeout(
                    Arc::from("late"),
                    Some(Arc::from("{}")),
                    Duration::from_millis(10),
                )
                .await
        });
        // Receipt proves admission happened before the deadline.
        let command = receiver.recv().await.unwrap();
        assert_eq!(update.await.unwrap(), Err(SessionControlError::TimedOut));
        let super::SessionCommand::UpdateConfig {
            generation: received,
            prompt: next_prompt,
            guided_json: next_schema,
            accepted,
            decision,
        } = command;
        let mut prompt = Arc::from("current");
        let mut schema = Some(Arc::from("old-schema"));
        super::apply_config_command(
            received,
            generation,
            next_prompt,
            next_schema,
            accepted,
            decision,
            &mut prompt,
            &mut schema,
            || {},
            || {},
        );
        assert_eq!(prompt.as_ref(), "current");
        assert_eq!(schema.as_deref(), Some("old-schema"));
    }

    #[tokio::test]
    async fn cancellation_after_open_receiver_check_prevents_replacement() {
        let generation = PipelineGeneration::new(3);
        let (control, mut receiver) = SessionControl::channel(generation);
        let update = tokio::spawn(async move {
            control
                .update_config(Arc::from("late"), Some(Arc::from("{}")))
                .await
        });
        let command = receiver.recv().await.unwrap();
        let (checked_tx, checked_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let super::SessionCommand::UpdateConfig {
                generation: received,
                prompt: next_prompt,
                guided_json: next_schema,
                accepted,
                decision,
            } = command;
            let mut prompt = Arc::from("current");
            let mut schema = Some(Arc::from("old-schema"));
            super::apply_config_command(
                received,
                generation,
                next_prompt,
                next_schema,
                accepted,
                decision,
                &mut prompt,
                &mut schema,
                || {
                    checked_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(1)).unwrap();
                },
                || {},
            );
            (prompt, schema)
        });
        checked_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        update.abort();
        let _ = update.await;
        resume_tx.send(()).unwrap();
        let (prompt, schema) = worker.join().unwrap();
        assert_eq!(prompt.as_ref(), "current");
        assert_eq!(schema.as_deref(), Some("old-schema"));
    }

    #[test]
    fn cancellation_after_worker_claim_returns_applied_result() {
        let generation = PipelineGeneration::new(3);
        let decision = Arc::new(super::ConfigUpdateDecision(std::sync::Mutex::new(
            super::ConfigUpdateState::Pending,
        )));
        let cancellation = Arc::clone(&decision);
        let (accepted, response) = tokio::sync::oneshot::channel();
        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut prompt = Arc::from("current");
            let mut schema = Some(Arc::from("old-schema"));
            super::apply_config_command(
                generation,
                generation,
                Arc::from("next"),
                Some(Arc::from("{}")),
                accepted,
                decision,
                &mut prompt,
                &mut schema,
                || {},
                || {
                    claimed_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(1)).unwrap();
                },
            );
            (prompt, schema)
        });
        claimed_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let canceller = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            cancellation.cancel_or_result()
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        resume_tx.send(()).unwrap();
        let (prompt, schema) = worker.join().unwrap();
        assert_eq!(canceller.join().unwrap(), Ok(()));
        assert_eq!(response.blocking_recv().unwrap(), Ok(()));
        assert_eq!(prompt.as_ref(), "next");
        assert_eq!(schema.as_deref(), Some("{}"));
    }

    #[tokio::test]
    async fn stale_generation_rejects_both_prompt_and_schema() {
        let received = PipelineGeneration::new(3);
        let active = PipelineGeneration::new(4);
        let (control, mut receiver) = SessionControl::channel(received);
        let update = tokio::spawn(async move {
            control
                .update_config(Arc::from("next"), Some(Arc::from("{}")))
                .await
        });
        let command = receiver.recv().await.unwrap();
        let super::SessionCommand::UpdateConfig {
            generation,
            prompt: next_prompt,
            guided_json: next_schema,
            accepted,
            decision,
        } = command;
        let mut prompt = Arc::from("current");
        let mut schema = Some(Arc::from("old-schema"));
        super::apply_config_command(
            generation,
            active,
            next_prompt,
            next_schema,
            accepted,
            decision,
            &mut prompt,
            &mut schema,
            || {},
            || {},
        );
        assert_eq!(
            update.await.unwrap(),
            Err(SessionControlError::StaleGeneration {
                expected: active,
                received,
            })
        );
        assert_eq!(prompt.as_ref(), "current");
        assert_eq!(schema.as_deref(), Some("old-schema"));
    }

    #[test]
    fn unexpected_worker_exit_faults_generation_and_stops_siblings() {
        let stopping = Arc::new(AtomicBool::new(false));
        let mut runtime = PipelineRuntime::new(PipelineGeneration::new(9), Arc::clone(&stopping));
        runtime.push(StageHandle::new(
            PipelineStage::Decode,
            std::thread::spawn(|| {}),
        ));
        let sibling_stop = Arc::clone(&stopping);
        runtime.push(StageHandle::new(
            PipelineStage::Vlm,
            std::thread::spawn(move || {
                while !sibling_stop.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }),
        ));

        let outcome = runtime.supervise(Duration::from_secs(1), |_| {});
        assert_eq!(
            outcome,
            PipelineShutdown::Faulted(super::PipelineFault {
                stage: PipelineStage::Decode,
                reason: PipelineFaultReason::UnexpectedExit,
            })
        );
    }

    #[test]
    fn explicit_stop_is_a_clean_shutdown() {
        let stopping = Arc::new(AtomicBool::new(false));
        let mut runtime = PipelineRuntime::new(PipelineGeneration::new(10), Arc::clone(&stopping));
        let worker_stop = Arc::clone(&stopping);
        runtime.push(StageHandle::new(
            PipelineStage::Analysis,
            std::thread::spawn(move || {
                while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }),
        ));
        runtime.stop();

        assert_eq!(
            runtime.supervise(Duration::from_secs(1), |_| panic!("clean stop faulted")),
            PipelineShutdown::Clean
        );
    }

    #[test]
    fn explicit_stop_enforces_join_deadline() {
        let stopping = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let mut runtime = PipelineRuntime::new(PipelineGeneration::new(11), Arc::clone(&stopping));
        let worker_release = Arc::clone(&release);
        runtime.push(StageHandle::new(
            PipelineStage::Analysis,
            std::thread::spawn(move || {
                while !worker_release.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }),
        ));
        runtime.stop();

        let outcome = runtime.supervise(Duration::from_millis(20), |_| {
            panic!("clean stop invoked fault callback")
        });
        release.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(
            outcome,
            PipelineShutdown::JoinDeadline {
                fault: None,
                overrun: super::PipelineFault {
                    stage: PipelineStage::Analysis,
                    reason: PipelineFaultReason::JoinDeadline,
                },
                detached: 1,
            }
        );
    }
}
