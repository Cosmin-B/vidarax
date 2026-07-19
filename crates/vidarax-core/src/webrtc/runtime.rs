use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionControlError {
    Closed,
    StaleGeneration {
        expected: PipelineGeneration,
        received: PipelineGeneration,
    },
}

impl fmt::Display for SessionControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("pipeline generation is closed"),
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
        if self.is_stopping() {
            return Err(SessionControlError::Closed);
        }
        let (accepted, response) = oneshot::channel();
        self.commands
            .send(SessionCommand::UpdateConfig {
                generation: self.generation,
                prompt,
                guided_json,
                accepted,
            })
            .await
            .map_err(|_| SessionControlError::Closed)?;
        response.await.map_err(|_| SessionControlError::Closed)?
    }
}

pub fn apply_pending_session_commands(
    receiver: &mut mpsc::Receiver<SessionCommand>,
    generation: PipelineGeneration,
    prompt: &mut Arc<str>,
    guided_json: &mut Option<Arc<str>>,
) {
    while let Ok(command) = receiver.try_recv() {
        match command {
            SessionCommand::UpdateConfig {
                generation: received,
                prompt: next_prompt,
                guided_json: next_guided_json,
                accepted,
            } => {
                // The HTTP acknowledgement deadline owns command validity. If
                // the caller timed out and dropped its receiver while this
                // worker was in inference, do not apply the now-unobservable
                // update later.
                if accepted.is_closed() {
                    continue;
                }
                if received != generation {
                    let _ = accepted.send(Err(SessionControlError::StaleGeneration {
                        expected: generation,
                        received,
                    }));
                    continue;
                }
                *prompt = next_prompt;
                *guided_json = next_guided_json;
                let _ = accepted.send(Ok(()));
            }
        }
    }
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

/// Join deadline for generation teardown. A VLM worker can legitimately sit
/// inside one tiered call for the sum of both pass timeouts, so the deadline
/// must exceed the slowest path (clip mode) or ordinary teardown during an
/// in-flight call gets reported as a forced shutdown and detaches threads.
pub fn supervise_join_deadline() -> Duration {
    Duration::from_millis(CLIP_FIRST_PASS_TIMEOUT_MS + CLIP_SECOND_PASS_TIMEOUT_MS + 5_000)
}

#[derive(Debug)]
pub struct PipelineStartError {
    pub fault: PipelineFault,
    pub join_deadline: Option<PipelineFault>,
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
    ) -> Self {
        Self {
            fault: PipelineFault {
                stage,
                reason: PipelineFaultReason::SpawnFailure,
            },
            join_deadline,
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
