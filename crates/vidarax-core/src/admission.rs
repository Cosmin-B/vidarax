//! Exact, bounded admission for blocking inference calls.
//!
//! Provider work runs on OS threads or `spawn_blocking`, so a mutex and
//! condition variable keep the policy small without adding synchronization to
//! decode or gate hot paths.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionLimits {
    pub global_in_flight: usize,
    pub per_principal_in_flight: usize,
    pub global_waiters: usize,
    pub wait_timeout: Duration,
    /// Aggregate output-token reservation across active provider calls.
    pub max_in_flight_tokens: u64,
    /// Aggregate encoded media bytes held by active provider calls.
    pub max_in_flight_bytes: u64,
}

impl AdmissionLimits {
    fn validate(self) -> Result<Self, &'static str> {
        if self.global_in_flight == 0 {
            return Err("global inference limit must be greater than zero");
        }
        if self.per_principal_in_flight == 0 {
            return Err("per-principal inference limit must be greater than zero");
        }
        if self.per_principal_in_flight > self.global_in_flight {
            return Err("per-principal inference limit cannot exceed the global limit");
        }
        if self.global_waiters == 0 {
            return Err("inference waiter limit must be greater than zero");
        }
        if self.wait_timeout.is_zero() {
            return Err("inference wait timeout must be greater than zero");
        }
        if self.max_in_flight_tokens == 0 || self.max_in_flight_bytes == 0 {
            return Err("inference token and byte budgets must be greater than zero");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyClass {
    UrgentLive,
    Live,
    Offline,
}

impl LatencyClass {
    const fn rank(self) -> u8 {
        match self {
            Self::UrgentLive => 0,
            Self::Live => 1,
            Self::Offline => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AdmissionRequest<'a> {
    pub principal: &'a str,
    pub stream: &'a str,
    pub class: LatencyClass,
    /// Total queue budget from admission arrival.
    pub deadline: Duration,
    /// Conservative service time reserved before the deadline.
    pub estimated_service: Duration,
    pub tokens: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitClass {
    Global,
    Principal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    WaiterLimit,
    Timeout { blocked_by: LimitClass },
    DeadlineMissed,
    RequestBudget,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WaiterLimit => f.write_str("inference admission waiter limit reached"),
            Self::Timeout {
                blocked_by: LimitClass::Global,
            } => f.write_str("timed out waiting for global inference capacity"),
            Self::Timeout {
                blocked_by: LimitClass::Principal,
            } => f.write_str("timed out waiting for principal inference capacity"),
            Self::DeadlineMissed => {
                f.write_str("inference deadline cannot be met before provider dispatch")
            }
            Self::RequestBudget => {
                f.write_str("inference request exceeds the process token or byte budget")
            }
        }
    }
}

impl std::error::Error for AdmissionError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionSnapshot {
    pub active: usize,
    pub waiting: usize,
    pub principal_entries: usize,
    pub acquired_total: u64,
    pub timeout_global_total: u64,
    pub timeout_principal_total: u64,
    pub waiter_rejected_total: u64,
    pub wait_duration_us: u64,
    pub wait_duration_count: u64,
    pub active_tokens: u64,
    pub active_bytes: u64,
    pub deadline_missed_total: u64,
    pub budget_rejected_total: u64,
    pub urgent_acquired_total: u64,
    pub live_acquired_total: u64,
    pub offline_acquired_total: u64,
}

#[derive(Debug, Default)]
struct PrincipalCounts {
    active: usize,
    waiting: usize,
}

#[derive(Debug)]
struct Waiter {
    ticket: u64,
    principal: Box<str>,
    stream: Box<str>,
    class: LatencyClass,
    queued_at: Instant,
    latest_start: Instant,
    tokens: u64,
    bytes: u64,
    wake: std::sync::Arc<Condvar>,
}

#[derive(Debug, Default)]
struct AdmissionState {
    active: usize,
    active_tokens: u64,
    active_bytes: u64,
    principals: HashMap<Box<str>, PrincipalCounts>,
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
    last_granted_principal: Option<Box<str>>,
    last_granted_stream: Option<Box<str>>,
}

#[derive(Debug, Default)]
struct AdmissionMetrics {
    acquired_total: AtomicU64,
    timeout_global_total: AtomicU64,
    timeout_principal_total: AtomicU64,
    waiter_rejected_total: AtomicU64,
    wait_duration_us: AtomicU64,
    wait_duration_count: AtomicU64,
    deadline_missed_total: AtomicU64,
    budget_rejected_total: AtomicU64,
    urgent_acquired_total: AtomicU64,
    live_acquired_total: AtomicU64,
    offline_acquired_total: AtomicU64,
}

#[derive(Debug)]
pub struct InferenceAdmission {
    limits: AdmissionLimits,
    state: Mutex<AdmissionState>,
    changed: Condvar,
    metrics: AdmissionMetrics,
}

impl InferenceAdmission {
    pub fn new(limits: AdmissionLimits) -> Result<Self, &'static str> {
        Ok(Self {
            limits: limits.validate()?,
            state: Mutex::new(AdmissionState::default()),
            changed: Condvar::new(),
            metrics: AdmissionMetrics::default(),
        })
    }

    pub fn limits(&self) -> AdmissionLimits {
        self.limits
    }

    pub fn record_deadline_missed(&self) {
        self.metrics
            .deadline_missed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn acquire<'a>(
        &'a self,
        principal: &'a str,
    ) -> Result<AdmissionPermit<'a>, AdmissionError> {
        self.acquire_with_timeout(principal, self.limits.wait_timeout)
    }

    pub fn acquire_with_timeout<'a>(
        &'a self,
        principal: &'a str,
        timeout: Duration,
    ) -> Result<AdmissionPermit<'a>, AdmissionError> {
        self.acquire_scheduled_with_wait(
            AdmissionRequest {
                principal,
                stream: "direct",
                class: LatencyClass::Live,
                deadline: timeout.saturating_add(Duration::from_secs(86_400)),
                estimated_service: Duration::ZERO,
                tokens: 0,
                bytes: 0,
            },
            timeout,
        )
    }

    pub fn acquire_scheduled<'a>(
        &'a self,
        request: AdmissionRequest<'a>,
    ) -> Result<AdmissionPermit<'a>, AdmissionError> {
        self.acquire_scheduled_with_wait(request, self.limits.wait_timeout)
    }

    fn acquire_scheduled_with_wait<'a>(
        &'a self,
        request: AdmissionRequest<'a>,
        wait_timeout: Duration,
    ) -> Result<AdmissionPermit<'a>, AdmissionError> {
        let started = Instant::now();
        if request.tokens > self.limits.max_in_flight_tokens
            || request.bytes > self.limits.max_in_flight_bytes
        {
            self.metrics
                .budget_rejected_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::RequestBudget);
        }
        let Some(queue_deadline) = request.deadline.checked_sub(request.estimated_service) else {
            self.metrics
                .deadline_missed_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::DeadlineMissed);
        };
        if queue_deadline.is_zero() {
            self.metrics
                .deadline_missed_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::DeadlineMissed);
        }
        let latest_start = started.checked_add(queue_deadline).unwrap_or(started);
        let wait_deadline = started
            .checked_add(wait_timeout)
            .unwrap_or(started)
            .min(latest_start);
        let mut state = self.lock_state();

        if self.can_acquire(
            &state,
            request.principal,
            request.tokens,
            request.bytes,
            None,
            started,
        ) {
            self.mark_acquired(
                &mut state,
                request.principal,
                request.stream,
                request.class,
                request.tokens,
                request.bytes,
                started.elapsed(),
            );
            return Ok(AdmissionPermit {
                admission: self,
                principal: request.principal,
                tokens: request.tokens,
                bytes: request.bytes,
                released: false,
            });
        }

        if state.waiters.len() >= self.limits.global_waiters {
            self.metrics
                .waiter_rejected_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(AdmissionError::WaiterLimit);
        }

        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let wake = std::sync::Arc::new(Condvar::new());
        state.waiters.push_back(Waiter {
            ticket,
            principal: request.principal.into(),
            stream: request.stream.into(),
            class: request.class,
            queued_at: started,
            latest_start,
            tokens: request.tokens,
            bytes: request.bytes,
            wake: std::sync::Arc::clone(&wake),
        });
        state
            .principals
            .entry(request.principal.into())
            .or_default()
            .waiting += 1;
        self.changed.notify_all();

        loop {
            let now = Instant::now();
            if self.can_acquire(
                &state,
                request.principal,
                request.tokens,
                request.bytes,
                Some(ticket),
                now,
            ) {
                self.remove_waiter(&mut state, ticket, request.principal);
                self.mark_acquired(
                    &mut state,
                    request.principal,
                    request.stream,
                    request.class,
                    request.tokens,
                    request.bytes,
                    started.elapsed(),
                );
                self.notify_selected(&state, now);
                return Ok(AdmissionPermit {
                    admission: self,
                    principal: request.principal,
                    tokens: request.tokens,
                    bytes: request.bytes,
                    released: false,
                });
            }

            if now >= wait_deadline {
                let blocked_by = self.blocked_by(&state, request.principal);
                self.remove_waiter(&mut state, ticket, request.principal);
                let error = if now >= latest_start {
                    self.metrics
                        .deadline_missed_total
                        .fetch_add(1, Ordering::Relaxed);
                    self.record_wait(started.elapsed());
                    AdmissionError::DeadlineMissed
                } else {
                    self.record_timeout(blocked_by, started.elapsed());
                    AdmissionError::Timeout { blocked_by }
                };
                self.notify_selected(&state, now);
                drop(state);
                return Err(error);
            }

            let remaining = wait_deadline.saturating_duration_since(now);
            let (next, _) = wake
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }
    }

    pub fn snapshot(&self) -> AdmissionSnapshot {
        let state = self.lock_state();
        AdmissionSnapshot {
            active: state.active,
            waiting: state.waiters.len(),
            principal_entries: state.principals.len(),
            acquired_total: self.metrics.acquired_total.load(Ordering::Relaxed),
            timeout_global_total: self.metrics.timeout_global_total.load(Ordering::Relaxed),
            timeout_principal_total: self.metrics.timeout_principal_total.load(Ordering::Relaxed),
            waiter_rejected_total: self.metrics.waiter_rejected_total.load(Ordering::Relaxed),
            wait_duration_us: self.metrics.wait_duration_us.load(Ordering::Relaxed),
            wait_duration_count: self.metrics.wait_duration_count.load(Ordering::Relaxed),
            active_tokens: state.active_tokens,
            active_bytes: state.active_bytes,
            deadline_missed_total: self.metrics.deadline_missed_total.load(Ordering::Relaxed),
            budget_rejected_total: self.metrics.budget_rejected_total.load(Ordering::Relaxed),
            urgent_acquired_total: self.metrics.urgent_acquired_total.load(Ordering::Relaxed),
            live_acquired_total: self.metrics.live_acquired_total.load(Ordering::Relaxed),
            offline_acquired_total: self.metrics.offline_acquired_total.load(Ordering::Relaxed),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, AdmissionState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn can_acquire(
        &self,
        state: &AdmissionState,
        principal: &str,
        tokens: u64,
        bytes: u64,
        own_ticket: Option<u64>,
        now: Instant,
    ) -> bool {
        if !self.has_capacity(state, principal, tokens, bytes) {
            return false;
        }
        if state.waiters.is_empty() {
            return own_ticket.is_none();
        }
        let selected = self.select_waiter(state, now);
        selected == own_ticket
    }

    fn has_capacity(
        &self,
        state: &AdmissionState,
        principal: &str,
        tokens: u64,
        bytes: u64,
    ) -> bool {
        state.active < self.limits.global_in_flight
            && state.active_tokens.saturating_add(tokens) <= self.limits.max_in_flight_tokens
            && state.active_bytes.saturating_add(bytes) <= self.limits.max_in_flight_bytes
            && state
                .principals
                .get(principal)
                .map_or(0, |counts| counts.active)
                < self.limits.per_principal_in_flight
    }

    fn select_waiter(&self, state: &AdmissionState, now: Instant) -> Option<u64> {
        const AGING: Duration = Duration::from_millis(250);
        state
            .waiters
            .iter()
            .filter(|waiter| {
                now < waiter.latest_start
                    && self.has_capacity(state, &waiter.principal, waiter.tokens, waiter.bytes)
            })
            .min_by_key(|waiter| {
                let promotions =
                    now.duration_since(waiter.queued_at).as_millis() / AGING.as_millis();
                let effective_class = waiter
                    .class
                    .rank()
                    .saturating_sub(promotions.min(u128::from(u8::MAX)) as u8);
                let repeated_principal = state
                    .last_granted_principal
                    .as_deref()
                    .is_some_and(|last| last == waiter.principal.as_ref());
                let repeated_stream = state
                    .last_granted_stream
                    .as_deref()
                    .is_some_and(|last| last == waiter.stream.as_ref());
                (
                    effective_class,
                    repeated_principal,
                    repeated_stream,
                    waiter.latest_start,
                    waiter.ticket,
                )
            })
            .map(|waiter| waiter.ticket)
    }

    fn notify_selected(&self, state: &AdmissionState, now: Instant) {
        let Some(ticket) = self.select_waiter(state, now) else {
            return;
        };
        if let Some(waiter) = state.waiters.iter().find(|waiter| waiter.ticket == ticket) {
            waiter.wake.notify_one();
        }
    }

    fn blocked_by(&self, state: &AdmissionState, principal: &str) -> LimitClass {
        let principal_active = state
            .principals
            .get(principal)
            .map_or(0, |counts| counts.active);
        if principal_active >= self.limits.per_principal_in_flight {
            LimitClass::Principal
        } else {
            LimitClass::Global
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mark_acquired(
        &self,
        state: &mut AdmissionState,
        principal: &str,
        stream: &str,
        class: LatencyClass,
        tokens: u64,
        bytes: u64,
        waited: Duration,
    ) {
        state.active += 1;
        state.active_tokens = state.active_tokens.saturating_add(tokens);
        state.active_bytes = state.active_bytes.saturating_add(bytes);
        state.last_granted_principal = Some(principal.into());
        state.last_granted_stream = Some(stream.into());
        state.principals.entry(principal.into()).or_default().active += 1;
        self.metrics.acquired_total.fetch_add(1, Ordering::Relaxed);
        match class {
            LatencyClass::UrgentLive => &self.metrics.urgent_acquired_total,
            LatencyClass::Live => &self.metrics.live_acquired_total,
            LatencyClass::Offline => &self.metrics.offline_acquired_total,
        }
        .fetch_add(1, Ordering::Relaxed);
        self.record_wait(waited);
    }

    fn remove_waiter(&self, state: &mut AdmissionState, ticket: u64, principal: &str) {
        if let Some(position) = state
            .waiters
            .iter()
            .position(|waiter| waiter.ticket == ticket)
        {
            state.waiters.remove(position);
        }
        if let Some(counts) = state.principals.get_mut(principal) {
            counts.waiting = counts.waiting.saturating_sub(1);
            if counts.waiting == 0 && counts.active == 0 {
                state.principals.remove(principal);
            }
        }
    }

    fn record_timeout(&self, blocked_by: LimitClass, waited: Duration) {
        match blocked_by {
            LimitClass::Global => &self.metrics.timeout_global_total,
            LimitClass::Principal => &self.metrics.timeout_principal_total,
        }
        .fetch_add(1, Ordering::Relaxed);
        self.record_wait(waited);
    }

    fn record_wait(&self, waited: Duration) {
        self.metrics.wait_duration_us.fetch_add(
            waited.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        self.metrics
            .wait_duration_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn release(&self, principal: &str, tokens: u64, bytes: u64) {
        let mut state = self.lock_state();
        state.active = state.active.saturating_sub(1);
        state.active_tokens = state.active_tokens.saturating_sub(tokens);
        state.active_bytes = state.active_bytes.saturating_sub(bytes);
        if let Some(counts) = state.principals.get_mut(principal) {
            counts.active = counts.active.saturating_sub(1);
            if counts.active == 0 && counts.waiting == 0 {
                state.principals.remove(principal);
            }
        }
        self.notify_selected(&state, Instant::now());
        drop(state);
    }

    #[cfg(test)]
    fn wait_for_waiters(&self, expected: usize, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let mut state = self.lock_state();
        while state.waiters.len() < expected {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "waiters did not reach {expected}");
            let (next, _) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }
    }
}

#[derive(Debug)]
#[must_use = "dropping the permit releases inference capacity"]
pub struct AdmissionPermit<'a> {
    admission: &'a InferenceAdmission,
    principal: &'a str,
    tokens: u64,
    bytes: u64,
    released: bool,
}

impl Drop for AdmissionPermit<'_> {
    fn drop(&mut self) {
        if !self.released {
            self.admission
                .release(self.principal, self.tokens, self.bytes);
            self.released = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use super::{
        AdmissionError, AdmissionLimits, AdmissionRequest, InferenceAdmission, LatencyClass,
        LimitClass,
    };

    fn limits(global: usize, per_principal: usize) -> AdmissionLimits {
        AdmissionLimits {
            global_in_flight: global,
            per_principal_in_flight: per_principal,
            global_waiters: 8,
            wait_timeout: Duration::from_secs(1),
            max_in_flight_tokens: 1_000_000,
            max_in_flight_bytes: 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn global_limit_is_exact_under_concurrency() {
        let admission = Arc::new(InferenceAdmission::new(limits(2, 2)).unwrap());
        let entered = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();

        for principal in ["one", "two"] {
            let admission = Arc::clone(&admission);
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            handles.push(thread::spawn(move || {
                let _permit = admission.acquire(principal).unwrap();
                entered.wait();
                release.wait();
            }));
        }

        entered.wait();
        assert_eq!(admission.snapshot().active, 2);
        assert!(matches!(
            admission.acquire_with_timeout("three", Duration::from_millis(10)),
            Err(AdmissionError::Timeout {
                blocked_by: LimitClass::Global,
            })
        ));
        assert_eq!(admission.snapshot().active, 2);
        release.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(admission.snapshot().active, 0);
    }

    #[test]
    fn principal_limit_is_exact_under_concurrency() {
        let admission = InferenceAdmission::new(limits(3, 1)).unwrap();
        let _held = admission.acquire("same").unwrap();
        let other = admission.acquire("other").unwrap();

        assert!(matches!(
            admission.acquire_with_timeout("same", Duration::from_millis(10)),
            Err(AdmissionError::Timeout {
                blocked_by: LimitClass::Principal,
            })
        ));
        assert_eq!(admission.snapshot().active, 2);
        drop(other);
    }

    #[test]
    fn eligible_principal_bypasses_locally_saturated_waiter() {
        let admission = Arc::new(InferenceAdmission::new(limits(2, 1)).unwrap());
        let held = admission.acquire("busy").unwrap();
        let waiter_started = Arc::new(Barrier::new(2));

        let waiting_admission = Arc::clone(&admission);
        let waiting_started = Arc::clone(&waiter_started);
        let waiting = thread::spawn(move || {
            waiting_started.wait();
            waiting_admission.acquire("busy").map(drop)
        });
        waiter_started.wait();
        admission.wait_for_waiters(1, Duration::from_secs(1));

        let eligible = admission
            .acquire_with_timeout("eligible", Duration::from_millis(50))
            .expect("an eligible principal must use free global capacity");
        assert_eq!(admission.snapshot().active, 2);
        drop(eligible);
        drop(held);
        waiting.join().unwrap().unwrap();
    }

    #[test]
    fn waiter_limit_rejects_without_growing_principal_state() {
        let mut configured = limits(1, 1);
        configured.global_waiters = 1;
        let admission = Arc::new(InferenceAdmission::new(configured).unwrap());
        let held = admission.acquire("held").unwrap();

        let waiting_admission = Arc::clone(&admission);
        let waiting = thread::spawn(move || waiting_admission.acquire("queued").map(drop));
        admission.wait_for_waiters(1, Duration::from_secs(1));
        let before = admission.snapshot().principal_entries;

        assert!(matches!(
            admission.acquire("rejected"),
            Err(AdmissionError::WaiterLimit)
        ));
        assert_eq!(admission.snapshot().principal_entries, before);

        drop(held);
        waiting.join().unwrap().unwrap();
    }

    #[test]
    fn timeout_removes_waiter_and_principal_entry() {
        let admission = InferenceAdmission::new(limits(1, 1)).unwrap();
        let held = admission.acquire("held").unwrap();

        assert!(matches!(
            admission.acquire_with_timeout("temporary", Duration::from_millis(10)),
            Err(AdmissionError::Timeout {
                blocked_by: LimitClass::Global,
            })
        ));
        let snapshot = admission.snapshot();
        assert_eq!(snapshot.waiting, 0);
        assert_eq!(snapshot.principal_entries, 1);
        drop(held);
        assert_eq!(admission.snapshot().principal_entries, 0);
    }

    #[test]
    fn permit_drop_releases_after_unwind() {
        let admission = InferenceAdmission::new(limits(1, 1)).unwrap();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _permit = admission.acquire("panic").unwrap();
            panic!("provider panicked");
        }));

        assert!(result.is_err());
        assert_eq!(admission.snapshot().active, 0);
        assert_eq!(admission.snapshot().principal_entries, 0);
        drop(admission.acquire("after").unwrap());
    }

    #[test]
    fn rejects_work_that_cannot_fit_or_start_before_its_deadline() {
        let mut configured = limits(1, 1);
        configured.max_in_flight_tokens = 128;
        configured.max_in_flight_bytes = 1_024;
        let admission = InferenceAdmission::new(configured).unwrap();

        assert!(matches!(
            admission.acquire_scheduled(AdmissionRequest {
                principal: "large",
                stream: "large-stream",
                class: LatencyClass::Offline,
                deadline: Duration::from_secs(1),
                estimated_service: Duration::from_millis(10),
                tokens: 129,
                bytes: 1,
            }),
            Err(AdmissionError::RequestBudget)
        ));

        let _held = admission.acquire("held").unwrap();
        assert!(matches!(
            admission.acquire_scheduled(AdmissionRequest {
                principal: "urgent",
                stream: "camera-7",
                class: LatencyClass::UrgentLive,
                deadline: Duration::from_millis(25),
                estimated_service: Duration::from_millis(15),
                tokens: 16,
                bytes: 32,
            }),
            Err(AdmissionError::DeadlineMissed)
        ));
        let snapshot = admission.snapshot();
        assert_eq!(snapshot.budget_rejected_total, 1);
        assert_eq!(snapshot.deadline_missed_total, 1);
    }

    #[test]
    fn urgent_live_work_precedes_queued_offline_work() {
        let admission = Arc::new(InferenceAdmission::new(limits(1, 1)).unwrap());
        let held = admission.acquire("held").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();

        let offline_admission = Arc::clone(&admission);
        let offline_tx = tx.clone();
        let offline = thread::spawn(move || {
            let permit = offline_admission
                .acquire_scheduled(AdmissionRequest {
                    principal: "offline-tenant",
                    stream: "archive-1",
                    class: LatencyClass::Offline,
                    deadline: Duration::from_secs(2),
                    estimated_service: Duration::from_millis(10),
                    tokens: 10,
                    bytes: 10,
                })
                .unwrap();
            offline_tx.send("offline").unwrap();
            drop(permit);
        });
        admission.wait_for_waiters(1, Duration::from_secs(1));

        let urgent_admission = Arc::clone(&admission);
        let urgent = thread::spawn(move || {
            let permit = urgent_admission
                .acquire_scheduled(AdmissionRequest {
                    principal: "live-tenant",
                    stream: "camera-1",
                    class: LatencyClass::UrgentLive,
                    deadline: Duration::from_secs(2),
                    estimated_service: Duration::from_millis(10),
                    tokens: 10,
                    bytes: 10,
                })
                .unwrap();
            tx.send("urgent").unwrap();
            drop(permit);
        });
        admission.wait_for_waiters(2, Duration::from_secs(1));
        drop(held);

        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), "urgent");
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), "offline");
        urgent.join().unwrap();
        offline.join().unwrap();
        let snapshot = admission.snapshot();
        assert_eq!(snapshot.urgent_acquired_total, 1);
        assert_eq!(snapshot.offline_acquired_total, 1);
    }
}
