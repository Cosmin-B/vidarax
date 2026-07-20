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
        Ok(self)
    }
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
}

#[derive(Debug, Default)]
struct AdmissionState {
    active: usize,
    principals: HashMap<Box<str>, PrincipalCounts>,
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
}

#[derive(Debug, Default)]
struct AdmissionMetrics {
    acquired_total: AtomicU64,
    timeout_global_total: AtomicU64,
    timeout_principal_total: AtomicU64,
    waiter_rejected_total: AtomicU64,
    wait_duration_us: AtomicU64,
    wait_duration_count: AtomicU64,
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
        let started = Instant::now();
        let deadline = started.checked_add(timeout).unwrap_or(started);
        let mut state = self.lock_state();

        if self.can_acquire(&state, principal, None) {
            self.mark_acquired(&mut state, principal, started.elapsed());
            return Ok(AdmissionPermit {
                admission: self,
                principal,
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
        state.waiters.push_back(Waiter {
            ticket,
            principal: principal.into(),
        });
        state
            .principals
            .entry(principal.into())
            .or_default()
            .waiting += 1;
        self.changed.notify_all();

        loop {
            if self.can_acquire(&state, principal, Some(ticket)) {
                self.remove_waiter(&mut state, ticket, principal);
                self.mark_acquired(&mut state, principal, started.elapsed());
                return Ok(AdmissionPermit {
                    admission: self,
                    principal,
                    released: false,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                let blocked_by = self.blocked_by(&state, principal);
                self.remove_waiter(&mut state, ticket, principal);
                self.record_timeout(blocked_by, started.elapsed());
                drop(state);
                self.changed.notify_all();
                return Err(AdmissionError::Timeout { blocked_by });
            }

            let remaining = deadline.saturating_duration_since(now);
            let (next, _) = self
                .changed
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
        own_ticket: Option<u64>,
    ) -> bool {
        if !self.has_capacity(state, principal) {
            return false;
        }

        for waiter in &state.waiters {
            if Some(waiter.ticket) == own_ticket {
                return true;
            }
            if self.has_capacity(state, &waiter.principal) {
                return false;
            }
        }

        own_ticket.is_none()
    }

    fn has_capacity(&self, state: &AdmissionState, principal: &str) -> bool {
        state.active < self.limits.global_in_flight
            && state
                .principals
                .get(principal)
                .map_or(0, |counts| counts.active)
                < self.limits.per_principal_in_flight
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

    fn mark_acquired(&self, state: &mut AdmissionState, principal: &str, waited: Duration) {
        state.active += 1;
        state.principals.entry(principal.into()).or_default().active += 1;
        self.metrics.acquired_total.fetch_add(1, Ordering::Relaxed);
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

    fn release(&self, principal: &str) {
        let mut state = self.lock_state();
        state.active = state.active.saturating_sub(1);
        if let Some(counts) = state.principals.get_mut(principal) {
            counts.active = counts.active.saturating_sub(1);
            if counts.active == 0 && counts.waiting == 0 {
                state.principals.remove(principal);
            }
        }
        drop(state);
        self.changed.notify_all();
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
    released: bool,
}

impl Drop for AdmissionPermit<'_> {
    fn drop(&mut self) {
        if !self.released {
            self.admission.release(self.principal);
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

    use super::{AdmissionError, AdmissionLimits, InferenceAdmission, LimitClass};

    fn limits(global: usize, per_principal: usize) -> AdmissionLimits {
        AdmissionLimits {
            global_in_flight: global,
            per_principal_in_flight: per_principal,
            global_waiters: 8,
            wait_timeout: Duration::from_secs(1),
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
}
