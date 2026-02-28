#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
    Expired,
}

impl StreamState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            StreamState::Completed
                | StreamState::Failed
                | StreamState::Cancelled
                | StreamState::Expired
        )
    }
}
