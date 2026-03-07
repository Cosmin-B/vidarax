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

    pub fn as_lowercase_str(self) -> &'static str {
        match self {
            StreamState::Pending => "pending",
            StreamState::Processing => "processing",
            StreamState::Completed => "completed",
            StreamState::Failed => "failed",
            StreamState::Cancelled => "cancelled",
            StreamState::Expired => "expired",
        }
    }
}
