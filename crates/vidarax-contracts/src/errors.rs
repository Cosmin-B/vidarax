#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    BadRequest,
    Unauthorized,
    NotFound,
    PayloadTooLarge,
    Unprocessable,
    RateLimited,
    Internal,
    Unavailable,
    Unknown,
}

impl ErrorClass {
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            ErrorClass::RateLimited | ErrorClass::Internal | ErrorClass::Unavailable
        )
    }
}

pub fn classify_status_code(code: u16) -> ErrorClass {
    match code {
        400 => ErrorClass::BadRequest,
        401 => ErrorClass::Unauthorized,
        404 => ErrorClass::NotFound,
        413 => ErrorClass::PayloadTooLarge,
        422 => ErrorClass::Unprocessable,
        429 => ErrorClass::RateLimited,
        500 => ErrorClass::Internal,
        503 => ErrorClass::Unavailable,
        _ => ErrorClass::Unknown,
    }
}
