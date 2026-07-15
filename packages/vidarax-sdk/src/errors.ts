/**
 * Typed error classes for the Vidarax SDK.
 *
 * Every error thrown by the SDK is an instance of `VidaraxError` or one of its
 * subclasses, so callers can use `instanceof` checks reliably.
 */

// ─── Structured API error shape ───────────────────────────────────────────────

/** A field-level validation detail returned in the `error.details` array. */
export interface FieldError {
  field: string;
  message: string;
}

/** The `error` object returned in Vidarax API error responses. */
export interface ApiErrorBody {
  code: string;
  message: string;
  request_id?: string;
  details?: FieldError[];
}

// ─── Base error ───────────────────────────────────────────────────────────────

/**
 * Base class for all SDK errors.
 *
 * Extends the native `Error` so stack traces are always present.
 */
export class VidaraxError extends Error {
  /** Stable machine-readable code, e.g. `"network_error"` or `"validation_error"`. */
  readonly code: string;

  constructor(message: string, code = "vidarax_error") {
    super(message);
    this.name = "VidaraxError";
    this.code = code;
    // Maintain correct prototype chain in transpiled environments.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── HTTP / API errors ────────────────────────────────────────────────────────

/**
 * Raised when the server returns a non-2xx HTTP status.
 *
 * The full structured error body is attached as `apiError` when the server
 * returned a JSON `{ "error": … }` envelope. WHIP, file-serving, malformed
 * body, and unknown-route failures may instead have a plain response body.
 */
export class HttpError extends VidaraxError {
  readonly status: number;
  readonly apiError: ApiErrorBody | null;

  constructor(status: number, message: string, apiError: ApiErrorBody | null = null) {
    const code = apiError?.code ?? "http_error";
    super(message, code);
    this.name = "HttpError";
    this.status = status;
    this.apiError = apiError;
    Object.setPrototypeOf(this, new.target.prototype);
  }

  /** Convenience: true for 4xx client errors. */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  /** Convenience: true for 5xx server errors. */
  get isServerError(): boolean {
    return this.status >= 500;
  }

  /** Convenience: true for 404. */
  get isNotFound(): boolean {
    return this.status === 404;
  }

  /** Convenience: true for 422. */
  get isValidationError(): boolean {
    return this.status === 422;
  }

  /** Convenience: true for 409. */
  get isConflict(): boolean {
    return this.status === 409;
  }
}

// ─── Network errors ───────────────────────────────────────────────────────────

/**
 * Raised when a fetch fails due to network connectivity or timeout, before
 * any HTTP response is received.
 */
export class NetworkError extends VidaraxError {
  override readonly cause: unknown;

  constructor(message: string, cause?: unknown) {
    super(message, "network_error");
    this.name = "NetworkError";
    this.cause = cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Retry exhaustion ─────────────────────────────────────────────────────────

/**
 * Raised after all retry attempts have been exhausted.
 *
 * The `lastError` property holds the underlying error from the final attempt.
 */
export class RetryExhaustedError extends VidaraxError {
  readonly attempts: number;
  readonly lastError: VidaraxError;

  constructor(attempts: number, lastError: VidaraxError) {
    super(
      `Request failed after ${attempts} attempt${attempts === 1 ? "" : "s"}: ${lastError.message}`,
      "retry_exhausted",
    );
    this.name = "RetryExhaustedError";
    this.attempts = attempts;
    this.lastError = lastError;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Upload errors ────────────────────────────────────────────────────────────

/** Raised when a file upload operation fails. */
export class UploadError extends VidaraxError {
  constructor(message: string) {
    super(message, "upload_error");
    this.name = "UploadError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Parse errors ────────────────────────────────────────────────────────────

/** Raised when the SDK cannot parse a response from the server. */
export class ParseError extends VidaraxError {
  readonly raw: string;

  constructor(message: string, raw: string) {
    super(message, "parse_error");
    this.name = "ParseError";
    this.raw = raw;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Type guard helpers ───────────────────────────────────────────────────────

/** Returns true if `err` is any `VidaraxError`. */
export function isVidaraxError(err: unknown): err is VidaraxError {
  return err instanceof VidaraxError;
}

/** Returns true if `err` is an `HttpError`. */
export function isHttpError(err: unknown): err is HttpError {
  return err instanceof HttpError;
}

/** Returns true if `err` is a `NetworkError`. */
export function isNetworkError(err: unknown): err is NetworkError {
  return err instanceof NetworkError;
}
