//! Per-request observability for `/diarize` and `/embed_window`.
//!
//! A sidecar is only debuggable from the *caller's* side if a job the caller sent can be found
//! in the sidecar's log. So every request gets an id, and that id is both the parent span for
//! all work the request does — including speakrs' own pipeline events, which run on a
//! `spawn_blocking` thread and only inherit the span because the closure re-enters it — and a
//! response header the caller can record.
//!
//! ## Ids come from the caller when the caller has one
//!
//! `x-request-id` is honoured if present, so a job traced through OpenTranscribe keeps one id
//! end to end. That makes it attacker-influenced input that lands in a log file, so it is
//! sanitized (see [`sanitize_id`]) rather than trusted: a newline in a header must not be able
//! to forge a log record.
//!
//! ## What is deliberately NOT logged
//!
//! Never the full media path — it is a user's filename on a shared volume. The basename is the
//! most that goes in, and only because "which file failed" is the first question anyone asks.
//!
//! Keeping the *span field* to a basename is not sufficient on its own, and assuming otherwise
//! was a live bug: the underlying I/O errors interpolate the path they were given, so a failed
//! decode logged `opening /audio/private/Board Meeting Q3 - CONFIDENTIAL.wav: No such file`
//! through the `error` field while the `audio` field dutifully said `smoke.wav`. Error text is
//! therefore redacted through [`RequestLog::redact`] before it is logged. The HTTP *response*
//! keeps the full path — the caller supplied it, so it is not a disclosure to them, and
//! trimming it would break error handling that already parses these messages.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use axum::http::HeaderMap;
use tracing::field::Empty;

/// Header the id is read from and echoed back on.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Cap on an inbound id. Long enough for a UUID or a trace id, short enough that a caller
/// cannot pad the log with kilobytes per request.
const MAX_ID_LEN: usize = 64;

/// Keep an inbound id only insofar as it is safe to write to a log.
///
/// Control characters — newlines above all — are dropped rather than escaped, because an id is
/// an opaque token and a caller with a legitimate one loses nothing. Everything outside the
/// conservative token alphabet goes too. An id that is empty after filtering is treated as
/// absent, so a hostile or malformed header degrades to a generated id instead of an error.
fn sanitize_id(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
        .take(MAX_ID_LEN)
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// Per-process prefix, so ids from two runs of the container do not collide in an aggregator
/// that only sees the sequence number.
fn process_nonce() -> &'static str {
    static NONCE: OnceLock<String> = OnceLock::new();
    NONCE.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        format!("{:06x}", (nanos ^ std::process::id() as u64) & 0xff_ffff)
    })
}

fn generate_id() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{:06}",
        process_nonce(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    )
}

/// How a request ended. Drives both the fields logged and the level it is logged at.
pub enum Outcome {
    Diarize {
        num_speakers: usize,
        segments: usize,
    },
    Embed {
        dim: usize,
    },
    /// `class` is the machine-readable failure kind (`bad_device`, `audio_decode`, …); it is
    /// what makes "is this broken or is the caller wrong" answerable from a log query.
    Failed {
        class: &'static str,
        status: u16,
        message: String,
    },
}

/// One in-flight request. Created before admission so queueing time is inside the measurement.
pub struct RequestLog {
    id: String,
    endpoint: &'static str,
    span: tracing::Span,
    started: Instant,
    /// `(full path, basename)` for this request, once known. `OnceLock` because it is set once,
    /// from `&self`, and the log then crosses a `spawn_blocking` boundary — so it has to be
    /// `Send + Sync` without a lock on the hot path.
    audio_path: OnceLock<(String, String)>,
}

impl RequestLog {
    pub fn new(endpoint: &'static str, headers: &HeaderMap) -> Self {
        let id = headers
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(sanitize_id)
            .unwrap_or_else(generate_id);
        // Fields filled in later are declared Empty up front: a span's field set is fixed at
        // creation, so anything not declared here can never be recorded.
        let span = tracing::info_span!(
            "request",
            request_id = %id,
            endpoint,
            device = Empty,
            audio = Empty,
            gender = Empty,
        );
        Self {
            id,
            endpoint,
            span,
            started: Instant::now(),
            audio_path: OnceLock::new(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Clone into a `spawn_blocking` closure and `in_scope` it there, so engine-side events land
    /// under this request. Never hold an `Entered` guard across an `.await`.
    pub fn span(&self) -> tracing::Span {
        self.span.clone()
    }

    pub fn set_device(&self, device: &str) {
        self.span.record("device", device);
    }

    /// Record the request's media path. Only its **basename** reaches the span; the full path is
    /// retained solely so [`Self::redact`] can strip it back out of error text.
    pub fn set_audio(&self, path: &Path) {
        let Some(base) = path.file_name().map(|s| s.to_string_lossy().into_owned()) else {
            return;
        };
        self.span.record("audio", base.as_str());
        let full = path.to_string_lossy().into_owned();
        if full != base {
            let _ = self.audio_path.set((full, base));
        }
    }

    /// Replace this request's full media path with its basename wherever it appears. Applied to
    /// every message that gets logged, because the path is interpolated into I/O errors far from
    /// here and there is no practical way to stop that at the source.
    fn redact(&self, message: &str) -> String {
        match self.audio_path.get() {
            Some((full, base)) => message.replace(full.as_str(), base),
            None => message.to_string(),
        }
    }

    pub fn set_gender(&self, gender: bool) {
        self.span.record("gender", gender);
    }

    /// Emit the terminal record. Levels: success and client errors are normal operation
    /// (`info`/`warn`); only a 5xx means the sidecar itself misbehaved (`error`).
    pub fn finish(self, outcome: Outcome) {
        let ms = self.started.elapsed().as_secs_f64() * 1000.0;
        let ms = (ms * 10.0).round() / 10.0;
        let endpoint = self.endpoint;
        let outcome = match outcome {
            // Redact BEFORE the record is built, so there is no window in which the raw path
            // reaches a subscriber.
            Outcome::Failed {
                class,
                status,
                message,
            } => Outcome::Failed {
                class,
                status,
                message: self.redact(&message),
            },
            other => other,
        };
        self.span.in_scope(|| match outcome {
            Outcome::Diarize {
                num_speakers,
                segments,
            } => tracing::info!(
                outcome = "ok",
                duration_ms = ms,
                num_speakers,
                segments,
                "{endpoint} ok"
            ),
            Outcome::Embed { dim } => tracing::info!(
                outcome = "ok",
                duration_ms = ms,
                embedding_dim = dim,
                "{endpoint} ok"
            ),
            Outcome::Failed {
                class,
                status,
                message,
            } => {
                if status >= 500 {
                    tracing::error!(
                        outcome = "error",
                        duration_ms = ms,
                        error_class = class,
                        status,
                        error = %message,
                        "{endpoint} failed"
                    );
                } else {
                    tracing::warn!(
                        outcome = "error",
                        duration_ms = ms,
                        error_class = class,
                        status,
                        error = %message,
                        "{endpoint} failed"
                    );
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    fn headers(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        // from_bytes, so tests can push values a well-behaved client would never send.
        if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
            h.insert(HeaderName::from_static(REQUEST_ID_HEADER), v);
        }
        h
    }

    #[test]
    fn a_caller_supplied_id_is_kept_so_traces_join_end_to_end() {
        let log = RequestLog::new("/diarize", &headers("0f8c1a2b-4d5e-6f70-8192-a3b4c5d6e7f8"));
        assert_eq!(log.id(), "0f8c1a2b-4d5e-6f70-8192-a3b4c5d6e7f8");
    }

    #[test]
    fn a_generated_id_is_used_when_the_caller_supplies_none() {
        let a = RequestLog::new("/diarize", &HeaderMap::new());
        let b = RequestLog::new("/diarize", &HeaderMap::new());
        assert_ne!(a.id(), b.id(), "ids must not repeat within a process");
        assert!(!a.id().is_empty());
        assert!(a
            .id()
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-'));
    }

    #[test]
    fn control_characters_cannot_be_smuggled_into_the_log() {
        // A newline in an id would let a caller forge a whole log record.
        assert_eq!(sanitize_id("ab\ncd"), Some("abcd".to_string()));
        assert_eq!(
            sanitize_id("a\r\nINFO forged"),
            Some("aINFOforged".to_string())
        );
        assert_eq!(sanitize_id("a\u{1b}[31mred"), Some("a31mred".to_string()));
        assert!(!sanitize_id("x\ty").unwrap().contains('\t'));
    }

    #[test]
    fn an_unusable_id_falls_back_to_a_generated_one_rather_than_failing_the_request() {
        assert_eq!(sanitize_id(""), None);
        assert_eq!(sanitize_id("!!!"), None);
        let log = RequestLog::new("/diarize", &headers("   "));
        assert!(
            log.id().contains('-'),
            "expected a generated id, got {}",
            log.id()
        );
    }

    #[test]
    fn an_oversized_id_is_truncated_not_rejected() {
        let long = "a".repeat(500);
        let id = sanitize_id(&long).unwrap();
        assert_eq!(id.len(), MAX_ID_LEN);
    }

    /// Regression: verbatim from a live run before the fix. The span field said `smoke.wav`
    /// while the error field published the whole path, which is exactly the disclosure the
    /// basename rule exists to prevent.
    #[test]
    fn an_error_message_cannot_republish_the_full_media_path() {
        let log = RequestLog::new("/diarize", &HeaderMap::new());
        log.set_audio(Path::new(
            "/audio/private/Board Meeting Q3 - CONFIDENTIAL.wav",
        ));
        let leaked = "opening /audio/private/Board Meeting Q3 - CONFIDENTIAL.wav: \
                      No such file or directory (os error 2)";
        let redacted = log.redact(leaked);
        assert!(!redacted.contains("/audio/private"), "{redacted}");
        assert!(!redacted.contains("private"), "{redacted}");
        // The basename survives: "which file failed" must still be answerable.
        assert!(
            redacted.contains("Board Meeting Q3 - CONFIDENTIAL.wav"),
            "{redacted}"
        );
        assert!(redacted.starts_with("opening Board Meeting"), "{redacted}");
    }

    #[test]
    fn redaction_is_a_no_op_when_there_is_no_path() {
        // /embed_window with samples_b64 has no path at all.
        let log = RequestLog::new("/embed_window", &HeaderMap::new());
        assert_eq!(log.redact("invalid base64"), "invalid base64");
    }

    #[test]
    fn a_bare_filename_needs_no_redaction_and_is_not_mangled() {
        let log = RequestLog::new("/diarize", &HeaderMap::new());
        log.set_audio(Path::new("smoke.wav"));
        assert_eq!(
            log.redact("opening smoke.wav: boom"),
            "opening smoke.wav: boom"
        );
    }

    #[test]
    fn a_generated_id_is_prefixed_per_process() {
        let id = generate_id();
        let (prefix, seq) = id.split_once('-').expect("id has a nonce prefix");
        assert_eq!(prefix.len(), 6, "{id}");
        assert!(prefix.chars().all(|c| c.is_ascii_hexdigit()), "{id}");
        assert!(seq.chars().all(|c| c.is_ascii_digit()), "{id}");
    }
}
