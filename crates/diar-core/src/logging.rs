//! Process-wide logging setup, shared by `diar-server` and `diar-cli`.
//!
//! ## Why this lives in diar-core
//!
//! Both binaries need the *same* answers to "what does an unset `RUST_LOG` mean" and "how do I
//! ask for JSON", but they need different sinks: the server logs to **stdout** (12-factor;
//! `docker logs` and compose capture it) while the CLI logs to **stderr**, because its stdout
//! carries the machine-readable JSONL a benchmark harness parses. So the policy is shared and
//! the writer is a parameter. This module never installs anything on its own — a library that
//! grabs the global subscriber behind its caller's back is its own bug class.
//!
//! ## The default level is `info`, not "off"
//!
//! [`EnvFilter::from_default_env`] with `RUST_LOG` unset falls back to `ERROR`. That sounds
//! survivable and is not: nothing in this workspace emits at `error` level. speakrs' 40 events
//! are debug/trace/info and diar-core's 2 are `warn!`, so an `ERROR` floor is indistinguishable
//! from `off` — which is how `diar-server` shipped, silent, while looking correctly wired. A
//! sidecar that needs an environment variable set before it says anything is a quieter version
//! of that same bug, so an absent or empty `RUST_LOG` means [`DEFAULT_FILTER`] here.
//!
//! ## A malformed filter must not silence the process
//!
//! A typo in `RUST_LOG` falls back to [`DEFAULT_FILTER`] and records a warning rather than
//! aborting or (worse) muting the server. The warnings are carried on [`LogSettings`] instead of
//! being printed where they are discovered, because at that point there is no subscriber to
//! print them to; the caller emits them immediately after installing one.

use std::io::IsTerminal;

use tracing_subscriber::EnvFilter;

/// Standard filter variable. Same name and syntax the rest of the Rust ecosystem uses.
pub const FILTER_ENV: &str = "RUST_LOG";
/// `text` (default) or `json`.
pub const FORMAT_ENV: &str = "DIAR_LOG_FORMAT";
/// Applied when `RUST_LOG` is unset, empty, or unparseable.
pub const DEFAULT_FILTER: &str = "info";

/// How records are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// Human-readable single lines. The default: the common case is an operator reading
    /// `docker logs`.
    #[default]
    Text,
    /// One JSON object per line, for log aggregation when diar-native is embedded in a larger
    /// stack (OpenTranscribe) rather than run standalone.
    Json,
}

impl LogFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            LogFormat::Text => "text",
            LogFormat::Json => "json",
        }
    }

    /// Case- and whitespace-insensitive. Returns `None` for anything else so the caller can
    /// warn about the typo instead of silently picking a format the operator did not ask for.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "text" | "plain" | "human" => Some(LogFormat::Text),
            "json" => Some(LogFormat::Json),
            _ => None,
        }
    }
}

/// The resolved logging configuration plus anything questionable about how it was reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogSettings {
    pub format: LogFormat,
    /// A directive string known to parse as an [`EnvFilter`].
    pub filter: String,
    /// Non-fatal complaints, to be logged *after* a subscriber exists.
    pub warnings: Vec<String>,
}

impl LogSettings {
    /// Fresh filter per call: `EnvFilter` is not `Clone`, and building it twice (once to
    /// validate, once to install) is cheaper than the alternatives.
    pub fn env_filter(&self) -> EnvFilter {
        EnvFilter::try_new(&self.filter)
            .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
    }
}

/// Resolve settings from explicit values — the whole of the policy, with no environment access,
/// so it is directly testable.
pub fn settings_from(filter_env: Option<&str>, format_env: Option<&str>) -> LogSettings {
    let mut warnings = Vec::new();

    let filter = match filter_env.map(str::trim).filter(|s| !s.is_empty()) {
        None => DEFAULT_FILTER.to_string(),
        Some(raw) => match EnvFilter::try_new(raw) {
            Ok(_) => raw.to_string(),
            Err(e) => {
                warnings.push(format!(
                    "{FILTER_ENV}=\"{raw}\" is not a valid filter ({e}); using {DEFAULT_FILTER}"
                ));
                DEFAULT_FILTER.to_string()
            }
        },
    };

    let format = match format_env.map(str::trim).filter(|s| !s.is_empty()) {
        None => LogFormat::default(),
        Some(raw) => LogFormat::parse(raw).unwrap_or_else(|| {
            warnings.push(format!(
                "{FORMAT_ENV}=\"{raw}\" is not a known format (text|json); using text"
            ));
            LogFormat::Text
        }),
    };

    LogSettings {
        format,
        filter,
        warnings,
    }
}

/// Resolve settings from the process environment.
pub fn settings_from_env() -> LogSettings {
    settings_from(
        std::env::var(FILTER_ENV).ok().as_deref(),
        std::env::var(FORMAT_ENV).ok().as_deref(),
    )
}

/// Build the dispatcher without installing it. Exposed so tests can capture output through a
/// custom writer and assert on the bytes the real code path would have produced.
pub fn dispatch<W>(settings: &LogSettings, writer: W, ansi: bool) -> tracing::Dispatch
where
    W: for<'a> tracing_subscriber::fmt::MakeWriter<'a> + Send + Sync + 'static,
{
    let base = tracing_subscriber::fmt()
        .with_env_filter(settings.env_filter())
        .with_writer(writer)
        .with_target(true);
    match settings.format {
        // `flatten_event` lifts event fields to the top level and `with_current_span` carries
        // the per-request span's fields, so a single record is self-contained for an aggregator.
        LogFormat::Json => tracing::Dispatch::new(
            base.json()
                .flatten_event(true)
                .with_current_span(true)
                .with_span_list(false)
                .finish(),
        ),
        LogFormat::Text => tracing::Dispatch::new(base.with_ansi(ansi).finish()),
    }
}

/// Install `settings` globally against `writer` and flush any warnings gathered while resolving
/// them. Returns `Err` only if a subscriber was already installed.
pub fn init_with<W>(settings: &LogSettings, writer: W, ansi: bool) -> Result<(), String>
where
    W: for<'a> tracing_subscriber::fmt::MakeWriter<'a> + Send + Sync + 'static,
{
    tracing::dispatcher::set_global_default(dispatch(settings, writer, ansi))
        .map_err(|e| e.to_string())?;
    for warning in &settings.warnings {
        tracing::warn!("{warning}");
    }
    Ok(())
}

/// Server entry point: read the environment, log to **stdout**.
///
/// ANSI is enabled only when stdout is a terminal, so `docker logs` and file sinks get clean
/// text instead of escape sequences.
pub fn init_stdout() -> LogSettings {
    let settings = settings_from_env();
    let ansi = std::io::stdout().is_terminal();
    if let Err(e) = init_with(&settings, std::io::stdout, ansi) {
        // Pre-subscriber failure: stderr is the only sink that certainly works.
        eprintln!("warning: could not install the log subscriber: {e}");
    }
    settings
}

/// CLI entry point: read the environment, log to **stderr** so stdout stays parseable JSONL.
pub fn init_stderr() -> LogSettings {
    let settings = settings_from_env();
    let ansi = std::io::stderr().is_terminal();
    if let Err(e) = init_with(&settings, std::io::stderr, ansi) {
        eprintln!("warning: could not install the log subscriber: {e}");
    }
    settings
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// Collects everything written, so a test can assert on the exact bytes the configured
    /// subscriber emits rather than on the builder that was supposed to configure it.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Capture;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Emit a fixed set of records through `settings` and return what was written.
    fn emit(settings: &LogSettings) -> String {
        let capture = Capture::default();
        let dispatch = dispatch(settings, capture.clone(), false);
        tracing::dispatcher::with_default(&dispatch, || {
            tracing::info!(num_speakers = 3, segments = 42, "diarize complete");
            tracing::debug!(target: "speakrs::pipeline", stage = "clustering", "clustering done");
            tracing::trace!(target: "speakrs::pipeline", "trace detail");
        });
        capture.text()
    }

    #[test]
    fn unset_rust_log_defaults_to_info_not_silence() {
        // The bug this module exists to fix: with no RUST_LOG the server emitted nothing.
        let settings = settings_from(None, None);
        assert_eq!(settings.filter, DEFAULT_FILTER);
        assert!(settings.warnings.is_empty());
        let out = emit(&settings);
        assert!(out.contains("diarize complete"), "info was dropped: {out:?}");
        // ...but info is a *floor*, not "everything": debug/trace stay off by default.
        assert!(!out.contains("clustering done"), "debug leaked at default: {out:?}");
    }

    /// Characterization test for the thing this module exists to avoid.
    ///
    /// `EnvFilter::from_default_env()` with `RUST_LOG` unset falls back to `ERROR` — not to
    /// nothing, but to a level **no code in this workspace emits**. speakrs' 40 events are
    /// debug/trace/info and diar-core's 2 are `warn!`, so the observable result in production
    /// was still total silence. That is what `diar-server` shipped with. Pinned here so nobody
    /// "simplifies" [`settings_from`] back into it without this failing first.
    #[test]
    fn from_default_env_drops_every_level_this_workspace_actually_emits() {
        assert!(
            std::env::var(FILTER_ENV).is_err(),
            "this test is only meaningful with {FILTER_ENV} unset"
        );
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_writer(capture.clone())
            .with_ansi(false)
            .finish();
        tracing::dispatcher::with_default(&tracing::Dispatch::new(subscriber), || {
            tracing::warn!("gender requested but the model is not deployed"); // diar-core
            tracing::info!("diarize complete"); // the request line
            tracing::debug!(target: "speakrs::pipeline", "clustering done"); // speakrs
        });
        assert_eq!(
            capture.text(),
            "",
            "from_default_env started passing a level this workspace emits"
        );
    }

    #[test]
    fn an_empty_rust_log_is_treated_as_unset() {
        // `RUST_LOG=${RUST_LOG:-}` in a compose file expands to exactly this.
        for raw in ["", "   "] {
            let settings = settings_from(Some(raw), None);
            assert_eq!(settings.filter, DEFAULT_FILTER, "raw={raw:?}");
            assert!(settings.warnings.is_empty(), "raw={raw:?}");
        }
    }

    #[test]
    fn rust_log_is_honored_and_reaches_speakrs_targets() {
        let settings = settings_from(Some("speakrs=debug"), None);
        assert_eq!(settings.filter, "speakrs=debug");
        let out = emit(&settings);
        // The documented incantation must actually surface speakrs' pipeline events.
        assert!(out.contains("clustering done"), "speakrs debug missing: {out:?}");
        assert!(!out.contains("trace detail"), "trace leaked at debug: {out:?}");
    }

    #[test]
    fn a_malformed_rust_log_warns_and_falls_back_instead_of_going_silent() {
        let settings = settings_from(Some("speakrs=nonsense"), None);
        assert_eq!(settings.filter, DEFAULT_FILTER);
        assert_eq!(settings.warnings.len(), 1, "{:?}", settings.warnings);
        assert!(settings.warnings[0].contains(FILTER_ENV));
        // The point of the fallback: a typo must not mute the process.
        assert!(emit(&settings).contains("diarize complete"));
    }

    #[test]
    fn log_format_parses_the_documented_spellings() {
        assert_eq!(LogFormat::parse("json"), Some(LogFormat::Json));
        assert_eq!(LogFormat::parse(" JSON "), Some(LogFormat::Json));
        assert_eq!(LogFormat::parse("text"), Some(LogFormat::Text));
        assert_eq!(LogFormat::parse("Text"), Some(LogFormat::Text));
        assert_eq!(LogFormat::parse("yaml"), None);
        assert_eq!(LogFormat::parse(""), None);
    }

    #[test]
    fn format_defaults_to_text_and_an_unknown_one_warns() {
        assert_eq!(settings_from(None, None).format, LogFormat::Text);
        assert_eq!(settings_from(None, Some("")).format, LogFormat::Text);
        assert_eq!(settings_from(None, Some("json")).format, LogFormat::Json);

        let settings = settings_from(None, Some("logfmt"));
        assert_eq!(settings.format, LogFormat::Text);
        assert_eq!(settings.warnings.len(), 1, "{:?}", settings.warnings);
        assert!(settings.warnings[0].contains(FORMAT_ENV));
    }

    #[test]
    fn json_mode_emits_one_parseable_object_per_line() {
        let settings = settings_from(Some("info"), Some("json"));
        let out = emit(&settings);
        let mut lines = 0;
        for line in out.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("not JSON: {line:?} ({e})"));
            assert_eq!(v["level"], "INFO");
            assert_eq!(v["message"], "diarize complete");
            // flatten_event: structured fields are top-level, not stringified into the message.
            assert_eq!(v["num_speakers"], 3);
            assert_eq!(v["segments"], 42);
            lines += 1;
        }
        assert_eq!(lines, 1, "expected exactly one record, got {out:?}");
    }

    #[test]
    fn text_mode_is_not_json() {
        let out = emit(&settings_from(Some("info"), None));
        let line = out.lines().next().expect("no output");
        assert!(serde_json::from_str::<serde_json::Value>(line).is_err());
        assert!(line.contains("INFO"), "{line:?}");
        assert!(line.contains("diarize complete"), "{line:?}");
    }

    #[test]
    fn text_mode_has_no_ansi_escapes_when_ansi_is_off() {
        // Containers are not terminals; escape codes in `docker logs` are pure noise.
        let capture = Capture::default();
        let settings = settings_from(Some("info"), None);
        tracing::dispatcher::with_default(&dispatch(&settings, capture.clone(), false), || {
            tracing::info!("hello");
        });
        assert!(!capture.text().contains('\u{1b}'), "{:?}", capture.text());
    }

    #[test]
    fn json_mode_carries_span_fields_for_request_correlation() {
        // The per-request span is what makes a speakrs event attributable to a caller's job.
        let capture = Capture::default();
        let settings = settings_from(Some("info"), Some("json"));
        tracing::dispatcher::with_default(&dispatch(&settings, capture.clone(), false), || {
            let span = tracing::info_span!("request", request_id = "abc123", endpoint = "/diarize");
            span.in_scope(|| tracing::info!(outcome = "ok", "request complete"));
        });
        let out = capture.text();
        let line = out.lines().find(|l| l.contains("request complete")).unwrap();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["span"]["request_id"], "abc123");
        assert_eq!(v["span"]["endpoint"], "/diarize");
        assert_eq!(v["outcome"], "ok");
    }
}
