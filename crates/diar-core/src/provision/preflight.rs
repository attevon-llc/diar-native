//! HuggingFace token + gate preflight.
//!
//! Runs BEFORE anything heavy: before the python subprocess is launched, before a byte is
//! downloaded, before the models dir is touched. Two HTTPS calls, ~200 ms, no python. Every
//! failure here is a message a human can act on, and because both calls are made from Rust
//! a stack trace is structurally impossible on this path — which is the acceptance
//! criterion "missing/invalid HF token fails with a message naming the gate URL, not a
//! traceback".
//!
//! ## Why two calls, and why not the obvious one
//!
//! The model-info API is useless as a gate check — measured:
//!
//! | call | no token | garbage token | valid token |
//! |---|---|---|---|
//! | `GET /api/models/pyannote/speaker-diarization-community-1` | 200 | 200 | 200 |
//! | `GET /pyannote/speaker-diarization-community-1/resolve/main/config.yaml` | 401 GatedRepo | 401 GatedRepo | 200 |
//! | `GET /api/whoami-v2` | 401 | 401 | 200 |
//!
//! So model-info cannot distinguish anything, and a file *resolve* is the only call that
//! actually exercises the gate. `whoami-v2` is what separates "your token is bad" from
//! "your token is fine but this account has not accepted the terms" — without it both
//! collapse into one 401 and the operator gets sent to fix the wrong thing.

use std::collections::BTreeMap;

/// The gated pipeline. Self-contained: `config.yaml`, `segmentation/pytorch_model.bin`,
/// `embedding/pytorch_model.bin`, `plda/plda.npz`, `plda/xvec_transform.npz`.
///
/// This is the ONLY gated repo provisioning needs. Downloading `pyannote/segmentation-3.0`
/// or `pyannote/wespeaker-voxceleb-resnet34-LM` as well would be actively wrong, not merely
/// redundant: RESULTS §1 proves by checkpoint sha256 that community-1's segmentation and
/// embedding weights DIFFER from those standalone repos, and records that a previous set of
/// ONNX artifacts exported from the standalone (fallback 3.1) checkpoints silently measured
/// a different model.
pub const PIPELINE_REPO: &str = "pyannote/speaker-diarization-community-1";
/// A file inside the gated repo. Resolving it is the discriminating call.
pub const PIPELINE_GATE_FILE: &str = "config.yaml";
/// Ungated (`gated: False`, verified) — needs no token at all.
pub const GENDER_REPO: &str = "prithivMLmods/Common-Voice-Gender-Detection";

const HF_ENDPOINT_DEFAULT: &str = "https://huggingface.co";

/// Environment variables consulted for a token, in precedence order. First one SET wins
/// (even if empty), so an explicitly-blanked `HF_TOKEN` is not silently overridden by a
/// stale `HUGGING_FACE_HUB_TOKEN` further down the list.
pub const TOKEN_ENV_VARS: &[&str] = &["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// Resolve a token from the environment. Returns the variable name too, so messages can say
/// which one was used without ever printing the value.
pub fn token_from_env() -> Option<(&'static str, String)> {
    for var in TOKEN_ENV_VARS {
        if let Ok(v) = std::env::var(var) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some((var, v));
            }
        }
    }
    None
}

/// Minimal HTTP response shape. Only what preflight needs.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    /// Lowercased header names.
    pub headers: BTreeMap<String, String>,
    pub body: String,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }
}

/// Injection point for tests. Every status path below is exercised from canned responses
/// with no network, which is the only way to test the "valid token, gate not accepted"
/// branch deterministically — it needs a real account that has NOT accepted the terms,
/// which cannot be arranged in CI.
pub trait Transport {
    fn get(&self, url: &str, token: Option<&str>) -> Result<Response, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightError {
    /// No token, or one HuggingFace rejects outright.
    TokenMissingOrInvalid { had_token: bool },
    /// Token authenticates fine; this account has not accepted the repo terms.
    GateNotAccepted { user: String },
    /// Anything else — surfaced verbatim rather than guessed at.
    Other { status: u16, detail: String },
    /// Could not reach HuggingFace at all.
    Transport(String),
}

impl PreflightError {
    /// The actionable sentence. Never contains the token.
    pub fn message(&self) -> String {
        match self {
            PreflightError::TokenMissingOrInvalid { had_token } => {
                let lead = if *had_token {
                    "Your HuggingFace token was rejected (HTTP 401)."
                } else {
                    "No HuggingFace token was supplied."
                };
                format!(
                    "{lead} A token is required because {PIPELINE_REPO} is gated. \
                     Create a read token at https://huggingface.co/settings/tokens and pass \
                     it with `--hf-token` or, preferably, the HF_TOKEN environment variable \
                     (a token on the command line is visible to every process via `ps`)."
                )
            }
            PreflightError::GateNotAccepted { user } => format!(
                "Your HuggingFace token is valid, but the account `{user}` has not accepted \
                 the terms for {PIPELINE_REPO}. Open \
                 https://huggingface.co/{PIPELINE_REPO} while signed in as `{user}`, accept \
                 the terms (they are auto-approved — it is an email-capture prompt, not a \
                 review), then re-run this command. The pipeline is CC-BY-4.0 and free; \
                 nothing is purchased and no weights are redistributed by diar-native."
            ),
            PreflightError::Other { status, detail } => format!(
                "Unexpected response from HuggingFace while checking access to \
                 {PIPELINE_REPO}: HTTP {status}{}. Check https://status.huggingface.co/ and \
                 retry.",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(" ({detail})")
                }
            ),
            PreflightError::Transport(e) => format!(
                "Could not reach HuggingFace to check access to {PIPELINE_REPO}: {e}. \
                 Provisioning needs outbound HTTPS to huggingface.co. If you are behind a \
                 proxy set HTTPS_PROXY, or set HF_ENDPOINT to a mirror."
            ),
        }
    }
}

/// What a successful preflight learned. The revision is recorded in the marker, so the
/// exact upstream commit the weights came from is pinned without a second network call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preflight {
    pub user: String,
    pub pipeline_repo: String,
    /// `x-repo-commit` from the resolve call.
    pub pipeline_revision: Option<String>,
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|v| v.trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| HF_ENDPOINT_DEFAULT.to_string())
}

/// Check the token, then the gate. `token` of `None` still runs both calls — an anonymous
/// whoami returns 401 and produces the "no token" message, which is the right one.
pub fn check(transport: &dyn Transport, token: Option<&str>) -> Result<Preflight, PreflightError> {
    let base = endpoint();

    // 1. Is the token itself good? This is what makes the second failure unambiguous.
    let whoami = transport
        .get(&format!("{base}/api/whoami-v2"), token)
        .map_err(PreflightError::Transport)?;
    if whoami.status == 401 || whoami.status == 403 {
        return Err(PreflightError::TokenMissingOrInvalid {
            had_token: token.is_some_and(|t| !t.is_empty()),
        });
    }
    if whoami.status != 200 {
        return Err(PreflightError::Other {
            status: whoami.status,
            detail: hf_error_detail(&whoami),
        });
    }
    let user = parse_user(&whoami.body).unwrap_or_else(|| "(unknown)".to_string());

    // 2. Has THIS account accepted the terms? Only a file resolve exercises the gate.
    let url = format!("{base}/{PIPELINE_REPO}/resolve/main/{PIPELINE_GATE_FILE}");
    let resolve = transport
        .get(&url, token)
        .map_err(PreflightError::Transport)?;
    match resolve.status {
        200 => Ok(Preflight {
            user,
            pipeline_repo: PIPELINE_REPO.to_string(),
            pipeline_revision: resolve.header("x-repo-commit").map(str::to_string),
        }),
        401 | 403 => {
            // The token already passed whoami, so a 401 here can only be the gate.
            if resolve
                .header("x-error-code")
                .is_some_and(|c| c.eq_ignore_ascii_case("GatedRepo"))
                || resolve.status == 403
                || resolve.status == 401
            {
                Err(PreflightError::GateNotAccepted { user })
            } else {
                Err(PreflightError::Other {
                    status: resolve.status,
                    detail: hf_error_detail(&resolve),
                })
            }
        }
        s => Err(PreflightError::Other {
            status: s,
            detail: hf_error_detail(&resolve),
        }),
    }
}

fn hf_error_detail(r: &Response) -> String {
    r.header("x-error-message")
        .map(str::to_string)
        .unwrap_or_default()
}

/// Pull `name` out of the whoami payload without a full JSON model.
fn parse_user(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v.get("name")
        .and_then(|n| n.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.get("fullname")
                .and_then(|n| n.as_str())
                .map(str::to_string)
        })
}

/// Real network transport. `ureq` rather than `reqwest`: a small dependency tree, rustls, no
/// C toolchain, and no tokio pulled into `diar-core` (which is used by the CLI too). The
/// `ort` pin history in this repo is a standing warning about dependency churn.
pub struct UreqTransport;

impl Transport for UreqTransport {
    fn get(&self, url: &str, token: Option<&str>) -> Result<Response, String> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(20))
            // Do NOT follow redirects: HuggingFace answers a successful gated resolve with a
            // redirect to CDN storage, and following it would replace the headers we need
            // (x-repo-commit) with the CDN's and download the file body for no reason.
            .redirects(0)
            .build();
        let mut req = agent.get(url).set("user-agent", "diar-native-provision/1");
        if let Some(t) = token {
            if !t.is_empty() {
                req = req.set("authorization", &format!("Bearer {t}"));
            }
        }
        let resp = match req.call() {
            Ok(r) => r,
            // ureq models >=400 as Err(Status(..)); that is a real HTTP answer, not a
            // transport failure, and preflight's whole job is to read those codes.
            Err(ureq::Error::Status(_, r)) => r,
            Err(e) => return Err(e.to_string()),
        };
        let status = resp.status();
        let mut headers = BTreeMap::new();
        for name in resp.headers_names() {
            if let Some(v) = resp.header(&name) {
                headers.insert(name.to_ascii_lowercase(), v.to_string());
            }
        }
        // A 3xx from the no-redirect agent is a SUCCESSFUL resolve: access was granted and
        // HuggingFace is pointing at storage. Normalize it so callers see 200.
        let status = if (300..400).contains(&status) { 200 } else { status };
        let body = resp.into_string().unwrap_or_default();
        Ok(Response { status, headers, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Replays canned responses in call order.
    struct Canned {
        responses: RefCell<Vec<Response>>,
        seen: RefCell<Vec<(String, bool)>>,
    }

    impl Canned {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                responses: RefCell::new(responses),
                seen: RefCell::new(Vec::new()),
            }
        }
    }

    impl Transport for Canned {
        fn get(&self, url: &str, token: Option<&str>) -> Result<Response, String> {
            self.seen
                .borrow_mut()
                .push((url.to_string(), token.is_some()));
            let mut r = self.responses.borrow_mut();
            if r.is_empty() {
                return Err("no canned response left".into());
            }
            Ok(r.remove(0))
        }
    }

    fn resp(status: u16, headers: &[(&str, &str)], body: &str) -> Response {
        Response {
            status,
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))
                .collect(),
            body: body.to_string(),
        }
    }

    fn whoami_ok(name: &str) -> Response {
        resp(200, &[], &format!(r#"{{"name":"{name}","type":"user"}}"#))
    }

    #[test]
    fn no_token_reports_a_missing_token_and_names_the_token_page() {
        let t = Canned::new(vec![resp(401, &[], "")]);
        let err = check(&t, None).unwrap_err();
        assert_eq!(err, PreflightError::TokenMissingOrInvalid { had_token: false });
        let m = err.message();
        assert!(m.contains("settings/tokens"), "{m}");
        assert!(m.contains("No HuggingFace token was supplied"), "{m}");
    }

    #[test]
    fn garbage_token_reports_rejection_not_a_gate_problem() {
        let t = Canned::new(vec![resp(401, &[], "")]);
        let err = check(&t, Some("hf_invalid")).unwrap_err();
        assert_eq!(err, PreflightError::TokenMissingOrInvalid { had_token: true });
        let m = err.message();
        assert!(m.contains("rejected"), "{m}");
        // Must NOT send the operator to accept terms — the token is the problem.
        assert!(!m.contains("accept the terms"), "{m}");
        assert!(!m.contains("hf_invalid"), "token leaked into message: {m}");
    }

    #[test]
    fn valid_token_without_the_gate_names_the_gate_url_and_the_account() {
        let t = Canned::new(vec![
            whoami_ok("dave"),
            resp(401, &[("x-error-code", "GatedRepo")], ""),
        ]);
        let err = check(&t, Some("hf_good")).unwrap_err();
        assert_eq!(err, PreflightError::GateNotAccepted { user: "dave".into() });
        let m = err.message();
        assert!(m.contains(&format!("https://huggingface.co/{PIPELINE_REPO}")), "{m}");
        assert!(m.contains("dave"), "must name the signed-in account: {m}");
        assert!(m.contains("CC-BY-4.0"), "{m}");
        assert!(!m.contains("hf_good"), "token leaked: {m}");
    }

    #[test]
    fn valid_token_with_the_gate_accepted_captures_the_revision() {
        let t = Canned::new(vec![
            whoami_ok("dave"),
            resp(
                200,
                &[("x-repo-commit", "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")],
                "",
            ),
        ]);
        let ok = check(&t, Some("hf_good")).unwrap();
        assert_eq!(ok.user, "dave");
        assert_eq!(
            ok.pipeline_revision.as_deref(),
            Some("3533c8cf8e369892e6b79ff1bf80f7b0286a54ee")
        );
    }

    #[test]
    fn the_gate_check_actually_resolves_a_file_not_the_model_info_api() {
        // The model-info API returns 200 with no token AND with a garbage token, so a
        // preflight built on it would pass for everyone and the gate error would surface
        // as a python traceback deep inside the download instead.
        let t = Canned::new(vec![whoami_ok("dave"), resp(200, &[], "")]);
        check(&t, Some("hf_good")).unwrap();
        let seen = t.seen.borrow();
        assert!(seen[0].0.ends_with("/api/whoami-v2"), "{:?}", seen[0]);
        assert!(
            seen[1].0.contains("/resolve/main/config.yaml"),
            "gate check must be a file resolve, got {:?}",
            seen[1]
        );
        assert!(!seen[1].0.contains("/api/models/"), "{:?}", seen[1]);
    }

    #[test]
    fn unexpected_status_is_surfaced_verbatim_rather_than_guessed_at() {
        let t = Canned::new(vec![
            whoami_ok("dave"),
            resp(503, &[("x-error-message", "service unavailable")], ""),
        ]);
        let err = check(&t, Some("hf_good")).unwrap_err();
        assert_eq!(
            err,
            PreflightError::Other {
                status: 503,
                detail: "service unavailable".into()
            }
        );
        assert!(err.message().contains("503"));
        assert!(err.message().contains("service unavailable"));
    }

    #[test]
    fn network_failure_is_distinguished_from_a_rejection() {
        struct Dead;
        impl Transport for Dead {
            fn get(&self, _: &str, _: Option<&str>) -> Result<Response, String> {
                Err("dns failure".into())
            }
        }
        let err = check(&Dead, Some("hf_good")).unwrap_err();
        assert!(matches!(err, PreflightError::Transport(_)));
        let m = err.message();
        assert!(m.contains("Could not reach HuggingFace"), "{m}");
        assert!(m.contains("HTTPS_PROXY"), "{m}");
    }

    #[test]
    fn token_env_precedence_is_first_set_wins() {
        // Serialized implicitly: these vars are only touched here.
        for v in TOKEN_ENV_VARS {
            std::env::remove_var(v);
        }
        assert!(token_from_env().is_none());
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "third");
        assert_eq!(token_from_env().unwrap().0, "HUGGING_FACE_HUB_TOKEN");
        std::env::set_var("HUGGINGFACE_TOKEN", "second");
        assert_eq!(token_from_env().unwrap().0, "HUGGINGFACE_TOKEN");
        std::env::set_var("HF_TOKEN", "first");
        assert_eq!(token_from_env().unwrap().0, "HF_TOKEN");
        for v in TOKEN_ENV_VARS {
            std::env::remove_var(v);
        }
    }

    #[test]
    fn whoami_name_falls_back_to_fullname() {
        assert_eq!(parse_user(r#"{"name":"a"}"#).as_deref(), Some("a"));
        assert_eq!(parse_user(r#"{"fullname":"b"}"#).as_deref(), Some("b"));
        assert_eq!(parse_user("not json"), None);
    }
}
