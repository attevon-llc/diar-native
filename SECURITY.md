# Security Policy

## Reporting a vulnerability

**Do not open a public GitHub issue for a security problem.**

Report privately, whichever is easier for you:

1. **GitHub private vulnerability reporting** — the *Security* tab of
   [`attevon-llc/diar-native`](https://github.com/attevon-llc/diar-native/security/advisories),
   then *Report a vulnerability*. This is preferred: it keeps the report, the fix and the
   advisory in one place.
2. **Email** — <davidamacey@gmail.com>, with `diar-native security` in the subject.

Please include enough to reproduce: affected version or image tag (e.g.
`davidamacey/diar-native:0.3.1`),
the request or input that triggers it, what you expected, what happened, and the impact you
believe it has. A minimal reproducer is worth more than a long description.

**What to expect:** acknowledgement within 3 business days, an assessment with a rough timeline
within 10 business days, and credit in the release notes when a fix ships — unless you would
rather stay anonymous, which is fine. Please give us a reasonable window to ship a fix before
disclosing publicly.

## Supported versions

This project ships as a container image rather than a versioned library. Only the **latest
released image** receives security fixes — currently **0.3.1**
(`davidamacey/diar-native:0.3.1` and the `-cpu`, `-cpu-arm64` and `-provision` variants; see
`docs/DEPLOYMENT.md` for digests, all scanned at 0 HIGH / 0 CRITICAL).

Note that the image and the **deployed binary** are not the same thing here: OpenTranscribe
copies `diar-server` out of this image at build time, so the sidecar running there is whatever
its pinned digest last provided, not necessarily the latest release. See `CHANGELOG.md`.

## Scope

In scope:

- The `diar-server` HTTP surface (`/diarize`, `/embed_window`, `/healthz`, `/readyz`) — including
  request parsing, path handling for `audio_path`, and resource-exhaustion paths.
- Media decoding (`crates/diar-core/src/audio.rs`, via `symphonia`) on untrusted input.
- The model provisioning path (`provision-models`, `verify-models`, `check-token`) — token
  handling, download verification, and the integrity check.
- The published container images and their build.

Out of scope:

- Findings that require the attacker to already have code execution or filesystem access on the
  host running the sidecar.
- Vulnerabilities in upstream [`avencera/speakrs`](https://github.com/avencera/speakrs), in
  `onnxruntime`, or in pyannote's models — report those to their maintainers. Tell us anyway if
  our vendored patch set changes the exposure.
- Anything requiring a model file the operator did not choose to install.
- Benchmark and validation scripts under `validation/`, which are developer tooling.

## Deployment expectations

`diar-server` is designed as an **internal sidecar**, not an internet-facing service. It has no
authentication, no authorization and no rate limiting of its own, by design — it is meant to sit
on a private compose network reachable only by the application that owns it. Exposing it
directly to untrusted clients is a deployment mistake, not a vulnerability in the service; put it
behind your application's auth boundary. `DIAR_MAX_INFLIGHT` bounds concurrency and is your main
resource-exhaustion control.

Note that `audio_path` reads a path from the request on the server's filesystem. That is
intentional for the trusted-sidecar model, and it is another reason the port must not be exposed.

## Model weights and Hugging Face tokens

The diarization models are **terms-gated derivatives of pyannote
`speaker-diarization-community-1`**. They are the operator's responsibility, and this project's
handling of them is deliberately narrow:

- **No weights are distributed here.** They are not in the repository, not in git history, and
  not baked into any published image. `.gitignore` excludes every `models*/` tree and
  `.dockerignore` is an allowlist so a new directory cannot leak into a build context by
  accident. Operators accept the upstream terms and provision the weights themselves.
- **`HF_TOKEN` is read from the environment only.** It is used to authenticate downloads from
  Hugging Face and for nothing else. It is never written to disk, never recorded in the
  provisioning marker, and never sent anywhere but Hugging Face. Log records must not contain it
  — if you find a code path that logs, echoes or persists a token, that is a valid security
  report under this policy.
- **CI never sees a token and never downloads weights.** Every test that needs model artifacts is
  `#[ignore]`d and gated behind an explicit environment variable, so no workflow can be induced
  into fetching gated material. Pull-request workflows have no access to repository secrets.
- Supply a token at runtime via your orchestrator's secret mechanism (compose `secrets`, a
  Kubernetes secret, `--env-file` outside version control). Do not put it in a committed compose
  file, a Dockerfile `ARG`/`ENV`, or a shell history you keep.

Provisioning verifies downloaded artifacts before use; a failed verification fails the startup
gate rather than falling back to a silently different model.

## Dependency handling

- `ort` is pinned at `=2.0.0-rc.12` for a correctness reason (rc.13 fails at session load), so
  automated dependency updates are configured to leave it alone. A security advisory against
  `ort` therefore needs a human decision, not a bot bump — report it and we will handle it
  explicitly.
- `speakrs` is vendored from our fork at a pinned commit via
  `scripts/bootstrap_vendor_speakrs.sh`, with local changes carried as
  `patches/0001-cuda-performance-patch-set.patch`. Security-relevant changes there are pulled in
  by moving the pin, not by editing the vendored tree in place without a regenerated patch.
