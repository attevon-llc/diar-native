//! Execution-device registry: which backends this build *can* serve, which are actually
//! loaded, and how a request selects between them.
//!
//! Why this exists: the CUDA image is a **superset** of the CPU image on amd64. The ORT CPU
//! execution provider is statically linked into the binary by `ort-sys` (the `onnxruntime_mlas`
//! kernel library is part of the core static objects), and `ort/cuda` only swaps in a different
//! prebuilt distribution — it is purely additive. speakrs agrees: `ExecutionMode::Cpu.validate()`
//! returns `Ok(())` unconditionally and `with_execution_mode` registers the CPU EP with no
//! feature gate. So a `--features cuda` build has always been able to run CPU inference; it just
//! had no way to *ask* for it after startup. This module is that way.
//!
//! ## Engine loads no longer have to be single-threaded (issue #3)
//!
//! [`diar_core::DiarEngine::load`] used to call `std::env::set_var("SPEAKRS_FBANK_POOL", ..)` and
//! have speakrs read it back inside the same call. glibc `setenv`/`getenv` is not thread-safe, so
//! that made "load on first use" unsound rather than merely unimplemented — and it was already
//! shakier than it looked here, since loading device *N+1* ran `setenv` while device *N*'s ORT
//! intra-op threads were alive.
//!
//! The pool size now travels as a [`diar_core::EngineConfig`] field into speakrs' `RuntimeConfig`,
//! so `DiarEngine::load` touches no process-global state. Loads may safely be lazy or concurrent.
//! They are still done here, serially, from `run()` before `axum::serve` — but that is now a
//! deliberate fail-fast choice (a bad `DIAR_DEVICES` should kill startup, not the first request),
//! not a soundness requirement. Lazy loading is unblocked: a resident CPU engine costs ~620 MB
//! RSS (RESULTS §7.34).

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};

use anyhow::Context;
use diar_core::{DiarEngine, EngineConfig, Mode};

/// A backend a request can be routed to. One-to-one with [`diar_core::Mode`]; kept separate so
/// the wire names, the capability list and the parse errors live with the server that serves
/// them rather than in the engine crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    CoreMl,
    CoreMlFast,
}

/// Display and selection order: accelerators first, CPU last. Used for the capability list in
/// `/healthz` and in error messages, so it should read the way an operator expects.
const CANONICAL: [Device; 4] = [
    Device::Cuda,
    Device::CoreMl,
    Device::CoreMlFast,
    Device::Cpu,
];

impl Device {
    /// Wire name. These are the strings accepted in the request `device` field, reported by
    /// `/healthz`, and accepted in `DIAR_DEVICES` / `DIAR_MODE` — deliberately one vocabulary.
    pub const fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::CoreMl => "coreml",
            Device::CoreMlFast => "coreml_fast",
        }
    }

    pub const fn to_mode(self) -> Mode {
        match self {
            Device::Cpu => Mode::Cpu,
            Device::Cuda => Mode::Cuda,
            Device::CoreMl => Mode::CoreMl,
            Device::CoreMlFast => Mode::CoreMlFast,
        }
    }

    /// The Cargo feature this device needs, or `None` when it is always available. CPU is
    /// unconditional: the CPU EP is statically linked, so it costs no feature and no bytes.
    pub const fn required_feature(self) -> Option<&'static str> {
        match self {
            Device::Cpu => None,
            Device::Cuda => Some("cuda"),
            Device::CoreMl | Device::CoreMlFast => Some("coreml"),
        }
    }

    /// Whether *this build* can serve the device at all — a compile-time fact, independent of
    /// what is loaded or whether a GPU is present at runtime.
    pub fn is_compiled_in(self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Cuda => cfg!(feature = "cuda"),
            Device::CoreMl | Device::CoreMlFast => cfg!(feature = "coreml"),
        }
    }

    /// Name lookup only — does not consider whether this build can serve the device.
    /// Use [`FromStr`] for the capability-checked version.
    pub fn from_name(name: &str) -> Option<Device> {
        CANONICAL.iter().copied().find(|d| d.as_str() == name)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Device {
    type Err = DeviceError;

    /// Parse *and* capability-check. A device this build cannot serve is rejected here, at parse
    /// time, from the compile-time capability list — never attempted and then failed deep inside
    /// a session builder.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match Device::from_name(name) {
            None => Err(DeviceError::Unknown(name.to_string())),
            Some(device) if !device.is_compiled_in() => Err(DeviceError::NotCompiledIn(device)),
            Some(device) => Ok(device),
        }
    }
}

/// Every device this build can serve, in [`CANONICAL`] order. Always contains [`Device::Cpu`].
pub fn supported() -> &'static [Device] {
    static SUPPORTED: OnceLock<Vec<Device>> = OnceLock::new();
    SUPPORTED.get_or_init(|| {
        CANONICAL
            .iter()
            .copied()
            .filter(|d| d.is_compiled_in())
            .collect()
    })
}

fn join(devices: &[Device]) -> String {
    devices
        .iter()
        .map(|d| d.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Why a requested device cannot be used. All three variants are *client* errors (HTTP 400) when
/// they come from a request body, and startup errors when they come from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// Not a device name at all.
    Unknown(String),
    /// A real device, but this build was not compiled with its backend.
    NotCompiledIn(Device),
    /// A device this build supports, but which this *process* did not load. Distinct from
    /// `NotCompiledIn` because the fix is different: set `DIAR_DEVICES`, not rebuild.
    NotLoaded {
        requested: Device,
        loaded: Vec<Device>,
    },
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::Unknown(name) => write!(
                f,
                "unsupported device '{name}'; this build serves [{}]",
                join(supported())
            ),
            // Mirrors speakrs' own ExecutionModeError wording ("{mode} requires the `{feature}`
            // Cargo feature") so the message reads the same wherever it surfaces from.
            DeviceError::NotCompiledIn(device) => write!(
                f,
                "{device} requires the `{}` Cargo feature; this build serves [{}]",
                device.required_feature().unwrap_or("<none>"),
                join(supported())
            ),
            DeviceError::NotLoaded { requested, loaded } => write!(
                f,
                "device '{requested}' is not loaded; this server is serving [{}] \
                 (add it to DIAR_DEVICES to load it)",
                join(loaded)
            ),
        }
    }
}

impl std::error::Error for DeviceError {}

/// Resolve the device list to load, from the two environment knobs.
///
/// * `DIAR_DEVICES` — comma-separated, **first entry is the default device**. Wins over
///   `DIAR_MODE`. Duplicates are collapsed, order preserved. Blank/whitespace-only is treated as
///   unset rather than as an error, because `DIAR_DEVICES=${SOMETHING:-}` in a compose file
///   expands to exactly that and should not be a fatal misconfiguration.
/// * `DIAR_MODE` — the pre-existing single-device knob, unchanged, used when `DIAR_DEVICES` is
///   absent. Note the deliberately preserved quirk: an unset *or unrecognized* value falls
///   through to `cuda`, exactly as the server has always behaved.
pub fn plan_devices(
    devices_env: Option<&str>,
    mode_env: Option<&str>,
) -> Result<Vec<Device>, DeviceError> {
    if let Some(list) = devices_env {
        let mut planned: Vec<Device> = Vec::new();
        for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let device = name.parse::<Device>()?;
            if !planned.contains(&device) {
                planned.push(device);
            }
        }
        if !planned.is_empty() {
            return Ok(planned);
        }
    }
    let device = device_from_mode(mode_env);
    // Capability-check the DIAR_MODE path too. Same fail-fast outcome as before (the engine load
    // would have rejected it), with the capability list added to the message.
    device.as_str().parse::<Device>()?;
    Ok(vec![device])
}

fn device_from_mode(mode_env: Option<&str>) -> Device {
    match mode_env {
        Some("cpu") => Device::Cpu,
        Some("coreml") => Device::CoreMl,
        Some("coreml_fast") => Device::CoreMlFast,
        _ => Device::Cuda,
    }
}

/// The loaded engines, one per device, plus which one unlabelled requests go to.
///
/// Each engine owns its own ORT sessions; sessions take their execution mode as a plain
/// per-session parameter, so a CUDA engine and a CPU engine coexist with no interaction. Path
/// selection inside speakrs is by *model-file presence*, not by mode, so a CPU engine over the
/// same models directory takes the same code path — and produces the same outputs — as the
/// CPU-only image does today.
pub struct EngineRegistry {
    default: Device,
    engines: Vec<(Device, Mutex<DiarEngine>)>,
}

impl EngineRegistry {
    /// Read `DIAR_DEVICES`/`DIAR_MODE` and load. Call only from `run()`, before `axum::serve` —
    /// see the module docs for why this is not merely a convention.
    pub fn load_from_env(models_dir: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let devices = plan_devices(
            std::env::var("DIAR_DEVICES").ok().as_deref(),
            std::env::var("DIAR_MODE").ok().as_deref(),
        )?;
        Self::load(models_dir, &devices)
    }

    /// Load the given devices **serially**. Fails fast: if any listed device cannot load, the
    /// whole server fails to start, because consumers depend on "diar-server exits when it
    /// cannot load models".
    pub fn load(models_dir: impl Into<PathBuf>, devices: &[Device]) -> anyhow::Result<Self> {
        let models_dir = models_dir.into();
        let default = *devices
            .first()
            .context("no execution devices configured (DIAR_DEVICES resolved to an empty list)")?;
        let mut engines = Vec::with_capacity(devices.len());
        for &device in devices {
            let engine = DiarEngine::load(&EngineConfig::new(models_dir.clone(), device.to_mode()))
                .with_context(|| format!("loading {device} engine"))?;
            engines.push((device, Mutex::new(engine)));
        }
        Ok(Self { default, engines })
    }

    pub fn default_device(&self) -> Device {
        self.default
    }

    /// Loaded and serving, in load order (first == default).
    pub fn devices(&self) -> Vec<Device> {
        self.engines.iter().map(|(d, _)| *d).collect()
    }

    /// Turn an optional request-supplied device name into a device this process can actually
    /// run. `None` (field omitted or null) means the default — the pre-existing behaviour.
    ///
    /// Handlers call this *before* acquiring an admission permit, so a bad device name costs a
    /// 400 and nothing else.
    pub fn resolve(&self, requested: Option<&str>) -> Result<Device, DeviceError> {
        let device = match requested {
            None => return Ok(self.default),
            Some(name) => name.parse::<Device>()?,
        };
        if self.engines.iter().any(|(d, _)| *d == device) {
            Ok(device)
        } else {
            Err(DeviceError::NotLoaded {
                requested: device,
                loaded: self.devices(),
            })
        }
    }

    /// The prototype engine for a device. `None` only if the device was never loaded; callers
    /// that went through [`Self::resolve`] have already excluded that.
    pub fn engine(&self, device: Device) -> Option<&Mutex<DiarEngine>> {
        self.engines
            .iter()
            .find(|(d, _)| *d == device)
            .map(|(_, m)| m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device that this build cannot serve, if there is one — lets the capability tests say
    /// something meaningful in every feature combination instead of being cfg'd out.
    fn unsupported_example() -> Option<Device> {
        CANONICAL.iter().copied().find(|d| !d.is_compiled_in())
    }

    #[test]
    fn wire_names_round_trip() {
        for device in CANONICAL {
            assert_eq!(Device::from_name(device.as_str()), Some(device));
        }
    }

    #[test]
    fn cpu_is_always_available() {
        // The whole premise of the superset image: CPU needs no Cargo feature and is always
        // compiled in, including in the CUDA build.
        assert!(Device::Cpu.is_compiled_in());
        assert!(Device::Cpu.required_feature().is_none());
        assert!(supported().contains(&Device::Cpu));
        assert_eq!("cpu".parse::<Device>(), Ok(Device::Cpu));
    }

    #[test]
    fn supported_is_exactly_the_compiled_in_devices() {
        for device in CANONICAL {
            assert_eq!(
                supported().contains(&device),
                device.is_compiled_in(),
                "{device} capability list disagrees with its cfg"
            );
        }
    }

    #[test]
    fn parse_accepts_every_supported_device() {
        for &device in supported() {
            assert_eq!(device.as_str().parse::<Device>(), Ok(device));
        }
    }

    #[test]
    fn parse_rejects_devices_this_build_cannot_serve() {
        let Some(device) = unsupported_example() else {
            return; // build has every backend compiled in; nothing to reject
        };
        let err = device.as_str().parse::<Device>().unwrap_err();
        assert_eq!(err, DeviceError::NotCompiledIn(device));
        let msg = err.to_string();
        assert!(msg.contains("Cargo feature"), "{msg}");
        assert!(msg.contains(Device::Cpu.as_str()), "{msg}");
    }

    #[test]
    fn parse_rejects_unknown_names_and_names_the_capability_list() {
        let err = "tpu".parse::<Device>().unwrap_err();
        assert_eq!(err, DeviceError::Unknown("tpu".into()));
        let msg = err.to_string();
        assert!(msg.contains("unsupported device 'tpu'"), "{msg}");
        assert!(msg.contains("this build serves"), "{msg}");
        assert!(msg.contains(Device::Cpu.as_str()), "{msg}");
        // A near-miss must not be silently coerced to something that runs.
        assert!("CPU".parse::<Device>().is_err());
        assert!("".parse::<Device>().is_err());
    }

    #[test]
    fn to_mode_maps_every_device() {
        assert_eq!(Device::Cpu.to_mode(), Mode::Cpu);
        assert_eq!(Device::Cuda.to_mode(), Mode::Cuda);
        assert_eq!(Device::CoreMl.to_mode(), Mode::CoreMl);
        assert_eq!(Device::CoreMlFast.to_mode(), Mode::CoreMlFast);
    }

    #[test]
    fn diar_mode_path_is_unchanged() {
        assert_eq!(plan_devices(None, Some("cpu")), Ok(vec![Device::Cpu]));
        // Preserved quirk: unset OR unrecognized both mean cuda, exactly as before the registry.
        for mode in [None, Some("nonsense"), Some("CUDA")] {
            let planned = plan_devices(None, mode);
            if Device::Cuda.is_compiled_in() {
                assert_eq!(planned, Ok(vec![Device::Cuda]), "mode={mode:?}");
            } else {
                assert_eq!(
                    planned,
                    Err(DeviceError::NotCompiledIn(Device::Cuda)),
                    "mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn diar_devices_wins_over_diar_mode() {
        assert_eq!(
            plan_devices(Some("cpu"), Some("cuda")),
            Ok(vec![Device::Cpu])
        );
    }

    #[test]
    fn diar_devices_dedupes_and_preserves_order() {
        // First entry is the default, so order is load-bearing and dedupe must keep the
        // first occurrence, not the last.
        assert_eq!(
            plan_devices(Some(" cpu , cpu ,cpu"), None),
            Ok(vec![Device::Cpu])
        );
        if Device::Cuda.is_compiled_in() {
            assert_eq!(
                plan_devices(Some("cpu,cuda,cpu"), None),
                Ok(vec![Device::Cpu, Device::Cuda])
            );
            assert_eq!(
                plan_devices(Some("cuda,cpu"), None),
                Ok(vec![Device::Cuda, Device::Cpu])
            );
        }
    }

    #[test]
    fn blank_diar_devices_falls_back_to_diar_mode() {
        // `DIAR_DEVICES=${FOO:-}` in a compose file expands to "" — must not be fatal.
        assert_eq!(plan_devices(Some(""), Some("cpu")), Ok(vec![Device::Cpu]));
        assert_eq!(
            plan_devices(Some("  , ,"), Some("cpu")),
            Ok(vec![Device::Cpu])
        );
    }

    #[test]
    fn diar_devices_rejects_a_bad_entry_rather_than_skipping_it() {
        assert_eq!(
            plan_devices(Some("cpu,tpu"), None),
            Err(DeviceError::Unknown("tpu".into()))
        );
    }

    #[test]
    fn not_loaded_message_distinguishes_itself_from_not_compiled_in() {
        let err = DeviceError::NotLoaded {
            requested: Device::Cpu,
            loaded: vec![Device::Cuda],
        };
        let msg = err.to_string();
        assert!(msg.contains("is not loaded"), "{msg}");
        assert!(msg.contains("DIAR_DEVICES"), "{msg}");
        assert!(!msg.contains("Cargo feature"), "{msg}");
    }
}
