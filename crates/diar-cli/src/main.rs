//! diar-cli — bench/ops runner for the diar-core engine.
//! Emits harness-layout RTTMs (`<label>_run<N>.rttm`) + timing JSONL, matching
//! validation/run_speakrs.sh so the M1 gate scores with the same tooling.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use diar_core::{DiarEngine, EngineConfig, Mode};

#[derive(Parser)]
#[command(about = "Diarize media files (wav/flac/mp3/m4a/ogg) via diar-core (speakrs engine)")]
struct Args {
    /// WAV files (16 kHz mono)
    files: Vec<PathBuf>,
    #[arg(long)]
    models_dir: PathBuf,
    #[arg(long, default_value = "cuda")]
    mode: String,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Explicit label (single-file runs); default = file stem
    #[arg(long)]
    label: Option<String>,
    // Angle brackets are deliberately avoided: this doc comment is BOTH clap's `--help` text
    // and rustdoc input, and rustdoc parsed `<label>` and `<N>` as unclosed HTML tags (caught
    // by the `docs` CI job). Backticks would fix rustdoc but then show up verbatim in `--help`.
    /// Also write one JSON file per run, LABEL_runN.json, with segments/centroids/exclusive
    #[arg(long, default_value_t = false)]
    json: bool,
}

/// 16 kHz mono WAVs keep the hound fast path; other media decodes via symphonia.
fn load_audio(path: &PathBuf) -> Result<Vec<f32>> {
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        if let Ok(samples) = load_wav(path) {
            return Ok(samples);
        }
    }
    diar_core::audio::decode_to_16k_mono(path)
}

fn load_wav(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).context("opening wav")?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        bail!(
            "expected 16 kHz mono, got {} Hz {} ch",
            spec.sample_rate,
            spec.channels
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    Ok(samples)
}

fn main() -> Result<()> {
    // Engine stage traces (fbank_ms, gpu_predict_ms, clustering timing) go to STDERR — stdout
    // stays parseable JSONL for the bench harness. Same RUST_LOG/DIAR_LOG_FORMAT policy as
    // diar-server (`diar_core::logging`), only the sink differs. Note the behaviour change:
    // this used to default to near-silence, and now defaults to `info`.
    diar_core::logging::init_stderr();
    let args = Args::parse();
    let mode = match args.mode.as_str() {
        "cpu" => Mode::Cpu,
        "cuda" => Mode::Cuda,
        "coreml" => Mode::CoreMl,
        "coreml_fast" => Mode::CoreMlFast,
        other => bail!("unknown mode '{other}' (cpu|cuda|coreml|coreml_fast)"),
    };
    fs::create_dir_all(&args.out_dir)?;
    let mut engine = DiarEngine::load(&EngineConfig::new(&args.models_dir, mode))?;

    for file in &args.files {
        let label = match (&args.label, args.files.len()) {
            (Some(l), 1) => l.clone(),
            _ => file
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "file".into()),
        };
        let audio = load_audio(file)?;
        let duration_s = audio.len() as f64 / 16_000.0;
        for run in 0..args.runs {
            let t0 = Instant::now();
            let out = engine.diarize(&audio, &label)?;
            let elapsed = t0.elapsed().as_secs_f64();
            fs::write(
                args.out_dir.join(format!("{label}_run{run}.rttm")),
                &out.rttm,
            )?;
            if args.json {
                fs::write(
                    args.out_dir.join(format!("{label}_run{run}.json")),
                    serde_json::to_vec_pretty(&out)?,
                )?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "label": label, "run": run, "mode": args.mode,
                    "duration_s": (duration_s * 10.0).round() / 10.0,
                    "elapsed_s": (elapsed * 100.0).round() / 100.0,
                    "rtf_x": ((duration_s / elapsed) * 10.0).round() / 10.0,
                    "num_speakers": out.num_speakers,
                    "segments": out.segments.len(),
                })
            );
        }
    }
    Ok(())
}
