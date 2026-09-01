//! What does ORT's load-time optimizer actually do to a graph on THIS build?
//!
//! Written for issue #14, where `models_folded/gender-wav2vec2.onnx` (fp16) fails to load on
//! linux/arm64 with a missing kernel for `com.microsoft.Gelu` — an op that is NOT in the
//! model file. ORT creates it during session load and then cannot execute it. Full write-up:
//! `docs/ORT_FUSION_FP16_AARCH64.md`; measurements: `validation/RESULTS.md` §7.40.
//!
//! Two subcommands:
//!
//!   load   <dumpdir> <model.onnx>...
//!       Load each graph at Level3 and write its OPTIMIZED form to <dumpdir>. Answers "does
//!       this graph load, and what did ORT rewrite it into". Use `inspect_dumps.py` on the
//!       output to list the contrib ops each graph acquired.
//!
//!   run    <model.onnx> <clips.bin> <dumpdir> <spec>...
//!       Same, plus run every clip through each configuration and print the logits, so a
//!       candidate fix can be checked for output identity against the unoptimized reference.
//!       spec := L0|L1|L2|L3 [ ":" optimizer-names ]
//!       L0 = no optimization at all — the semantic reference every other level must match.
//!
//! `clips.bin`: u32 clip-count, then per clip a u32 sample-count and that many little-endian
//! f32. `make_clips.py` regenerates the exact 6-clip gate corpus `export_gender.py` uses.
//!
//! THREE TRAPS this tool exists to keep you out of (all measured, all in the doc):
//!   1. the optimizer is named `GeluFusionL2`, NOT `GeluFusion`;
//!   2. an unrecognized optimizer name is SILENTLY IGNORED — no error, no warning, so a
//!      misspelled name looks applied and does nothing;
//!   3. the separator for multiple names is `;`, not `,`, despite ort's doc comment.
//! Which is why this prints the load outcome per spec instead of assuming a config took.

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::io::Read;
use std::path::Path;

fn load_clips(path: &str) -> Vec<Vec<f32>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .read_to_end(&mut buf)
        .unwrap();
    let rd = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
    let mut off = 4;
    (0..rd(0))
        .map(|_| {
            let len = rd(off);
            off += 4;
            let v: Vec<f32> = (0..len)
                .map(|i| f32::from_le_bytes(buf[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
                .collect();
            off += len * 4;
            v
        })
        .collect()
}

/// `L3:GeluFusionL2` -> (Level3, Some("GeluFusionL2")).
fn parse_spec(spec: &str) -> (GraphOptimizationLevel, Option<&str>) {
    let (level, disabled) = match spec.split_once(':') {
        Some((l, d)) => (l, Some(d)),
        None => (spec, None),
    };
    let level = match level {
        "L0" => GraphOptimizationLevel::Disable,
        "L1" => GraphOptimizationLevel::Level1,
        "L2" => GraphOptimizationLevel::Level2,
        "L3" => GraphOptimizationLevel::Level3,
        other => panic!("bad optimization level {other:?}; expected L0|L1|L2|L3"),
    };
    (level, disabled)
}

fn build(spec: &str, dump: &str, model: &str) -> ort::Result<Session> {
    let (level, disabled) = parse_spec(spec);
    let b = Session::builder()?.with_optimization_level(level)?;
    let b = match disabled {
        Some(d) => b.with_disabled_optimizers(d)?,
        None => b,
    };
    // Serializing the optimized graph is the only way to see what the fusions did; the
    // session itself exposes no node list. Verified to capture Level-2 fusions (§7.40: the
    // fp32 gender graph shows com.microsoft::Gelu here, the fp16 one does not).
    b.with_optimized_model_path(dump)?.commit_from_file(model)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    match cmd.as_str() {
        "load" => {
            let dump_dir = args.next().expect("usage: load <dumpdir> <model.onnx>...");
            for model in args {
                let stem = Path::new(&model).file_stem().unwrap().to_string_lossy().to_string();
                match build("L3", &format!("{dump_dir}/{stem}.opt.onnx"), &model) {
                    Ok(_) => println!("LOAD ok   {stem}"),
                    Err(e) => println!("LOAD FAIL {stem}\n    {e}"),
                }
            }
        }
        "run" => {
            let model = args.next().expect("usage: run <model> <clips.bin> <dumpdir> <spec>...");
            let clips = load_clips(&args.next().expect("clips.bin"));
            let dump_dir = args.next().expect("dumpdir");
            let specs: Vec<String> = args.collect();
            assert!(!specs.is_empty(), "give at least one spec, e.g. L3 L1 L0");
            println!("model: {model}");
            println!("clips: {} {:?}", clips.len(), clips.iter().map(Vec::len).collect::<Vec<_>>());
            for spec in &specs {
                let dump = format!("{dump_dir}/opt-{}.onnx", spec.replace([':', ',', ';', '.'], "_"));
                print!("SPEC {spec:<52} ");
                let mut session = match build(spec, &dump, &model) {
                    Ok(s) => {
                        println!("LOAD=ok");
                        s
                    }
                    // A load failure is a RESULT here, not an error to abort on — it is
                    // precisely what distinguishes a working fix from an ignored one.
                    Err(e) => {
                        println!("LOAD=FAIL\n  error: {e}");
                        continue;
                    }
                };
                for (i, clip) in clips.iter().enumerate() {
                    let arr = Array2::from_shape_vec((1, clip.len()), clip.clone()).unwrap();
                    let out = session
                        .run(ort::inputs!["input_values" => Tensor::from_array(arr).unwrap()])
                        .unwrap();
                    let (_, d) = out["logits"].try_extract_tensor::<f32>().unwrap();
                    println!(
                        "  RESULT\t{spec}\tclip{i}\tn={}\t{:.9e}\t{:.9e}",
                        clip.len(),
                        d[0],
                        d[1]
                    );
                }
            }
        }
        _ => {
            eprintln!("usage:\n  ort-fusion-probe load <dumpdir> <model.onnx>...");
            eprintln!("  ort-fusion-probe run  <model.onnx> <clips.bin> <dumpdir> <spec>...");
            eprintln!("see the module docs, docs/ORT_FUSION_FP16_AARCH64.md and RESULTS §7.40");
            std::process::exit(2);
        }
    }
}
