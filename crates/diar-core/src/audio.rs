//! Native media decoding: original media in (wav/flac/mp3/m4a/ogg, any rate/channels),
//! 16 kHz mono f32 out.
//!
//! This exists so diar-server and diar-cli can ingest real media directly instead of
//! requiring a pre-decoded 16 kHz WAV. The app pipeline keeps its existing WAV handoff —
//! `.wav` at 16 kHz mono short-circuits through the same integer→f32 mapping the hound
//! path uses, so that path stays byte-identical. Everything else decodes via symphonia
//! and, when needed, resamples through a windowed-sinc (rubato) to 16 kHz.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Engine sample rate.
pub const TARGET_RATE: u32 = 16_000;

/// Decode any supported media file to 16 kHz mono f32.
///
/// Multi-channel audio is downmixed by averaging; non-16 kHz audio is resampled with a
/// windowed-sinc. Lossless 16 kHz mono input reproduces the WAV fast path's samples
/// exactly; lossy formats decode deterministically but are decoder-defined, so callers
/// needing bit-parity with an ffmpeg-decoded WAV must keep feeding WAVs.
pub fn decode_to_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probing {}", path.display()))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("{}: no decodable audio track", path.display()))?;
    let track_id = track.id;
    let in_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("{}: track has no sample rate", path.display()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("building decoder for {}", path.display()))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream is delivered as an IO error in symphonia 0.5.
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => bail!("{}: demux error: {e}", path.display()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt packet is recoverable; skip it rather than failing the file.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => bail!("{}: decode error: {e}", path.display()),
        };

        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let buf = sample_buf
            .get_or_insert_with(|| SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        buf.copy_interleaved_ref(decoded);

        let samples = buf.samples();
        if channels == 1 {
            mono.extend_from_slice(samples);
        } else {
            let scale = 1.0 / channels as f32;
            mono.extend(
                samples
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() * scale),
            );
        }
    }

    if mono.is_empty() {
        bail!("{}: decoded zero samples", path.display());
    }
    if in_rate == TARGET_RATE {
        return Ok(mono);
    }
    resample_to_16k(&mono, in_rate)
}

/// Resample mono f32 samples to 16 kHz.
///
/// FFT-based (rubato `FftFixedIn`): A/B'd against the sinc resampler on a 44.1 kHz round
/// trip of the Karpathy 10-min clip — the sinc output diarized to 203 segments (vs 90 for
/// an ffmpeg-resampled control) while this one lands at 88, matching the control and the
/// original 16 kHz clip (92).
pub fn resample_to_16k(input: &[f32], in_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};

    const CHUNK: usize = 8192;
    let mut resampler = FftFixedIn::<f32>::new(in_rate as usize, TARGET_RATE as usize, CHUNK, 2, 1)
        .map_err(|e| anyhow!("building resampler ({in_rate} Hz -> 16 kHz): {e}"))?;

    let mut out = Vec::with_capacity(
        (input.len() as u64 * u64::from(TARGET_RATE) / u64::from(in_rate)) as usize + CHUNK,
    );
    for chunk in input.chunks(CHUNK) {
        let produced = if chunk.len() == CHUNK {
            resampler
                .process(&[chunk], None)
                .map_err(|e| anyhow!("resampling: {e}"))?
        } else {
            resampler
                .process_partial(Some(&[chunk]), None)
                .map_err(|e| anyhow!("resampling tail: {e}"))?
        };
        out.extend_from_slice(&produced[0]);
    }
    // Flush the resampler's internal delay line.
    let tail = resampler
        .process_partial::<&[f32]>(None, None)
        .map_err(|e| anyhow!("flushing resampler: {e}"))?;
    out.extend_from_slice(&tail[0]);
    Ok(out)
}
