//! Mono WAV input and output.

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Read a WAV file as mono f32 (first channel by default).
pub fn read_mono(path: &Path, channel: usize) -> Result<Audio, String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channel >= channels {
        return Err(format!(
            "{}: channel {channel} requested but file has {channels}",
            path.display()
        ));
    }
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        (SampleFormat::Int, bits) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
        (fmt, bits) => {
            return Err(format!(
                "{}: unsupported format {fmt:?}/{bits}",
                path.display()
            ))
        }
    };
    let samples = interleaved
        .iter()
        .skip(channel)
        .step_by(channels)
        .copied()
        .collect();
    Ok(Audio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Write mono 32-bit float WAV.
pub fn write_mono_f32(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|e| format!("{}: {e}", path.display()))?;
    for &s in samples {
        writer.write_sample(s).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())
}

/// Write mono 16-bit WAV, clipping at full scale.
pub fn write_mono_i16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|e| format!("{}: {e}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(v).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())
}
