pub mod bench;
pub mod compare;
pub mod compare_wav;
pub mod golden;
pub mod inspect;
pub mod render;

use crate::signal;
use crate::wav;
use std::path::{Path, PathBuf};

/// Input audio for compare/bench: a WAV file or a synthetic DI.
pub struct Stimulus {
    pub name: String,
    pub samples: Vec<f32>,
    pub sample_rate: f32,
}

pub fn stimulus(
    input: Option<&PathBuf>,
    channel: usize,
    seconds: f32,
    model_rate: f32,
) -> Result<Stimulus, String> {
    match input {
        Some(path) => {
            let audio = wav::read_mono(path, channel)?;
            if (audio.sample_rate as f32 - model_rate).abs() > 0.5 {
                eprintln!(
                    "warning: {} is {} Hz but the model expects {} Hz (no resampling is done)",
                    path.display(),
                    audio.sample_rate,
                    model_rate
                );
            }
            let samples = if seconds > 0.0 {
                let n = ((seconds * audio.sample_rate as f32) as usize).min(audio.samples.len());
                audio.samples[..n].to_vec()
            } else {
                audio.samples
            };
            Ok(Stimulus {
                name: path.display().to_string(),
                samples,
                sample_rate: audio.sample_rate as f32,
            })
        }
        None => {
            let seconds = if seconds > 0.0 { seconds } else { 5.0 };
            Ok(Stimulus {
                name: format!("synthetic DI {seconds}s"),
                samples: signal::synthetic_di(model_rate, seconds, 0xC0FFEE, 0.5),
                sample_rate: model_rate,
            })
        }
    }
}

pub fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}
