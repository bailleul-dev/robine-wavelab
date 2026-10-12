//! Deterministic test signals: no random number crate, and the continuous
//! ones are defined in time so that they can be sampled at any rate.

use std::f64::consts::TAU;

/// A guitar-like band-limited signal: three partials under an envelope that
/// rises from silence to `peak` volts over `duration` seconds.
pub fn chord(t: f64, duration: f64, peak: f64) -> f64 {
    let envelope = (t / duration).clamp(0.0, 1.0);
    let partials = 0.5 * (TAU * 110.0 * t).sin()
        + 0.3 * (TAU * 523.25 * t).sin()
        + 0.2 * (TAU * 2217.5 * t).sin();
    peak * envelope * partials
}

pub fn sample(rate: f64, seconds: f64, peak: f64) -> Vec<f64> {
    let n = (rate * seconds).round() as usize;
    (0..n)
        .map(|k| chord(k as f64 / rate, seconds, peak))
        .collect()
}

pub struct Lcg(u32);

impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }
    /// Uniform in [-1, 1).
    // Not an iterator: it never ends, and callers want a plain number.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// Guitar-ish synthetic DI: plucked tones (sum of decaying harmonics) with
/// random onsets, a little noise, peak-normalised to `peak`.
pub fn synthetic_di(sample_rate: f32, seconds: f32, seed: u32, peak: f32) -> Vec<f32> {
    let len = (sample_rate * seconds) as usize;
    let mut out = vec![0.0f32; len];
    let mut rng = Lcg::new(seed);
    let notes = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 659.25];
    let mut t0 = 0usize;
    while t0 < len {
        let f0 = notes[(rng.next().abs() * notes.len() as f32) as usize % notes.len()];
        let dur = (sample_rate * (0.15 + 0.5 * rng.next().abs())) as usize;
        let amp = 0.3 + 0.7 * rng.next().abs();
        let decay = 2.0 + 6.0 * rng.next().abs();
        for i in 0..dur.min(len - t0) {
            let t = i as f32 / sample_rate;
            let env = (-decay * t).exp();
            let mut s = 0.0;
            for h in 1..=6 {
                let hf = h as f32;
                s += (2.0 * core::f32::consts::PI * f0 * hf * t).sin() * (-0.6 * (hf - 1.0)).exp();
            }
            out[t0 + i] += amp * env * s;
        }
        t0 += (sample_rate * (0.08 + 0.3 * rng.next().abs())) as usize;
    }
    for s in out.iter_mut() {
        *s += 0.002 * rng.next();
    }
    let max = out.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-9);
    for s in out.iter_mut() {
        *s *= peak / max;
    }
    out
}
