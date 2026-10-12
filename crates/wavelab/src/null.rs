//! The null test: subtract one render from the other and look at what is
//! left. Unlike a list of criteria it cannot miss a kind of difference:
//! phase, intermodulation, a glitch on one attack all end up in the
//! residual. What it cannot do is say whether the residual is audible, so
//! the residual is also written out, to be listened to.

use crate::perceptual::{esr_db, nmr, Nmr};
use std::f64::consts::PI;

/// Frames this long are used to look for brief differences.
const FRAME_SECONDS: f64 = 0.01;
/// Frames quieter than this, relative to the loudest, are not rated: the
/// ratio of two near-silences means nothing.
const FRAME_FLOOR_DB: f64 = -50.0;
const BANDS_HZ: [f64; 9] = [62.5, 125.0, 250.0, 500.0, 1e3, 2e3, 4e3, 8e3, 16e3];

fn db(ratio: f64) -> f64 {
    20.0 * ratio.max(1e-15).log10()
}

fn energy(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// One octave around `hertz`, by a constant-peak band-pass biquad.
fn band(x: &[f64], rate: f64, hertz: f64) -> Vec<f64> {
    let w = 2.0 * PI * hertz / rate;
    let alpha = w.sin() / (2.0 * std::f64::consts::SQRT_2);
    let (b0, b2, a0, a1, a2) = (alpha, -alpha, 1.0 + alpha, -2.0 * w.cos(), 1.0 - alpha);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    x.iter()
        .map(|&v| {
            let y = (b0 * v + b2 * x2 - a1 * y1 - a2 * y2) / a0;
            (x2, x1, y2, y1) = (x1, v, y1, y);
            y
        })
        .collect()
}

pub struct Null {
    /// Gain applied to the candidate before subtracting, dB.
    pub gain_db: f64,
    /// Samples the candidate was found to lag the reference by.
    pub lag: isize,
    /// Residual relative to the reference over the whole file, dB.
    pub overall_db: f64,
    /// Residual relative to the reference in each 10 ms frame: median,
    /// 95th percentile and worst, dB, and when the worst happens, seconds.
    pub frames_db: [f64; 3],
    pub worst_at: f64,
    /// Residual relative to the reference per octave, dB (`None` where the
    /// reference has nothing).
    pub bands_db: Vec<(f64, Option<f64>)>,
    /// Error-to-signal ratio after the `1 - 0.95 z^-1` pre-emphasis used
    /// for neural amp models, dB.
    pub esr_preemphasis_db: f64,
    /// The residual against what the reference masks in the ear.
    pub nmr: Nmr,
    /// The two signals as they were compared: means removed, candidate
    /// shifted and scaled onto the reference, both at the level the ear
    /// model assumes (reference peaking at 0.9).
    pub reference: Vec<f32>,
    pub candidate: Vec<f32>,
    /// The residual itself, reference minus aligned candidate.
    pub residual: Vec<f32>,
}

/// Align `candidate` to `reference` (integer lag up to 10 ms, then the gain
/// that minimises the residual) and subtract.
pub fn null(reference: &[f32], candidate: &[f32], rate: u32) -> Null {
    let rate = f64::from(rate);
    let r: Vec<f64> = reference.iter().map(|&v| f64::from(v)).collect();
    let c: Vec<f64> = candidate.iter().map(|&v| f64::from(v)).collect();
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len().max(1) as f64;
    let (rm, cm) = (mean(&r), mean(&c));
    let r: Vec<f64> = r.iter().map(|v| v - rm).collect();
    let c: Vec<f64> = c.iter().map(|v| v - cm).collect();

    // Lag by cross-correlation over the first ten seconds.
    let span = r.len().min(c.len()).min((10.0 * rate) as usize);
    let reach = (0.01 * rate) as isize;
    let at = |x: &[f64], i: isize| {
        if i >= 0 && (i as usize) < x.len() {
            x[i as usize]
        } else {
            0.0
        }
    };
    let correlation = |lag: isize| {
        (0..span as isize)
            .map(|i| r[i as usize] * at(&c, i + lag))
            .sum::<f64>()
    };
    let lag = (-reach..=reach)
        .max_by(|a, b| correlation(*a).total_cmp(&correlation(*b)))
        .unwrap_or(0);

    let n = r.len().min(c.len());
    let aligned: Vec<f64> = (0..n as isize).map(|i| at(&c, i + lag)).collect();
    let r = &r[..n];
    let gain =
        r.iter().zip(&aligned).map(|(a, b)| a * b).sum::<f64>() / energy(&aligned).max(1e-30);
    let residual: Vec<f64> = r.iter().zip(&aligned).map(|(a, b)| a - gain * b).collect();

    let frame = (FRAME_SECONDS * rate) as usize;
    let levels: Vec<f64> = r.chunks_exact(frame).map(|f| energy(f).sqrt()).collect();
    let loudest = levels.iter().cloned().fold(0.0, f64::max);
    let mut rated: Vec<(f64, usize)> = r
        .chunks_exact(frame)
        .zip(residual.chunks_exact(frame))
        .enumerate()
        .filter(|(k, _)| db(levels[*k] / loudest.max(1e-30)) > FRAME_FLOOR_DB)
        .map(|(k, (a, d))| (db(energy(d).sqrt() / energy(a).sqrt()), k))
        .collect();
    rated.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pick = |q: f64| {
        rated
            .get(((rated.len() as f64 - 1.0) * q) as usize)
            .map_or(f64::NAN, |v| v.0)
    };
    let worst = rated.last().copied().unwrap_or((f64::NAN, 0));

    let bands_db = BANDS_HZ
        .iter()
        .filter(|&&hz| hz < 0.45 * rate)
        .map(|&hz| {
            let (a, d) = (
                energy(&band(r, rate, hz)),
                energy(&band(&residual, rate, hz)),
            );
            // A band holding less than a millionth of the signal is empty.
            (
                hz,
                (a > 1e-6 * energy(r)).then(|| 10.0 * (d / a).max(1e-30).log10()),
            )
        })
        .collect();

    let matched: Vec<f64> = aligned.iter().map(|v| gain * v).collect();
    let level = 0.9 / r.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
    Null {
        reference: r.iter().map(|v| (v * level) as f32).collect(),
        candidate: matched.iter().map(|v| (v * level) as f32).collect(),
        esr_preemphasis_db: esr_db(r, &matched, 0.95),
        nmr: nmr(r, &matched, rate),
        gain_db: db(gain.abs()),
        lag,
        overall_db: 10.0
            * (energy(&residual) / energy(r).max(1e-30))
                .max(1e-30)
                .log10(),
        frames_db: [pick(0.5), pick(0.95), worst.0],
        worst_at: worst.1 as f64 * FRAME_SECONDS,
        bands_db,
        residual: residual.iter().map(|&v| v as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn a_delayed_quieter_copy_nulls_completely() {
        let x = noise(48_000);
        let mut y = vec![0.0f32; 7];
        y.extend(x.iter().map(|v| 0.25 * v));
        let n = null(&x, &y, 48_000);
        assert_eq!(n.lag, 7);
        assert!((n.gain_db - 12.04).abs() < 0.01, "{}", n.gain_db);
        // The only residue is the seven samples the delay pushed off the end.
        assert!(n.overall_db < -100.0, "{}", n.overall_db);
    }

    #[test]
    fn a_brief_difference_shows_in_the_frames_not_in_the_total() {
        let x = noise(96_000);
        let mut y = x.clone();
        // 5 ms of one file replaced by silence, one second in.
        y[48_000..48_240].fill(0.0);
        let n = null(&x, &y, 48_000);
        assert!(n.overall_db < -25.0, "{}", n.overall_db);
        assert!(n.frames_db[2] > -6.0, "{:?}", n.frames_db);
        assert!(n.frames_db[0] < -60.0, "{:?}", n.frames_db);
        assert!((n.worst_at - 1.0).abs() < 0.011, "{}", n.worst_at);
    }

    #[test]
    fn a_difference_in_the_treble_shows_in_the_treble_bands() {
        let x = noise(48_000);
        // Add something that only exists in the treble: the difference of
        // successive samples of another noise.
        let other: Vec<f32> = noise(96_000)[48_000..].to_vec();
        let y: Vec<f32> = (0..x.len())
            .map(|k| x[k] + 0.05 * (other[k] - other[k.saturating_sub(1)]))
            .collect();
        let n = null(&x, &y, 48_000);
        let at = |hz: f64| n.bands_db.iter().find(|b| b.0 == hz).unwrap().1.unwrap();
        assert!(at(125.0) < at(8e3) - 15.0, "{:?}", n.bands_db);
    }
}
