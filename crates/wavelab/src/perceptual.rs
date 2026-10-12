//! Objective measures of how different two signals are.
//!
//! * The error-to-signal ratio (ESR), plain and after the first-order
//!   pre-emphasis `1 - 0.95 z^-1`: the loss neural amp models are trained
//!   and reported with, so the numbers compare with that literature.
//! * The noise-to-mask ratio (NMR): the difference between the two signals,
//!   set against the masking threshold that the reference itself produces
//!   in the ear, band by band and frame by frame. Below 0 dB the difference
//!   is under the threshold of the model.
//!
//! The NMR follows the ear model and the definitions of the basic version
//! of PEAQ (ITU-R BS.1387), as laid out by P. Kabal, "An Examination and
//! Interpretation of ITU-R BS.1387" (2002): Hann-windowed FFT, outer and
//! middle ear weighting, quarter-Bark bands from 80 Hz to 18 kHz, internal
//! noise, level-dependent spreading across bands, spreading in time, and a
//! masking offset. It is an implementation of that description, not the
//! standard's reference code, and it has not been run against the
//! standard's conformance set. PEAQ's final grade (the ODG) needs ten more
//! variables and a trained network, which are not here.

// Index loops read like the formulas they implement.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

/// Sound pressure level of a full-scale sine, dB SPL, as PEAQ assumes.
const FULL_SCALE_DB_SPL: f64 = 92.0;
const BARK_PER_BAND: f64 = 0.25;
/// A frame counts as disturbed when the noise exceeds the mask by this
/// much in any band (dB): PEAQ's "relative disturbed frames".
const DISTURBED_DB: f64 = 1.5;
/// Frames quieter than this relative to the loudest are not rated.
const SILENCE_DB: f64 = -60.0;

/// In-place radix-2 FFT.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let (wr, wi) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                (re[b], im[b]) = (re[a] - tr, im[a] - ti);
                (re[a], im[a]) = (re[a] + tr, im[a] + ti);
                (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr);
            }
        }
        len <<= 1;
    }
}

struct Ear {
    nfft: usize,
    hop: usize,
    window: Vec<f64>,
    /// Outer and middle ear weighting of each bin, as a power gain.
    weight: Vec<f64>,
    /// For each band, the bins it gathers and the share of each.
    bands: Vec<Vec<(usize, f64)>>,
    internal_noise: Vec<f64>,
    centre_hz: Vec<f64>,
    /// Masking offset of each band, as a power ratio.
    offset: Vec<f64>,
    /// Smoothing coefficient of the spreading in time.
    decay: Vec<f64>,
    /// Normalisation of the spreading across bands.
    unit_spread: Vec<f64>,
}

impl Ear {
    fn new(rate: f64) -> Ear {
        // About 43 ms per frame at any rate, as at 48 kHz in the standard.
        let nfft = 2048 * ((rate / 48_000.0).round().max(1.0) as usize).next_power_of_two();
        let scale = 10f64.powf(FULL_SCALE_DB_SPL / 20.0) * 4.0 / nfft as f64;
        let window = (0..nfft)
            .map(|n| scale * 0.5 * (1.0 - (2.0 * PI * n as f64 / (nfft as f64 - 1.0)).cos()))
            .collect();
        let bin_hz = rate / nfft as f64;
        let weight = (0..=nfft / 2)
            .map(|i| {
                let khz = (i as f64 * bin_hz / 1e3).max(1e-3);
                let db = -0.6 * 3.64 * khz.powf(-0.8) + 6.5 * (-0.6 * (khz - 3.3).powi(2)).exp()
                    - 1e-3 * khz.powf(3.6);
                10f64.powf(db / 10.0)
            })
            .collect();
        let bark = |hz: f64| 7.0 * (hz / 650.0).asinh();
        let hertz = |z: f64| 650.0 * (z / 7.0).sinh();
        let top_hz = 18_000f64.min(0.45 * rate);
        let (low, high) = (bark(80.0), bark(top_hz));
        let count = ((high - low) / BARK_PER_BAND).ceil() as usize;
        let mut bands = Vec::with_capacity(count);
        let mut centre_hz = Vec::with_capacity(count);
        for k in 0..count {
            let (zl, zu) = (
                low + k as f64 * BARK_PER_BAND,
                (low + (k + 1) as f64 * BARK_PER_BAND).min(high),
            );
            let (fl, fu) = (hertz(zl), hertz(zu));
            centre_hz.push(hertz(0.5 * (zl + zu)));
            let (first, last) = (
                (fl / bin_hz - 0.5).floor().max(0.0) as usize,
                (fu / bin_hz + 0.5).ceil() as usize,
            );
            bands.push(
                (first..=last.min(nfft / 2))
                    .filter_map(|i| {
                        let (bl, bu) = ((i as f64 - 0.5) * bin_hz, (i as f64 + 0.5) * bin_hz);
                        let share = (bu.min(fu) - bl.max(fl)) / bin_hz;
                        (share > 0.0).then_some((i, share))
                    })
                    .collect(),
            );
        }
        let internal_noise = centre_hz
            .iter()
            .map(|f| 10f64.powf(0.4 * 0.364 * (f / 1e3).powf(-0.8)))
            .collect();
        let offset = (0..count)
            .map(|k| {
                let z = k as f64 * BARK_PER_BAND;
                10f64.powf(if z <= 12.0 { 3.0 } else { 0.25 * z } / 10.0)
            })
            .collect();
        let hop = nfft / 2;
        let decay = centre_hz
            .iter()
            .map(|f| {
                let tau = 0.008 + (100.0 / f) * (0.030 - 0.008);
                (-(hop as f64 / rate) / tau).exp()
            })
            .collect();
        let mut ear = Ear {
            nfft,
            hop,
            window,
            weight,
            bands,
            internal_noise,
            centre_hz,
            offset,
            decay,
            unit_spread: vec![1.0; count],
        };
        ear.unit_spread = ear.spread(&vec![1.0; count]);
        ear
    }

    /// Weighted magnitude of each bin of the frame starting at `at`.
    fn spectrum(&self, x: &[f64], at: usize, re: &mut [f64], im: &mut [f64], out: &mut [f64]) {
        for (n, (r, w)) in re.iter_mut().zip(&self.window).enumerate() {
            *r = x[at + n] * w;
        }
        im.fill(0.0);
        fft(re, im);
        for (i, o) in out.iter_mut().enumerate() {
            *o = ((re[i] * re[i] + im[i] * im[i]) * self.weight[i]).sqrt();
        }
    }

    fn gather(&self, power: impl Fn(usize) -> f64, out: &mut [f64]) {
        for (band, o) in self.bands.iter().zip(out.iter_mut()) {
            *o = band
                .iter()
                .map(|&(i, share)| share * power(i))
                .sum::<f64>()
                .max(1e-12);
        }
    }

    /// Spread each band's energy over its neighbours: 27 dB per Bark
    /// downwards, and upwards a slope that flattens as the level rises.
    /// Energies add with an exponent of 0.4.
    fn spread(&self, energy: &[f64]) -> Vec<f64> {
        let count = energy.len();
        let down = 10f64.powf(-27.0 * BARK_PER_BAND / 10.0);
        let mut out = vec![0.0; count];
        for j in 0..count {
            let level = 10.0 * energy[j].log10();
            let slope = (-24.0 - 230.0 / self.centre_hz[j] + 0.2 * level).min(-1.0);
            let up = 10f64.powf(slope * BARK_PER_BAND / 10.0);
            // Share the band's energy so that its spread adds up to itself.
            let total = (1.0 - down.powi(j as i32 + 1)) / (1.0 - down)
                + up * (1.0 - up.powi((count - j - 1) as i32)) / (1.0 - up);
            let base = (energy[j] / total).powf(0.4);
            let (down4, up4) = (down.powf(0.4), up.powf(0.4));
            let mut v = base;
            for k in (0..=j).rev() {
                out[k] += v;
                v *= down4;
            }
            let mut v = base * up4;
            for o in out.iter_mut().skip(j + 1) {
                *o += v;
                v *= up4;
            }
        }
        out.iter()
            .zip(&self.unit_spread)
            .map(|(v, unit)| v.powf(2.5) / unit)
            .collect()
    }
}

/// Noise-to-mask ratio of the difference between two aligned signals.
#[derive(Debug, Clone, Copy)]
pub struct Nmr {
    /// Average over frames and bands, dB. Below zero, the difference sits
    /// under the masking threshold on average.
    pub total_db: f64,
    /// Share of the frames in which the difference exceeds the mask by
    /// 1.5 dB in at least one band.
    pub disturbed: f64,
    /// The largest excess over the mask in any band of any frame, dB.
    pub worst_db: f64,
    /// When that happens, seconds.
    pub worst_at: f64,
}

/// `reference` and `candidate` must be aligned in time and level. They are
/// rated as if played with the reference peaking just under full scale,
/// which is the level at which a difference is easiest to hear.
pub fn nmr(reference: &[f64], candidate: &[f64], rate: f64) -> Nmr {
    let ear = Ear::new(rate);
    let n = reference.len().min(candidate.len());
    let peak = reference[..n]
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-12);
    let level = 0.9 / peak;
    let r: Vec<f64> = reference[..n].iter().map(|v| v * level).collect();
    let c: Vec<f64> = candidate[..n].iter().map(|v| v * level).collect();

    let (bins, count) = (ear.nfft / 2 + 1, ear.bands.len());
    let (mut re, mut im) = (vec![0.0; ear.nfft], vec![0.0; ear.nfft]);
    let (mut sr, mut sc) = (vec![0.0; bins], vec![0.0; bins]);
    let (mut pitch, mut noise) = (vec![0.0; count], vec![0.0; count]);
    let mut smeared = vec![0.0; count];
    let mut frames: Vec<(f64, f64, f64)> = Vec::new(); // energy, mean NMR, max NMR
    let mut at = 0;
    while at + ear.nfft <= n {
        ear.spectrum(&r, at, &mut re, &mut im, &mut sr);
        ear.spectrum(&c, at, &mut re, &mut im, &mut sc);
        ear.gather(|i| sr[i] * sr[i], &mut pitch);
        let energy: f64 = pitch.iter().sum();
        for (p, internal) in pitch.iter_mut().zip(&ear.internal_noise) {
            *p += internal;
        }
        let excitation = ear.spread(&pitch);
        // The difference of the magnitudes: what the ear model is given is
        // a spectrum, so a pure phase difference is not a difference.
        ear.gather(|i| (sr[i] - sc[i]) * (sr[i] - sc[i]), &mut noise);
        let (mut sum, mut max) = (0.0, 0.0f64);
        for k in 0..count {
            smeared[k] = ear.decay[k] * smeared[k] + (1.0 - ear.decay[k]) * excitation[k];
            let mask = smeared[k].max(excitation[k]) / ear.offset[k];
            let ratio = noise[k] / mask;
            sum += ratio;
            max = max.max(ratio);
        }
        frames.push((energy, sum / count as f64, max));
        at += ear.hop;
    }
    let loudest = frames.iter().fold(0.0f64, |m, f| m.max(f.0));
    let floor = loudest * 10f64.powf(SILENCE_DB / 10.0);
    let rated: Vec<(usize, &(f64, f64, f64))> = frames
        .iter()
        .enumerate()
        .filter(|f| f.1 .0 > floor)
        .collect();
    if rated.is_empty() {
        return Nmr {
            total_db: f64::NEG_INFINITY,
            disturbed: 0.0,
            worst_db: f64::NEG_INFINITY,
            worst_at: 0.0,
        };
    }
    let db = |x: f64| 10.0 * x.max(1e-30).log10();
    let limit = 10f64.powf(DISTURBED_DB / 10.0);
    let worst = rated
        .iter()
        .max_by(|a, b| a.1 .2.total_cmp(&b.1 .2))
        .unwrap();
    Nmr {
        total_db: db(rated.iter().map(|f| f.1 .1).sum::<f64>() / rated.len() as f64),
        disturbed: rated.iter().filter(|f| f.1 .2 > limit).count() as f64 / rated.len() as f64,
        worst_db: db(worst.1 .2),
        worst_at: (worst.0 * ear.hop) as f64 / rate,
    }
}

/// Error-to-signal ratio in dB: energy of `reference - candidate` over the
/// energy of `reference`, both passed through `1 - coefficient z^-1` first
/// (0 for the plain ratio, 0.95 for the usual pre-emphasis, which weighs
/// the treble as the ear roughly does).
pub fn esr_db(reference: &[f64], candidate: &[f64], coefficient: f64) -> f64 {
    let (mut error, mut signal) = (0.0, 0.0);
    let (mut r1, mut c1) = (0.0, 0.0);
    for (&r, &c) in reference.iter().zip(candidate) {
        let (fr, fc) = (r - coefficient * r1, c - coefficient * c1);
        (r1, c1) = (r, c);
        error += (fr - fc) * (fr - fc);
        signal += fr * fr;
    }
    10.0 * (error / signal.max(1e-30)).max(1e-30).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    fn tone(hz: f64, amplitude: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| amplitude * (2.0 * PI * hz * k as f64 / RATE).sin())
            .collect()
    }

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Twenty sinusoids with scattered phases within 50 Hz of `hz`: noise
    /// that stays inside one critical band.
    fn narrow_noise(hz: f64, n: usize) -> Vec<f64> {
        let phases = noise(20, 0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|k| {
                let t = k as f64 / RATE;
                (0..20)
                    .map(|i| (2.0 * PI * ((hz - 50.0 + 5.0 * i as f64) * t + phases[i])).sin())
                    .sum()
            })
            .collect()
    }

    fn with(reference: &[f64], added: &[f64], db: f64) -> Vec<f64> {
        let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        let gain = rms(reference) / rms(added) * 10f64.powf(db / 20.0);
        reference
            .iter()
            .zip(added)
            .map(|(r, a)| r + gain * a)
            .collect()
    }

    #[test]
    fn fft_of_a_sine_is_one_line() {
        let n = 64;
        let mut re: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * 5.0 * k as f64 / n as f64).cos())
            .collect();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            let want = if k == 5 || k == n - 5 {
                n as f64 / 2.0
            } else {
                0.0
            };
            assert!(
                (re[k] - want).abs() < 1e-9 && im[k].abs() < 1e-9,
                "bin {k}: {} {}",
                re[k],
                im[k]
            );
        }
    }

    #[test]
    fn esr_is_the_level_of_what_was_added() {
        let x = noise(48_000, 1);
        let y = with(&x, &noise(48_000, 2), -40.0);
        assert!((esr_db(&x, &y, 0.0) + 40.0).abs() < 0.1);
        assert_eq!(esr_db(&x, &x, 0.95), -300.0);
        // Pre-emphasis weighs a treble error more than a bass one.
        let (bass, treble) = (
            with(&x, &tone(100.0, 1.0, 48_000), -40.0),
            with(&x, &tone(8_000.0, 1.0, 48_000), -40.0),
        );
        assert!(esr_db(&x, &treble, 0.95) > esr_db(&x, &bass, 0.95) + 20.0);
    }

    #[test]
    fn identical_signals_have_no_noise_to_mask() {
        let x = tone(1_000.0, 0.5, 96_000);
        let r = nmr(&x, &x, RATE);
        assert!(r.total_db < -100.0 && r.disturbed == 0.0, "{r:?}");
    }

    #[test]
    fn nmr_follows_the_level_of_the_difference() {
        let x = tone(1_000.0, 0.5, 96_000);
        let hiss = noise(96_000, 3);
        let at = |db: f64| nmr(&x, &with(&x, &hiss, db), RATE);
        let (faint, plain) = (at(-90.0), at(-30.0));
        // Broadband noise 30 dB under a pure tone is plainly audible; 90 dB
        // under, it is not.
        assert!(plain.total_db > 0.0 && plain.disturbed > 0.95, "{plain:?}");
        assert!(
            faint.total_db < -10.0 && faint.disturbed == 0.0,
            "{faint:?}"
        );
        // 10 dB more noise is 10 dB more noise-to-mask ratio.
        let step = at(-50.0).total_db - at(-60.0).total_db;
        assert!((step - 10.0).abs() < 1.0, "{step}");
    }

    #[test]
    fn a_tone_masks_what_is_near_it_better_than_what_is_far() {
        // The same amount of narrow noise, placed around the tone or three
        // octaves above it: the first hides behind the tone.
        let x = tone(1_000.0, 0.5, 96_000);
        let near = nmr(&x, &with(&x, &narrow_noise(1_000.0, 96_000), -45.0), RATE);
        let far = nmr(&x, &with(&x, &narrow_noise(8_000.0, 96_000), -45.0), RATE);
        assert!(
            far.worst_db > near.worst_db + 15.0,
            "near {near:?}, far {far:?}"
        );
        assert!(
            near.disturbed < 0.05 && far.disturbed > 0.95,
            "near {near:?}, far {far:?}"
        );
    }

    #[test]
    fn a_phase_shift_is_not_a_difference_but_fails_the_esr() {
        // Two partials; the candidate has the upper one a quarter turn late.
        let n = 96_000;
        let x: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 / RATE;
                0.4 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).sin()
            })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|k| {
                let t = k as f64 / RATE;
                0.4 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 1_320.0 * t).cos()
            })
            .collect();
        assert!(esr_db(&x, &y, 0.0) > -10.0);
        let r = nmr(&x, &y, RATE);
        assert!(r.total_db < -20.0 && r.disturbed == 0.0, "{r:?}");
    }
}
