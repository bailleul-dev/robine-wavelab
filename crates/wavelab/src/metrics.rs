//! Signal comparison metrics shared by `compare`, `compare-wav` and `golden`.

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use serde::Serialize;

/// Sample-exact delta between two equally long signals.
#[derive(Clone, Debug, Serialize)]
pub struct Delta {
    pub samples: usize,
    pub max_abs: f32,
    pub max_abs_index: usize,
    pub max_rel: f32,
    pub rms_error: f32,
    pub rms_reference: f32,
    /// Error-to-signal ratio: `sum(e^2) / sum(r^2)`.
    pub esr: f64,
    pub snr_db: f64,
    pub correlation: f64,
    /// First index where `|a-b| > tolerance`, if any.
    pub first_exceed: Option<usize>,
    pub exceed_count: usize,
    pub tolerance: f32,
}

pub fn delta(candidate: &[f32], reference: &[f32], tolerance: f32) -> Delta {
    assert_eq!(candidate.len(), reference.len());
    let n = candidate.len();
    let mut max_abs = 0.0f32;
    let mut max_abs_index = 0;
    let mut max_rel = 0.0f32;
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut cand2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut first_exceed = None;
    let mut exceed_count = 0;
    for i in 0..n {
        let c = candidate[i];
        let r = reference[i];
        let e = (c - r).abs();
        if e > max_abs {
            max_abs = e;
            max_abs_index = i;
        }
        if r.abs() > 1e-3 {
            max_rel = max_rel.max(e / r.abs());
        }
        if e > tolerance {
            exceed_count += 1;
            if first_exceed.is_none() {
                first_exceed = Some(i);
            }
        }
        err2 += (e as f64) * (e as f64);
        ref2 += (r as f64) * (r as f64);
        cand2 += (c as f64) * (c as f64);
        dot += (c as f64) * (r as f64);
    }
    let esr = if ref2 > 0.0 {
        err2 / ref2
    } else if err2 > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let snr_db = if err2 > 0.0 {
        10.0 * (ref2 / err2).log10()
    } else {
        f64::INFINITY
    };
    let correlation = if ref2 > 0.0 && cand2 > 0.0 {
        dot / (ref2.sqrt() * cand2.sqrt())
    } else {
        0.0
    };
    Delta {
        samples: n,
        max_abs,
        max_abs_index,
        max_rel,
        rms_error: (err2 / n.max(1) as f64).sqrt() as f32,
        rms_reference: (ref2 / n.max(1) as f64).sqrt() as f32,
        esr,
        snr_db,
        correlation,
        first_exceed,
        exceed_count,
        tolerance,
    }
}

/// Perceptual-ish quality metrics between two renders that may come from
/// different tools (optional alignment and gain matching).
#[derive(Clone, Debug, Serialize)]
pub struct Quality {
    pub lag_samples: i64,
    pub gain_db: f64,
    pub rms_candidate_db: f64,
    pub rms_reference_db: f64,
    pub peak_candidate_db: f64,
    pub peak_reference_db: f64,
    pub delta: Delta,
    /// RMS of the dB difference of long-term average spectra (20 Hz - 20 kHz).
    pub log_spectral_distance_db: f64,
    /// Same, A-weighting-like guitar emphasis (200 Hz - 6 kHz).
    pub band_spectral_distance_db: f64,
}

pub fn db(x: f64) -> f64 {
    20.0 * x.max(1e-12).log10()
}

pub fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

pub fn peak(x: &[f32]) -> f64 {
    x.iter().fold(0.0f32, |m, &v| m.max(v.abs())) as f64
}

/// Estimate the lag of `candidate` relative to `reference` (positive means
/// candidate is late) by FFT cross-correlation, limited to `max_lag`.
pub fn estimate_lag(candidate: &[f32], reference: &[f32], max_lag: usize) -> i64 {
    if max_lag == 0 {
        return 0;
    }
    let n = candidate.len().min(reference.len());
    let size = (n + max_lag).next_power_of_two() * 2;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(size);
    let ifft = planner.plan_fft_inverse(size);
    let mut a = vec![0.0f64; size];
    let mut b = vec![0.0f64; size];
    for i in 0..n {
        a[i] = candidate[i] as f64;
        b[i] = reference[i] as f64;
    }
    let mut fa = fft.make_output_vec();
    let mut fb = fft.make_output_vec();
    fft.process(&mut a, &mut fa).unwrap();
    fft.process(&mut b, &mut fb).unwrap();
    let mut prod: Vec<Complex<f64>> = fa.iter().zip(&fb).map(|(x, y)| x * y.conj()).collect();
    let mut corr = vec![0.0f64; size];
    ifft.process(&mut prod, &mut corr).unwrap();
    let mut best = 0i64;
    let mut best_val = f64::NEG_INFINITY;
    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            size - (-lag) as usize
        };
        if corr[idx] > best_val {
            best_val = corr[idx];
            best = lag;
        }
    }
    best
}

/// Long-term average spectrum in dB over Hann windows of `frame` samples.
fn average_spectrum_db(x: &[f32], frame: usize) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(frame);
    let hop = frame / 2;
    let window: Vec<f64> = (0..frame)
        .map(|i| 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / frame as f64).cos())
        .collect();
    let mut acc = vec![0.0f64; frame / 2 + 1];
    let mut count = 0usize;
    let mut buf = vec![0.0f64; frame];
    let mut spec = fft.make_output_vec();
    let mut start = 0;
    while start + frame <= x.len() {
        for i in 0..frame {
            buf[i] = x[start + i] as f64 * window[i];
        }
        fft.process(&mut buf, &mut spec).unwrap();
        for (a, s) in acc.iter_mut().zip(&spec) {
            *a += s.norm_sqr();
        }
        count += 1;
        start += hop;
    }
    if count == 0 {
        return acc;
    }
    acc.iter()
        .map(|&p| 10.0 * (p / count as f64).max(1e-20).log10())
        .collect()
}

fn spectral_distance(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    let frame = 4096.min(candidate.len().next_power_of_two().max(256));
    let a = average_spectrum_db(candidate, frame);
    let b = average_spectrum_db(reference, frame);
    let bin_hz = sample_rate / frame as f64;
    let mut sum = 0.0;
    let mut n = 0usize;
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        let f = k as f64 * bin_hz;
        if f >= lo && f <= hi {
            sum += (x - y) * (x - y);
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64).sqrt()
    }
}

pub fn quality(
    candidate: &[f32],
    reference: &[f32],
    sample_rate: f64,
    max_lag: usize,
    match_gain: bool,
    tolerance: f32,
) -> Quality {
    let lag = estimate_lag(candidate, reference, max_lag);
    // Align: drop `lag` samples from the late signal.
    let (c, r): (&[f32], &[f32]) = if lag >= 0 {
        (&candidate[lag as usize..], reference)
    } else {
        (candidate, &reference[(-lag) as usize..])
    };
    let n = c.len().min(r.len());
    let (c, r) = (&c[..n], &r[..n]);
    let gain = if match_gain {
        let dot: f64 = c.iter().zip(r).map(|(&x, &y)| x as f64 * y as f64).sum();
        let cc: f64 = c.iter().map(|&x| x as f64 * x as f64).sum();
        if cc > 0.0 {
            dot / cc
        } else {
            1.0
        }
    } else {
        1.0
    };
    let scaled: Vec<f32> = c.iter().map(|&x| (x as f64 * gain) as f32).collect();
    Quality {
        lag_samples: lag,
        gain_db: db(gain.abs()),
        rms_candidate_db: db(rms(candidate)),
        rms_reference_db: db(rms(reference)),
        peak_candidate_db: db(peak(candidate)),
        peak_reference_db: db(peak(reference)),
        delta: delta(&scaled, r, tolerance),
        log_spectral_distance_db: spectral_distance(
            &scaled,
            r,
            sample_rate,
            20.0,
            20_000.0_f64.min(sample_rate / 2.0),
        ),
        band_spectral_distance_db: spectral_distance(&scaled, r, sample_rate, 200.0, 6_000.0),
    }
}
