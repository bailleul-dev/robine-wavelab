//! How much aliasing a circuit makes at a given sample rate.
//!
//! A sine goes through the circuit; what comes out is steady and periodic,
//! so its spectrum holds the fundamental and its harmonics and nothing else,
//! except what the sampling folded back: harmonics past Nyquist land between
//! the true ones. The sine is chosen so that the analysis window holds a
//! whole number of its periods (no leakage, rectangular window), with an odd
//! number of them, so that folded harmonics never land on true ones.
//!
//! The same is done at a multiple of the rate, on the same frequencies and
//! the same window length in seconds: that run folds much less, and is the
//! floor the base rate is judged against.

use crate::commands::{build, load_with, positions};
use clap::Args;
use realfft::RealFftPlanner;
use std::path::PathBuf;

#[derive(Args)]
pub struct AliasArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Sample rate under test.
    #[arg(long, default_value_t = 48_000.0)]
    rate: f64,
    /// The reference runs at this multiple of the rate.
    #[arg(long, default_value_t = 8)]
    factor: usize,
    /// Test frequencies in Hz, each moved to the nearest one that fits the
    /// window an odd number of times.
    #[arg(long, value_delimiter = ',', default_values_t = [110.0, 440.0, 1000.0, 2500.0, 5000.0])]
    frequencies: Vec<f64>,
    /// Peak input voltages.
    #[arg(long, value_delimiter = ',', default_values_t = [0.1, 0.3, 1.0])]
    peaks: Vec<f64>,
    /// Control position, `name=value`. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Seconds run before the analysis, for the circuit to settle.
    #[arg(long, default_value_t = 0.5)]
    settle: f64,
    /// Analysis window, in samples at the base rate (a power of two).
    #[arg(long, default_value_t = 1 << 15)]
    window: usize,
    /// Upper edge of the band looked at, in Hz.
    #[arg(long, default_value_t = 20_000.0)]
    band: f64,
}

/// Energies in the band, from a spectrum whose bin `k` is `k` periods per
/// window.
struct Split {
    fundamental: f64,
    harmonics: f64,
    /// Everything that is neither the fundamental, a harmonic nor DC.
    folded: f64,
    /// The strongest single folded bin.
    worst: f64,
}

fn split(power: &[f64], k0: usize, top: usize) -> Split {
    let mut s = Split {
        fundamental: power[k0],
        harmonics: 0.0,
        folded: 0.0,
        worst: 0.0,
    };
    for (k, &p) in power.iter().enumerate().take(top + 1).skip(1) {
        if k == k0 {
        } else if k.is_multiple_of(k0) {
            s.harmonics += p;
        } else {
            s.folded += p;
            s.worst = s.worst.max(p);
        }
    }
    s
}

fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut input = signal.to_vec();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).expect("fft sizes");
    spectrum.iter().map(|c| c.norm_sqr()).collect()
}

/// The steady output over the window, at `rate`, for a sine of `k0`
/// periods per window of `n` samples.
fn steady(
    circuit: &robine_spice::Circuit,
    positions: &[f64],
    rate: f64,
    n: usize,
    k0: usize,
    peak: f64,
    settle: f64,
) -> Result<(Vec<f64>, u64), String> {
    let mut processor = build(circuit, rate, positions)?;
    let lead = (settle * rate).ceil() as usize;
    let lead = lead.div_ceil(n) * n; // whole windows, so the phase lines up
    let step = std::f64::consts::TAU * k0 as f64 / n as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..lead + n {
        let y = processor.process_sample((peak * (step * i as f64).sin()) as f32);
        if i >= lead {
            out.push(f64::from(y));
        }
    }
    Ok((out, processor.solver_stats().failures))
}

fn db10(x: f64) -> f64 {
    10.0 * x.max(1e-300).log10()
}

pub fn alias(args: AliasArgs) -> crate::commands::CmdResult {
    if !args.window.is_power_of_two() {
        return Err("--window must be a power of two".into());
    }
    let circuit = load_with(&args.circuit, &args.source)?;
    let positions = positions(&circuit, &args.sets)?;
    let n = args.window;
    let reference_rate = args.rate * args.factor as f64;
    let top = (args.band * n as f64 / args.rate).floor() as usize;
    let controls: Vec<String> = circuit
        .controls()
        .iter()
        .zip(&positions)
        .map(|(c, p)| format!("{}={p}", c.name))
        .collect();
    println!(
        "{} — {} Hz against {} Hz, window {} samples ({:.3} s), band to {} Hz, {}",
        circuit.title(),
        args.rate,
        reference_rate,
        n,
        n as f64 / args.rate,
        args.band,
        controls.join(" ")
    );
    println!("folded: energy between the harmonics, relative to the fundamental; worst: the strongest single folded line");
    println!("| f Hz | peak V | THD dB | folded dB | worst dB | reference folded dB | reference worst dB | unconverged |");
    println!("|-----:|-------:|-------:|----------:|---------:|--------------------:|-------------------:|------------:|");
    let mut all_converged = true;
    for &frequency in &args.frequencies {
        let mut k0 = (frequency * n as f64 / args.rate).round().max(1.0) as usize;
        if k0.is_multiple_of(2) {
            k0 += 1;
        }
        let f0 = k0 as f64 * args.rate / n as f64;
        for &peak in &args.peaks {
            let (base, base_fail) =
                steady(&circuit, &positions, args.rate, n, k0, peak, args.settle)?;
            let (reference, reference_fail) = steady(
                &circuit,
                &positions,
                reference_rate,
                n * args.factor,
                k0,
                peak,
                args.settle,
            )?;
            all_converged &= base_fail == 0 && reference_fail == 0;
            let b = split(&power_spectrum(&base), k0, top);
            let r = split(&power_spectrum(&reference), k0, top);
            println!(
                "| {:.1} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} |",
                f0,
                peak,
                db10(b.harmonics / b.fundamental),
                db10(b.folded / b.fundamental),
                db10(b.worst / b.fundamental),
                db10(r.folded / r.fundamental),
                db10(r.worst / r.fundamental),
                base_fail + reference_fail
            );
        }
    }
    Ok(all_converged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pure_sine_folds_nothing_and_a_square_folds() {
        let n = 1 << 12;
        let k0 = 101;
        let step = std::f64::consts::TAU * k0 as f64 / n as f64;
        let sine: Vec<f64> = (0..n).map(|i| (step * i as f64).sin()).collect();
        let s = split(&power_spectrum(&sine), k0, n / 2);
        assert!(db10(s.folded / s.fundamental) < -200.0);
        assert!(db10(s.harmonics / s.fundamental) < -200.0);
        // A sampled square has odd harmonics up to infinity: those past
        // Nyquist come back between the true ones.
        let square: Vec<f64> = sine.iter().map(|v| v.signum()).collect();
        let s = split(&power_spectrum(&square), k0, n / 2);
        assert!(db10(s.harmonics / s.fundamental) > -10.0);
        assert!(db10(s.folded / s.fundamental) > -40.0);
    }
}
