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
//! floor the base rate is judged against. It is also what the base rate
//! should sound like: the fundamental and the harmonics are compared with
//! it, amplitude and phase, so that a cure for folding that changes the
//! sound shows.
//!
//! `--oracle` asks what the base rate could do at best if something stood
//! in for its nonlinear solve: the base run then solves nothing, and its
//! nonlinear ports are given the currents of the reference run, taken at
//! the base rate's instants, as they are (`sampled`) or first low-passed
//! below half the base rate (`lowpass`).

use crate::commands::{build, load_with, positions};
use clap::{Args, ValueEnum};
use realfft::num_complex::Complex;
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
    /// Average device currents over each sample interval at the base rate
    /// (robine-spice's `set_antialiasing`). The reference never does.
    #[arg(long)]
    antialiasing: bool,
    /// Instead of solving, give the base run's nonlinear ports the
    /// reference run's currents.
    #[arg(long, value_enum)]
    oracle: Option<Oracle>,
    /// With `--oracle lowpass`, how far ahead the low-pass may look, in
    /// samples at the base rate (it is cut there and renormalised). Without
    /// it, the filter is symmetric.
    #[arg(long)]
    lookahead: Option<f64>,
    /// A trained stand-in for one nonlinear port's solve (JSON, from the
    /// training scripts), used at the base rate; the other ports are held
    /// at their resting current.
    #[arg(long)]
    model: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Oracle {
    /// The reference currents at the base rate's instants.
    Sampled,
    /// The same, low-passed first (zero phase, cut at half the base rate).
    Lowpass,
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

/// Spectrum scaled by the length, so that runs of different lengths over
/// the same time compare bin for bin.
fn spectrum(signal: &[f64]) -> Vec<Complex<f64>> {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut input = signal.to_vec();
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).expect("fft sizes");
    let scale = 1.0 / signal.len() as f64;
    spectrum.iter().map(|c| c * scale).collect()
}

fn norms(spectrum: &[Complex<f64>]) -> Vec<f64> {
    spectrum.iter().map(|c| c.norm_sqr()).collect()
}

/// Energy of the difference between two spectra over the fundamental and
/// its harmonics in the band, relative to the second one's fundamental,
/// ours first delayed by `delay` (in windows: bin `k` turns by `k delay`
/// periods).
fn harmonic_error(
    ours: &[Complex<f64>],
    reference: &[Complex<f64>],
    k0: usize,
    top: usize,
    delay: f64,
) -> f64 {
    let error: f64 = (k0..=top)
        .step_by(k0)
        .map(|k| {
            let turn = Complex::from_polar(1.0, -std::f64::consts::TAU * k as f64 * delay);
            (ours[k] * turn - reference[k]).norm_sqr()
        })
        .sum();
    error / reference[k0].norm_sqr()
}

/// The delay, within two samples either way, that brings ours closest to
/// the reference, in samples, and the error left.
fn best_delay(
    ours: &[Complex<f64>],
    reference: &[Complex<f64>],
    k0: usize,
    top: usize,
    window: usize,
) -> (f64, f64) {
    let error = |samples: f64| harmonic_error(ours, reference, k0, top, samples / window as f64);
    let (mut best, mut least) = (0.0, error(0.0));
    for step in -400..=400 {
        let samples = f64::from(step) / 200.0;
        let e = error(samples);
        if e < least {
            (best, least) = (samples, e);
        }
    }
    (best, least)
}

/// Where and how a run samples: its rate, its window in samples, and
/// whether device currents are averaged over each interval.
#[derive(Clone, Copy)]
struct Run {
    rate: f64,
    window: usize,
    antialiasing: bool,
}

/// The test tone: `k0` periods per window, its peak, and how long it plays
/// before the window.
#[derive(Clone, Copy)]
struct Tone {
    k0: usize,
    peak: f64,
    settle: f64,
}

/// What a run produced: the output over the window, the samples Newton
/// failed on, and, if asked for, the nonlinear port currents of every
/// sample (one row of `ports` values per sample).
struct Steady {
    output: Vec<f64>,
    failures: u64,
    currents: Vec<f64>,
    ports: usize,
}

/// The steady output over the window of `run`. With `imposed`, the
/// nonlinear ports are not solved but given those currents, one row per
/// sample.
fn steady(
    circuit: &robine_spice::Circuit,
    positions: &[f64],
    run: Run,
    tone: Tone,
    record: bool,
    imposed: Option<&[f64]>,
    model: Option<&std::path::Path>,
) -> Result<Steady, String> {
    let Run {
        rate,
        window: n,
        antialiasing,
    } = run;
    let mut processor = build(circuit, rate, positions)?;
    processor.set_antialiasing(antialiasing);
    let ports = processor.nonlinear_port_count();
    let lead = (tone.settle * rate).ceil() as usize;
    let lead = lead.div_ceil(n) * n; // whole windows, so the phase lines up
    let step = std::f64::consts::TAU * tone.k0 as f64 / n as f64;
    let mut output = Vec::with_capacity(n);
    let mut currents = Vec::with_capacity(if record { (lead + n) * ports } else { 0 });
    let rest = processor.linear_system().rest_currents;
    let mut model = match model {
        Some(path) => {
            let m = crate::model::Model::read(path)?;
            let port = m.port;
            Some(m.start(rest[port]))
        }
        None => None,
    };
    let mut given = rest.clone();
    for i in 0..lead + n {
        let x = (tone.peak * (step * i as f64).sin()) as f32;
        let y = match (imposed, model.as_mut()) {
            (Some(imposed), _) => {
                processor.process_sample_with_currents(x, &imposed[i * ports..(i + 1) * ports])
            }
            (None, Some(model)) => {
                let port = model.port();
                let drive = processor.port_drive(x)[port];
                given[port] = model.next(drive);
                processor.process_sample_with_currents(x, &given)
            }
            (None, None) => processor.process_sample(x),
        };
        if record {
            currents.extend_from_slice(processor.port_currents());
        }
        if i >= lead {
            output.push(f64::from(y));
        }
    }
    Ok(Steady {
        output,
        failures: processor.solver_stats().failures,
        currents,
        ports,
    })
}

/// Rows of `ports` currents at `factor` times the base rate, brought to
/// the base rate: every `factor`-th row, after a zero-phase low-pass at
/// `cutoff` (a fraction of the high rate) if one is given.
fn decimate(
    currents: &[f64],
    ports: usize,
    factor: usize,
    cutoff: Option<f64>,
    lookahead: Option<f64>,
) -> Vec<f64> {
    let rows = currents.len() / ports;
    let taps: Vec<f64> = match cutoff {
        None => vec![1.0],
        Some(fc) => {
            // Blackman-windowed sinc, normalised to unit gain at DC.
            let half = 50 * factor;
            let mut h: Vec<f64> = (0..=2 * half)
                .map(|k| {
                    let t = k as f64 - half as f64;
                    let sinc = if t == 0.0 {
                        2.0 * fc
                    } else {
                        (std::f64::consts::TAU * fc * t).sin() / (std::f64::consts::PI * t)
                    };
                    let w = k as f64 / (2 * half) as f64;
                    let blackman = 0.42 - 0.5 * (std::f64::consts::TAU * w).cos()
                        + 0.08 * (2.0 * std::f64::consts::TAU * w).cos();
                    sinc * blackman
                })
                .collect();
            if let Some(ahead) = lookahead {
                let last = half + (ahead * factor as f64).round() as usize;
                h.iter_mut().skip(last + 1).for_each(|v| *v = 0.0);
            }
            let sum: f64 = h.iter().sum();
            h.iter_mut().for_each(|v| *v /= sum);
            h
        }
    };
    let half = taps.len() / 2;
    let mut out = Vec::with_capacity(rows / factor * ports);
    for row in (0..rows).step_by(factor) {
        for port in 0..ports {
            let mut acc = 0.0;
            for (k, h) in taps.iter().enumerate() {
                // Held at the ends.
                let r = (row + k).saturating_sub(half).min(rows - 1);
                acc += h * currents[r * ports + port];
            }
            out.push(acc);
        }
    }
    out
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
        "{} — {} Hz{} against {} Hz, window {} samples ({:.3} s), band to {} Hz, {}",
        circuit.title(),
        args.rate,
        match (args.oracle, args.antialiasing, &args.model) {
            (Some(Oracle::Sampled), ..) => " given the reference's port currents",
            (Some(Oracle::Lowpass), ..) => " given the reference's port currents, low-passed",
            (None, _, Some(_)) => " with a trained stand-in",
            (None, true, None) => " with antialiasing",
            (None, false, None) => "",
        },
        reference_rate,
        n,
        n as f64 / args.rate,
        args.band,
        controls.join(" ")
    );
    println!("folded: energy between the harmonics, relative to the fundamental; worst: the strongest single folded line;");
    println!("harmonic error: the fundamental and harmonics against the reference's, amplitude and phase;");
    println!("aligned: the same once ours is moved by the delay (in samples) that fits best");
    println!("| f Hz | peak V | THD dB | folded dB | worst dB | harmonic error dB | aligned dB | delay | reference folded dB | reference worst dB | unconverged |");
    println!("|-----:|-------:|-------:|----------:|---------:|------------------:|-----------:|------:|--------------------:|-------------------:|------------:|");
    let mut all_converged = true;
    for &frequency in &args.frequencies {
        let mut k0 = (frequency * n as f64 / args.rate).round().max(1.0) as usize;
        if k0.is_multiple_of(2) {
            k0 += 1;
        }
        let f0 = k0 as f64 * args.rate / n as f64;
        for &peak in &args.peaks {
            let base_run = Run {
                rate: args.rate,
                window: n,
                antialiasing: args.antialiasing,
            };
            let reference_run = Run {
                rate: reference_rate,
                window: n * args.factor,
                antialiasing: false,
            };
            let tone = Tone {
                k0,
                peak,
                settle: args.settle,
            };
            let reference = steady(
                &circuit,
                &positions,
                reference_run,
                tone,
                args.oracle.is_some(),
                None,
                None,
            )?;
            let imposed = args.oracle.map(|oracle| {
                let cutoff = match oracle {
                    Oracle::Sampled => None,
                    Oracle::Lowpass => Some(0.5 / args.factor as f64),
                };
                decimate(
                    &reference.currents,
                    reference.ports,
                    args.factor,
                    cutoff,
                    args.lookahead,
                )
            });
            let base = steady(
                &circuit,
                &positions,
                base_run,
                tone,
                false,
                imposed.as_deref(),
                args.model.as_deref(),
            )?;
            let (base_fail, reference_fail) = (base.failures, reference.failures);
            let (base, reference) = (base.output, reference.output);
            all_converged &= base_fail == 0 && reference_fail == 0;
            let (bs, rs) = (spectrum(&base), spectrum(&reference));
            let b = split(&norms(&bs), k0, top);
            let r = split(&norms(&rs), k0, top);
            let (delay, aligned) = best_delay(&bs, &rs, k0, top, n);
            println!(
                "| {:.1} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {:.1} | {:.1} | {} |",
                f0,
                peak,
                db10(b.harmonics / b.fundamental),
                db10(b.folded / b.fundamental),
                db10(b.worst / b.fundamental),
                db10(harmonic_error(&bs, &rs, k0, top, 0.0)),
                db10(aligned),
                delay,
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
        let s = split(&norms(&spectrum(&sine)), k0, n / 2);
        assert!(db10(s.folded / s.fundamental) < -200.0);
        assert!(db10(s.harmonics / s.fundamental) < -200.0);
        // A sampled square has odd harmonics up to infinity: those past
        // Nyquist come back between the true ones.
        let square: Vec<f64> = sine.iter().map(|v| v.signum()).collect();
        let s = split(&norms(&spectrum(&square)), k0, n / 2);
        assert!(db10(s.harmonics / s.fundamental) > -10.0);
        assert!(db10(s.folded / s.fundamental) > -40.0);
    }
}
