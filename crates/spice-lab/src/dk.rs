//! The reduced linear system of a circuit, for training something that
//! stands in for its nonlinear solve.
//!
//! Writes the system as JSON (see robine-spice's `Processor::linear_system`
//! for what the matrix holds). With `--input`, also runs a WAV through the
//! circuit with its own solver and writes, for every sample, the drive of
//! each nonlinear port and the current the solver answered with: one row
//! of `2 ports` little-endian f64 per sample.

use crate::commands::{build, load_with, positions, CmdResult};
use clap::Args;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use wavelab::wav;

#[derive(Args)]
pub struct DkArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Sample rate.
    #[arg(long, default_value_t = 48_000.0)]
    rate: f64,
    /// Control position, `name=value`. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Where the system goes (JSON).
    #[arg(short, long)]
    output: PathBuf,
    /// A WAV to run through the circuit (first channel, 1.0 = 1 V; its
    /// rate must be `--rate`).
    #[arg(long)]
    input: Option<PathBuf>,
    /// Where the drives and currents of that run go.
    #[arg(long)]
    trace: Option<PathBuf>,
}

fn list(values: &[f64]) -> String {
    let mut out = String::from("[");
    for (k, v) in values.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v:e}");
    }
    out.push(']');
    out
}

pub fn dk(args: DkArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let positions = positions(&circuit, &args.sets)?;
    let mut processor = build(&circuit, args.rate, &positions)?;
    let system = processor.linear_system();
    if system.drive != system.states + 2 {
        return Err("circuits with delay lines or periodic sources are not supported".into());
    }
    let json = format!(
        "{{\"rate\":{},\"ports\":{},\"states\":{},\"drive\":{},\"matrix\":{},\"gain\":{},\"rest\":{},\"rest_currents\":{}}}\n",
        args.rate,
        system.ports,
        system.states,
        system.drive,
        list(&system.matrix),
        list(&system.gain),
        list(&system.rest),
        list(&system.rest_currents)
    );
    std::fs::write(&args.output, json).map_err(|e| format!("{}: {e}", args.output.display()))?;
    println!(
        "{}: {} ports, {} states",
        args.output.display(),
        system.ports,
        system.states
    );

    if let Some(input) = &args.input {
        let trace = args.trace.as_ref().ok_or("--input needs --trace")?;
        let audio = wav::read_mono(input, 0)?;
        let (samples, rate) = (audio.samples, audio.sample_rate);
        if (f64::from(rate) - args.rate).abs() > 0.5 {
            return Err(format!(
                "{} is at {rate} Hz, not {}",
                input.display(),
                args.rate
            ));
        }
        let mut bytes = Vec::with_capacity(samples.len() * system.ports * 16);
        for &x in &samples {
            let drive = processor.port_drive(x).to_vec();
            processor.process_sample(x);
            for v in drive.iter().chain(processor.port_currents()) {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::File::create(trace)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("{}: {e}", trace.display()))?;
        let failures = processor.solver_stats().failures;
        println!(
            "{}: {} samples, {} unconverged",
            trace.display(),
            samples.len(),
            failures
        );
        return Ok(failures == 0);
    }
    Ok(true)
}
