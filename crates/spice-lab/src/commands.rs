use crate::{ngspice, schx};
use clap::Args;
use robine_spice::{Circuit, ControlKind, Guitar, Integration, Processor, Source};
use std::path::{Path, PathBuf};
use std::time::Instant;
use wavelab::{signal, wav};

type CmdResult = Result<bool, String>;

fn load(path: &Path) -> Result<Circuit, String> {
    let circuit = Circuit::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    for warning in circuit.warnings() {
        eprintln!("warning: {warning}");
    }
    Ok(circuit)
}

fn parse_source(text: &str) -> Result<Source, String> {
    Ok(match text.to_ascii_lowercase().as_str() {
        "ideal" => Source::Ideal,
        "buffer" => Source::Buffer {
            output_resistance: 470.0,
        },
        "single-coil" => Source::Guitar(Guitar::single_coil()),
        "humbucker" => Source::Guitar(Guitar::humbucker()),
        other => match other.strip_prefix("buffer:").map(str::parse::<f64>) {
            Some(Ok(output_resistance)) => Source::Buffer { output_resistance },
            _ => {
                return Err(format!(
                    "unknown source `{text}` (ideal, buffer, buffer:<ohms>, single-coil, humbucker)"
                ))
            }
        },
    })
}

fn load_with(path: &Path, source: &str) -> Result<Circuit, String> {
    load(path)?
        .with_source(parse_source(source)?)
        .map_err(|e| e.to_string())
}

/// Control positions after applying `name=value` overrides.
fn positions(circuit: &Circuit, sets: &[String]) -> Result<Vec<f64>, String> {
    let mut positions: Vec<f64> = circuit.controls().iter().map(|c| c.value).collect();
    for set in sets {
        let (name, value) = set
            .split_once('=')
            .ok_or_else(|| format!("`{set}`: expected name=value"))?;
        let value: f64 = value.parse().map_err(|_| format!("`{set}`: bad value"))?;
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("`{set}`: positions go from 0 to 1"));
        }
        let index = circuit
            .controls()
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                let known: Vec<&str> = circuit.controls().iter().map(|c| c.name.as_str()).collect();
                format!("no control `{name}` (controls: {})", known.join(", "))
            })?;
        positions[index] = value;
    }
    Ok(positions)
}

/// `ROBINE_SPICE_INTEGRATION=bdf2` switches every command to BDF2.
fn integration() -> Integration {
    match std::env::var("ROBINE_SPICE_INTEGRATION").as_deref() {
        Ok("bdf2") => Integration::Bdf2,
        _ => Integration::Trapezoidal,
    }
}

fn build(circuit: &Circuit, rate: f64, positions: &[f64]) -> Result<Processor, String> {
    let mut processor = circuit
        .build_with(rate, integration())
        .map_err(|e| e.to_string())?;
    for (index, &value) in positions.iter().enumerate() {
        processor.set_control(index, value);
    }
    processor.reset();
    Ok(processor)
}

fn db(ratio: f64) -> f64 {
    20.0 * ratio.max(1e-300).log10()
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(s, n), v| (s + v * v, n + 1));
    (sum / count.max(1) as f64).sqrt()
}

#[derive(Args)]
pub struct ConvertArgs {
    /// LiveSPICE schematic.
    schematic: PathBuf,
    /// Netlist to write (standard output if omitted).
    #[arg(short, long)]
    output: Option<PathBuf>,
}

pub fn convert(args: ConvertArgs) -> CmdResult {
    let xml = std::fs::read_to_string(&args.schematic)
        .map_err(|e| format!("{}: {e}", args.schematic.display()))?;
    let source = args
        .schematic
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let netlist = schx::convert(&xml, &source)?;
    // Refuse to write something the library would not read back.
    let circuit =
        Circuit::parse(&netlist).map_err(|e| format!("converted netlist is invalid: {e}"))?;
    match args.output {
        Some(path) => {
            std::fs::write(&path, &netlist).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!(
                "{}: {} nodes, {} elements, {} control(s)",
                path.display(),
                circuit.node_count(),
                circuit.element_count(),
                circuit.controls().len()
            );
        }
        None => print!("{netlist}"),
    }
    Ok(true)
}

#[derive(Args)]
pub struct InspectArgs {
    /// Netlist.
    circuit: PathBuf,
    /// What drives the input: `ideal`, `buffer` (470 ohm), `buffer:<ohms>`,
    /// `single-coil` or `humbucker` (a passive guitar, which adds the
    /// controls Guitar.Volume and Guitar.Tone).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Sample rate to build at.
    #[arg(long, default_value_t = 48_000.0)]
    rate: f64,
    /// Print the flattened plain-SPICE deck.
    #[arg(long)]
    spice: bool,
}

pub fn inspect(args: InspectArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    if args.spice {
        let defaults = positions(&circuit, &[])?;
        print!("{}", circuit.to_spice(&defaults));
        return Ok(true);
    }
    let (out, reference) = circuit.output_nodes();
    println!("title     {}", circuit.title());
    println!("nodes     {}", circuit.node_count());
    println!("elements  {}", circuit.element_count());
    println!("input     {}", circuit.input_source());
    println!(
        "output    v({out}{})",
        if reference == "0" {
            String::new()
        } else {
            format!(",{reference}")
        }
    );
    for c in circuit.controls() {
        match c.kind {
            ControlKind::Potentiometer { resistance, taper } => println!(
                "control   {:<12} {resistance:>10.0} ohm  {taper:?}  at {}",
                c.name, c.value
            ),
            ControlKind::DelayTime {
                min_seconds,
                max_seconds,
            } => println!(
                "control   {:<12} delay {:.0} to {:.0} ms  at {}",
                c.name,
                min_seconds * 1e3,
                max_seconds * 1e3,
                c.value
            ),
        }
    }
    let mut processor = circuit.build(args.rate).map_err(|e| e.to_string())?;
    let (ports, states) = (processor.nonlinear_port_count(), processor.state_count());
    let lines = processor.delay_line_count();
    println!("\nprecomputed at build time, per sample only:");
    println!(
        "  linear     {states} states + {lines} delay lines + input -> {ports} port voltages, {states} states, {lines} delay lines, output"
    );
    println!(
        "             ({} multiply-adds, no solve)",
        (ports + states + 1 + lines) * (states + 2 + lines + ports)
    );
    println!("  nonlinear  Newton on {ports} unknown(s): v = p + K i(v), K is {ports} x {ports}");
    for (index, c) in circuit.controls().iter().enumerate() {
        println!(
            "  control    {:<12} {}",
            c.name,
            if processor.control_shapes_nonlinear_solve(index) {
                "changes K: part of the nonlinear problem"
            } else {
                "linear only: the nonlinear problem ignores it"
            }
        );
    }
    println!(
        "rest      {:.6} V at the output",
        processor.process_sample(0.0)
    );
    Ok(true)
}

#[derive(Args)]
pub struct RenderArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input: `ideal`, `buffer` (470 ohm), `buffer:<ohms>`,
    /// `single-coil` or `humbucker` (a passive guitar, which adds the
    /// controls Guitar.Volume and Guitar.Tone).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Input WAV (first channel).
    #[arg(short, long)]
    input: PathBuf,
    /// Output WAV (mono 32-bit float).
    #[arg(short, long)]
    output: PathBuf,
    /// Control position, `name=value` with value in 0..1. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Volts at the circuit input for a full-scale sample.
    #[arg(long, default_value_t = 1.0)]
    volts: f32,
    /// Gain applied to the circuit's output voltage, in dB.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    gain_db: f32,
    /// Scale the result so that it peaks at -1 dBFS (overrides --gain-db).
    #[arg(long)]
    normalize: bool,
    /// Move a control back and forth while rendering: `name=low:high:hertz`
    /// (a wah rocked at 1.5 Hz is `Wah=0.1:0.9:1.5`). Repeatable.
    #[arg(long = "sweep")]
    sweeps: Vec<String>,
}

/// A control rocked sinusoidally between two positions.
struct Sweep {
    control: usize,
    low: f64,
    high: f64,
    hertz: f64,
}

fn parse_sweep(circuit: &Circuit, text: &str) -> Result<Sweep, String> {
    let bad = || format!("`{text}`: expected name=low:high:hertz");
    let (name, range) = text.split_once('=').ok_or_else(bad)?;
    let values: Vec<f64> = range
        .split(':')
        .map(|v| v.parse::<f64>().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    let [low, high, hertz] = values[..] else {
        return Err(bad());
    };
    if !(0.0..=1.0).contains(&low) || !(0.0..=1.0).contains(&high) || hertz <= 0.0 {
        return Err(format!(
            "`{text}`: positions go from 0 to 1 and the rate is positive"
        ));
    }
    let control = circuit
        .controls()
        .iter()
        .position(|c| c.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("no control `{name}`"))?;
    Ok(Sweep {
        control,
        low,
        high,
        hertz,
    })
}

pub fn render(args: RenderArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let audio = wav::read_mono(&args.input, 0)?;
    let mut processor = build(
        &circuit,
        f64::from(audio.sample_rate),
        &positions(&circuit, &args.sets)?,
    )?;
    let sweeps: Vec<Sweep> = args
        .sweeps
        .iter()
        .map(|text| parse_sweep(&circuit, text))
        .collect::<Result<_, _>>()?;
    // Remove the output's resting voltage so the file has no DC offset.
    let rest = processor.process_sample(0.0);
    processor.reset();
    let rate = f64::from(audio.sample_rate);
    let started = Instant::now();
    let mut out = Vec::with_capacity(audio.samples.len());
    // Controls move in steps of 16 samples, as a host would do per block.
    for (block, samples) in audio.samples.chunks(16).enumerate() {
        let t = (block * 16) as f64 / rate;
        for sweep in &sweeps {
            let phase = (std::f64::consts::TAU * sweep.hertz * t).sin();
            let middle = 0.5 * (sweep.low + sweep.high);
            let depth = 0.5 * (sweep.high - sweep.low);
            processor.set_control(sweep.control, middle + depth * phase);
        }
        out.extend(
            samples
                .iter()
                .map(|&x| processor.process_sample(x * args.volts) - rest),
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    let volts = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let gain = if args.normalize {
        10f32.powf(-1.0 / 20.0) / volts.max(1e-9)
    } else {
        10f32.powf(args.gain_db / 20.0)
    };
    for v in &mut out {
        *v *= gain;
    }
    wav::write_mono_f32(&args.output, &out, audio.sample_rate)?;
    let stats = processor.solver_stats();
    let peak = volts;
    eprintln!(
        "{}: {:.2} s at {} Hz, {:.3} V peak out of the circuit, {:.1}x real time, {:.2} iterations/sample, {} unconverged",
        args.output.display(),
        out.len() as f64 / f64::from(audio.sample_rate),
        audio.sample_rate,
        peak,
        out.len() as f64 / f64::from(audio.sample_rate) / elapsed,
        stats.iterations as f64 / out.len().max(1) as f64,
        stats.failures
    );
    Ok(stats.failures == 0)
}

#[derive(Args)]
pub struct CompareArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input: `ideal`, `buffer` (470 ohm), `buffer:<ohms>`,
    /// `single-coil` or `humbucker` (a passive guitar, which adds the
    /// controls Guitar.Volume and Guitar.Tone).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Length of the test signal in seconds.
    #[arg(long, default_value_t = 0.05)]
    seconds: f64,
    /// Peak input voltage reached at the end of the signal.
    #[arg(long, default_value_t = 1.0)]
    peak: f64,
    /// Sample rates to run both simulators at.
    #[arg(long, value_delimiter = ',', default_values_t = [48_000.0, 96_000.0, 192_000.0, 384_000.0])]
    rates: Vec<f64>,
    /// Control position, `name=value`. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Fail unless the error at the highest rate is below this (dB relative to the signal).
    #[arg(long, default_value_t = -70.0, allow_negative_numbers = true)]
    threshold_db: f64,
    /// Where the decks and ngspice output go.
    #[arg(long, default_value = "target/compare")]
    work_dir: PathBuf,
    /// Tightest ngspice `reltol` to try first (it falls back to looser ones).
    #[arg(long, default_value_t = 1e-7)]
    ngspice_reltol: f64,
    /// Write `time,input,ngspice,robine` for every rate to `<work-dir>/<rate>/trace.csv`.
    #[arg(long)]
    trace: bool,
}

struct Delta {
    max_abs: f64,
    relative_db: f64,
    signal_rms: f64,
}

fn delta(ours: &[f64], reference: &[f64]) -> Delta {
    let mean = reference.iter().sum::<f64>() / reference.len() as f64;
    let signal_rms = rms(reference.iter().map(|v| v - mean));
    let error_rms = rms(ours.iter().zip(reference).map(|(a, b)| a - b));
    Delta {
        max_abs: ours
            .iter()
            .zip(reference)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs())),
        relative_db: db(error_rms / signal_rms.max(1e-300)),
        signal_rms,
    }
}

fn run_ours(processor: &mut Processor, input: &[f64]) -> Vec<f64> {
    input
        .iter()
        .map(|&x| f64::from(processor.process_sample(x as f32)))
        .collect()
}

pub fn compare(args: CompareArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let positions = positions(&circuit, &args.sets)?;
    println!(
        "{} — {} s, {} V peak, robine-spice vs ngspice (same flattened deck)",
        circuit.title(),
        args.seconds,
        args.peak
    );
    println!(
        "| rate Hz | max error V | error dB | output RMS V | iterations/sample | unconverged | ngspice reltol |"
    );
    println!(
        "|--------:|------------:|---------:|-------------:|------------------:|------------:|---------------:|"
    );
    let mut last = f64::INFINITY;
    let mut failures = 0;
    for &rate in &args.rates {
        // Both simulators are given the f32 samples the processor consumes.
        let input: Vec<f64> = signal::sample(rate, args.seconds, args.peak)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
        let reference = ngspice::run(
            &circuit,
            &positions,
            &input,
            rate,
            &args.work_dir.join(format!("{rate}")),
            args.ngspice_reltol,
        )?;
        let (reference, reltol) = (reference.output, reference.reltol);
        let mut processor = build(&circuit, rate, &positions)?;
        let ours = run_ours(&mut processor, &input);
        if args.trace {
            let path = args.work_dir.join(format!("{rate}")).join("trace.csv");
            let mut csv = String::from("time,input,ngspice,robine\n");
            for (k, ((i, r), o)) in input.iter().zip(&reference).zip(&ours).enumerate() {
                csv.push_str(&format!("{:e},{i:e},{r:e},{o:e}\n", k as f64 / rate));
            }
            std::fs::write(&path, csv).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        let d = delta(&ours, &reference);
        let stats = processor.solver_stats();
        println!(
            "| {rate:.0} | {:.3e} | {:.1} | {:.4} | {:.2} | {} | {reltol:e} |",
            d.max_abs,
            d.relative_db,
            d.signal_rms,
            stats.iterations as f64 / input.len() as f64,
            stats.failures
        );
        last = d.relative_db;
        failures += stats.failures;
    }
    let pass = last <= args.threshold_db && failures == 0;
    println!(
        "\n{}: {last:.1} dB at the highest rate (threshold {:.1} dB), {failures} unconverged sample(s)",
        if pass { "PASS" } else { "FAIL" },
        args.threshold_db
    );
    Ok(pass)
}

#[derive(Args)]
pub struct BenchArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input: `ideal`, `buffer` (470 ohm), `buffer:<ohms>`,
    /// `single-coil` or `humbucker` (a passive guitar, which adds the
    /// controls Guitar.Volume and Guitar.Tone).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Sample rates to measure.
    #[arg(long, value_delimiter = ',', default_values_t = [48_000.0, 96_000.0, 192_000.0])]
    rates: Vec<f64>,
    /// Seconds of audio per measurement.
    #[arg(long, default_value_t = 5.0)]
    seconds: f64,
    /// Peak input voltage.
    #[arg(long, default_value_t = 1.0)]
    peak: f64,
    /// Measurements per rate; the fastest is reported too.
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// Control position, `name=value`. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
}

pub fn bench(args: BenchArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let positions = positions(&circuit, &args.sets)?;
    println!("{}", circuit.title());
    println!("| rate Hz | ns/sample | best | RTF | CPU % | iterations/sample | unconverged | set_control µs |");
    println!("|--------:|----------:|-----:|----:|------:|------------------:|------------:|---------------:|");
    let mut clean = true;
    for &rate in &args.rates {
        let input: Vec<f32> = signal::sample(rate, args.seconds, args.peak)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let mut output = vec![0.0f32; input.len()];
        let mut processor = build(&circuit, rate, &positions)?;
        let (mut total, mut best) = (0.0, f64::INFINITY);
        for _ in 0..args.iters.max(1) {
            processor.reset();
            let started = Instant::now();
            for (i, o) in input.chunks(64).zip(output.chunks_mut(64)) {
                processor.process(i, o);
            }
            let ns = started.elapsed().as_secs_f64() * 1e9 / input.len() as f64;
            std::hint::black_box(&output);
            total += ns;
            best = best.min(ns);
        }
        let mean = total / args.iters.max(1) as f64;
        let stats = processor.solver_stats();
        let samples = (input.len() * args.iters.max(1)) as f64;

        // Cost of a knob move: alternate between two positions.
        let control_us = if positions.is_empty() {
            f64::NAN
        } else {
            let rounds = 20_000;
            let started = Instant::now();
            for k in 0..rounds {
                let index = k % positions.len();
                processor.set_control(
                    index,
                    if (k / positions.len()) % 2 == 0 {
                        0.25
                    } else {
                        0.75
                    },
                );
            }
            started.elapsed().as_secs_f64() * 1e6 / rounds as f64
        };
        println!(
            "| {rate:.0} | {mean:.0} | {best:.0} | {:.0} | {:.2} | {:.2} | {} | {control_us:.2} |",
            1e9 / (mean * rate),
            mean * rate / 1e7,
            stats.iterations as f64 / samples,
            stats.failures
        );
        clean &= stats.failures == 0;
    }
    Ok(clean)
}

#[derive(Args)]
pub struct GoldenArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// Directory the fixtures are written to.
    #[arg(short, long)]
    output: PathBuf,
    /// Base name of the fixture files.
    #[arg(long)]
    name: String,
    /// Sample rate of the fixture.
    #[arg(long, default_value_t = 192_000.0)]
    rate: f64,
    /// Length in seconds.
    #[arg(long, default_value_t = 0.03)]
    seconds: f64,
    /// Peak input voltage.
    #[arg(long, default_value_t = 1.0)]
    peak: f64,
    /// Control position, `name=value`. Repeatable.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Tightest ngspice `reltol` to try first (it falls back to looser ones).
    #[arg(long, default_value_t = 1e-7)]
    ngspice_reltol: f64,
    #[arg(long, default_value = "target/golden")]
    work_dir: PathBuf,
}

pub fn golden(args: GoldenArgs) -> CmdResult {
    let circuit = load(&args.circuit)?;
    let positions = positions(&circuit, &args.sets)?;
    let input: Vec<f32> = signal::sample(args.rate, args.seconds, args.peak)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let as_f64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
    let reference = ngspice::run(
        &circuit,
        &positions,
        &as_f64,
        args.rate,
        &args.work_dir,
        args.ngspice_reltol,
    )?
    .output;
    let mut processor = build(&circuit, args.rate, &positions)?;
    let d = delta(&run_ours(&mut processor, &as_f64), &reference);

    std::fs::create_dir_all(&args.output).map_err(|e| e.to_string())?;
    let write = |suffix: &str, samples: &[f32]| -> Result<(), String> {
        let path = args.output.join(format!("{}.{suffix}.f32", args.name));
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
    };
    write("in", &input)?;
    write(
        "ngspice",
        &reference.iter().map(|&v| v as f32).collect::<Vec<_>>(),
    )?;
    println!(
        "{}: {} samples at {} Hz, controls {:?}; robine-spice is {:.1} dB from ngspice (max {:.2e} V)",
        args.name,
        input.len(),
        args.rate,
        positions,
        d.relative_db,
        d.max_abs
    );
    Ok(true)
}

#[derive(Args)]
pub struct SurveyArgs {
    /// Directory of netlists (searched one level deep).
    #[arg(default_value = "circuits")]
    directory: PathBuf,
    /// Peak input voltage.
    #[arg(long, default_value_t = 0.5)]
    peak: f64,
    /// Rates at which each circuit is compared with ngspice.
    #[arg(long, value_delimiter = ',', default_values_t = [48_000.0, 192_000.0])]
    rates: Vec<f64>,
    /// Length of the comparison signal in seconds.
    #[arg(long, default_value_t = 0.02)]
    seconds: f64,
    /// Tightest ngspice `reltol` to try first.
    #[arg(long, default_value_t = 1e-5)]
    ngspice_reltol: f64,
    /// Skip the ngspice comparison.
    #[arg(long)]
    no_ngspice: bool,
    #[arg(long, default_value = "target/survey")]
    work_dir: PathBuf,
}

fn netlists(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let list = |dir: &Path| -> Result<Vec<PathBuf>, String> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        Ok(entries)
    };
    let mut found = Vec::new();
    for entry in list(directory)? {
        if entry.is_dir() {
            found.extend(
                list(&entry)?
                    .into_iter()
                    .filter(|p| p.extension().is_some_and(|e| e == "cir")),
            );
        } else if entry.extension().is_some_and(|e| e == "cir") {
            found.push(entry);
        }
    }
    found.sort_by_key(|p| {
        (
            p.parent().map(Path::to_path_buf),
            p.file_name().map(|n| n.to_os_string()),
        )
    });
    Ok(found)
}

/// Every circuit of a directory: does it run, how fast, and how far from
/// ngspice.
pub fn survey(args: SurveyArgs) -> CmdResult {
    let mut header = String::from(
        "| circuit | ports | states | controls | ns/sample | CPU % | unconverged | output RMS V |",
    );
    let mut rule = String::from(
        "|---------|------:|-------:|---------:|----------:|------:|------------:|-------------:|",
    );
    if !args.no_ngspice {
        for rate in &args.rates {
            header.push_str(&format!(" vs ngspice {} kHz |", rate / 1e3));
            rule.push_str("---:|");
        }
    }
    println!("{header}\n{rule}");
    let mut clean = true;
    for path in netlists(&args.directory)? {
        let name = path
            .strip_prefix(&args.directory)
            .unwrap_or(&path)
            .with_extension("")
            .display()
            .to_string();
        let circuit = match Circuit::read(&path) {
            Ok(c) => c,
            Err(e) => {
                println!("| {name} | {e} |");
                clean = false;
                continue;
            }
        };
        let positions: Vec<f64> = circuit.controls().iter().map(|c| c.value).collect();
        let rate = 48_000.0;
        let mut processor = match build(&circuit, rate, &positions) {
            Ok(p) => p,
            Err(e) => {
                println!("| {name} | {e} |");
                clean = false;
                continue;
            }
        };
        let input: Vec<f32> = signal::sample(rate, 1.0, args.peak)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let mut output = vec![0.0f32; input.len()];
        let started = Instant::now();
        processor.process(&input, &mut output);
        let ns = started.elapsed().as_secs_f64() * 1e9 / input.len() as f64;
        let stats = processor.solver_stats();
        let finite = output.iter().all(|v| v.is_finite());
        let mean = output.iter().map(|&v| f64::from(v)).sum::<f64>() / output.len() as f64;
        let level = rms(output.iter().map(|&v| f64::from(v) - mean));
        clean &= stats.failures == 0 && finite;
        let mut row = format!(
            "| {name} | {} | {} | {} | {ns:.0} | {:.2} | {} | {} |",
            processor.nonlinear_port_count(),
            processor.state_count(),
            positions.len(),
            ns * rate / 1e7,
            stats.failures,
            if finite {
                format!("{level:.4}")
            } else {
                "not finite".to_string()
            },
        );
        if !args.no_ngspice {
            for &rate in &args.rates {
                let input: Vec<f64> = signal::sample(rate, args.seconds, args.peak)
                    .into_iter()
                    .map(|v| f64::from(v as f32))
                    .collect();
                let dir = args.work_dir.join(&name).join(format!("{rate}"));
                let cell = match ngspice::run(
                    &circuit,
                    &positions,
                    &input,
                    rate,
                    &dir,
                    args.ngspice_reltol,
                ) {
                    Ok(reference) => {
                        let mut processor = build(&circuit, rate, &positions)?;
                        let d = delta(&run_ours(&mut processor, &input), &reference.output);
                        if d.signal_rms < 1e-9 {
                            format!("{:.1e} V", d.max_abs)
                        } else {
                            format!("{:.1} dB", d.relative_db)
                        }
                    }
                    Err(e) if e.contains("did not finish") => "timeout".to_string(),
                    Err(_) => "ngspice failed".to_string(),
                };
                row.push_str(&format!(" {cell} |"));
            }
        }
        println!("{row}");
    }
    Ok(clean)
}
