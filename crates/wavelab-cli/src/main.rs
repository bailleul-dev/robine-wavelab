use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use wavelab::{judge, null, peaq, wav};

type CmdResult = Result<bool, String>;

/// Measures on audio files, whatever produced them: a circuit simulation, a
/// neural capture, a real device recorded through an interface.
#[derive(Parser)]
#[command(name = "wavelab", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the probe signal that `judge` knows how to read.
    Probe(ProbeArgs),
    /// Judge whether two renders of the probe come from devices that behave alike.
    Judge(JudgeArgs),
    /// Subtract one render from another and rate what is left.
    Null(NullArgs),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Probe(a) => probe(a),
        Command::Judge(a) => judge_files(a),
        Command::Null(a) => null_files(a),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

#[derive(Args)]
struct ProbeArgs {
    /// WAV to write (mono 32-bit float).
    #[arg(short, long)]
    output: PathBuf,
    /// Sample rate.
    #[arg(long, default_value_t = 48_000)]
    rate: u32,
}

fn probe(args: ProbeArgs) -> CmdResult {
    let samples = judge::probe(args.rate);
    wav::write_mono_f32(&args.output, &samples, args.rate)?;
    eprintln!(
        "{}: {:.1} s at {} Hz, loudest tone at {:.1} dBFS. Pass it through each device, then `spice-lab judge`.",
        args.output.display(),
        samples.len() as f64 / f64::from(args.rate),
        args.rate,
        20.0 * samples.iter().fold(0.0f32, |m, v| m.max(v.abs())).log10()
    );
    Ok(true)
}

#[derive(Args)]
struct JudgeArgs {
    /// The probe as rendered by the device taken as the truth.
    #[arg(short, long)]
    reference: PathBuf,
    /// The probe as rendered by the device being judged.
    #[arg(short, long)]
    candidate: PathBuf,
    /// Tolerance on the frequency response, dB.
    #[arg(long, default_value_t = 1.0)]
    response_db: f64,
    /// Tolerance on the output level at each input level, dB.
    #[arg(long, default_value_t = 1.0)]
    dynamics_db: f64,
    /// Tolerance on each harmonic, dB.
    #[arg(long, default_value_t = 2.0)]
    harmonics_db: f64,
    /// Tolerance on the attack, dB.
    #[arg(long, default_value_t = 1.5)]
    attack_db: f64,
    /// Tolerance on the gain after a burst, dB.
    #[arg(long, default_value_t = 1.5)]
    recovery_db: f64,
}

fn judge_files(args: JudgeArgs) -> CmdResult {
    let reference = wav::read_mono(&args.reference, 0)?;
    let candidate = wav::read_mono(&args.candidate, 0)?;
    if reference.sample_rate != candidate.sample_rate {
        return Err(format!(
            "the two files have different sample rates ({} and {} Hz)",
            reference.sample_rate, candidate.sample_rate
        ));
    }
    let measure = |audio: &wav::Audio, path: &Path| {
        judge::measure(&audio.samples, audio.sample_rate)
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    let tolerance = judge::Tolerance {
        response_db: args.response_db,
        dynamics_db: args.dynamics_db,
        harmonics_db: args.harmonics_db,
        attack_db: args.attack_db,
        recovery_db: args.recovery_db,
        ..Default::default()
    };
    println!(
        "reference: {}\ncandidate: {}\n",
        args.reference.display(),
        args.candidate.display()
    );
    Ok(judge::report(
        &measure(&reference, &args.reference)?,
        &measure(&candidate, &args.candidate)?,
        &tolerance,
    ))
}

#[derive(Args)]
struct NullArgs {
    /// The render taken as the truth.
    #[arg(short, long)]
    reference: PathBuf,
    /// The render compared with it.
    #[arg(short, long)]
    candidate: PathBuf,
    /// Write the residual here, at its true level relative to the reference
    /// (mono 32-bit float).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Fail if the error-to-signal ratio is above this, dB.
    #[arg(long, allow_negative_numbers = true)]
    max_esr_db: Option<f64>,
    /// Fail if the noise-to-mask ratio is above this, dB.
    #[arg(long, allow_negative_numbers = true)]
    max_nmr_db: Option<f64>,
    /// Fail if the difference rises above the masking threshold in any
    /// band of any frame.
    #[arg(long)]
    inaudible: bool,
    /// Also grade the difference with PEAQ (ITU-R BS.1387, basic version)
    /// through GstPEAQ, which `make peaq` builds. 48 kHz files only.
    #[arg(long)]
    peaq: bool,
    /// Fail if PEAQ's objective difference grade is below this (0 is
    /// imperceptible, -1 perceptible but not annoying, -4 very annoying).
    #[arg(long, allow_negative_numbers = true)]
    min_odg: Option<f64>,
    /// Print one line of JSON instead of the tables.
    #[arg(long)]
    json: bool,
}

fn null_files(args: NullArgs) -> CmdResult {
    let reference = wav::read_mono(&args.reference, 0)?;
    let candidate = wav::read_mono(&args.candidate, 0)?;
    if reference.sample_rate != candidate.sample_rate {
        return Err(format!(
            "the two files have different sample rates ({} and {} Hz)",
            reference.sample_rate, candidate.sample_rate
        ));
    }
    let n = null::null(
        &reference.samples,
        &candidate.samples,
        reference.sample_rate,
    );
    let grade = match args.peaq || args.min_odg.is_some() {
        true => Some(peaq::grade(
            &n.reference,
            &n.candidate,
            reference.sample_rate,
        )?),
        false => None,
    };
    let pass = args.max_esr_db.is_none_or(|limit| n.overall_db <= limit)
        && args.max_nmr_db.is_none_or(|limit| n.nmr.total_db <= limit)
        && (!args.inaudible || n.nmr.disturbed == 0.0)
        && args
            .min_odg
            .is_none_or(|limit| grade.is_some_and(|g| g.0 >= limit));
    if let Some(path) = &args.output {
        wav::write_mono_f32(path, &n.residual, reference.sample_rate)?;
    }
    if args.json {
        println!(
            "{{\"esr_db\":{:.2},\"esr_preemphasis_db\":{:.2},\"nmr_db\":{:.2},\"nmr_worst_db\":{:.2},\"disturbed_frames\":{:.4},{}\"gain_db\":{:.3},\"lag_samples\":{},\"pass\":{pass}}}",
            n.overall_db,
            n.esr_preemphasis_db,
            n.nmr.total_db,
            n.nmr.worst_db,
            n.nmr.disturbed,
            grade.map_or(String::new(), |(odg, di)| format!(
                "\"peaq_odg\":{odg:.3},\"peaq_distortion_index\":{di:.3},"
            )),
            n.gain_db,
            n.lag
        );
        return Ok(pass);
    }
    println!(
        "reference: {}\ncandidate: {}\n",
        args.reference.display(),
        args.candidate.display()
    );
    println!(
        "Aligned by {:+.2} dB and {} sample(s), then subtracted.\n",
        n.gain_db, n.lag
    );
    println!("Error-to-signal ratio (what is left, relative to the reference):\n");
    println!("| over | |");
    println!("|------|---:|");
    println!("| the whole file | {:.1} dB |", n.overall_db);
    println!(
        "| the whole file, treble pre-emphasised | {:.1} dB |",
        n.esr_preemphasis_db
    );
    println!(
        "| half of the 10 ms frames, at most | {:.1} dB |",
        n.frames_db[0]
    );
    println!(
        "| 95 % of the 10 ms frames, at most | {:.1} dB |",
        n.frames_db[1]
    );
    println!(
        "| the worst 10 ms frame (at {:.2} s) | {:.1} dB |",
        n.worst_at, n.frames_db[2]
    );
    println!("\n| octave around | |");
    println!("|--------------:|---:|");
    for (hz, level) in &n.bands_db {
        match level {
            Some(level) => println!("| {hz:.0} Hz | {level:.1} dB |"),
            None => println!("| {hz:.0} Hz | nothing there |"),
        }
    }
    println!(
        "\nNoise-to-mask ratio (what is left, against what the reference masks in the ear):\n"
    );
    println!("| | |");
    println!("|---|---:|");
    println!(
        "| average over frames and bands | {:.1} dB |",
        n.nmr.total_db
    );
    println!(
        "| worst band of the worst frame (at {:.2} s) | {:.1} dB |",
        n.nmr.worst_at, n.nmr.worst_db
    );
    println!(
        "| frames where it rises above the mask | {:.1} % |",
        100.0 * n.nmr.disturbed
    );
    println!(
        "\nVerdict: {}",
        if n.nmr.disturbed == 0.0 {
            "the difference stays under the masking threshold in every band of every frame."
        } else {
            "the difference rises above the masking threshold."
        }
    );
    if let Some((odg, di)) = grade {
        println!("\nPEAQ (ITU-R BS.1387 basic, by GstPEAQ): objective difference grade {odg:.2}, distortion index {di:.2}");
        println!("(0 imperceptible, -1 perceptible but not annoying, -2 slightly annoying, -3 annoying, -4 very annoying)");
    }
    if let Some(path) = &args.output {
        println!("residual written to {}", path.display());
    }
    Ok(pass)
}
