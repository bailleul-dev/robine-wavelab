use super::write_json;
use crate::metrics::quality;
use crate::wav;
use clap::Args as ClapArgs;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    /// Candidate render.
    candidate: PathBuf,
    /// Reference render.
    reference: PathBuf,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Maximum lag searched when aligning, in milliseconds (0 disables).
    #[arg(long, default_value_t = 50.0)]
    max_lag_ms: f32,
    /// Least-squares gain-match the candidate before measuring.
    #[arg(long)]
    match_gain: bool,
    /// Tolerance used for the exceed counters.
    #[arg(long, default_value_t = 1e-3)]
    tolerance: f32,
    #[arg(long)]
    json: Option<PathBuf>,
}

pub fn run(args: Args) -> Result<bool, String> {
    let cand = wav::read_mono(&args.candidate, args.channel)?;
    let reference = wav::read_mono(&args.reference, args.channel)?;
    if cand.sample_rate != reference.sample_rate {
        return Err(format!(
            "sample rates differ: {} vs {}",
            cand.sample_rate, reference.sample_rate
        ));
    }
    let max_lag = (args.max_lag_ms * cand.sample_rate as f32 / 1000.0) as usize;
    let q = quality(
        &cand.samples,
        &reference.samples,
        cand.sample_rate as f64,
        max_lag,
        args.match_gain,
        args.tolerance,
    );
    println!("candidate : {}", args.candidate.display());
    println!("reference : {}", args.reference.display());
    println!(
        "lag       : {} samples ({:.2} ms)",
        q.lag_samples,
        q.lag_samples as f64 * 1000.0 / cand.sample_rate as f64
    );
    println!(
        "gain      : {:+.2} dB (applied: {})",
        q.gain_db, args.match_gain
    );
    println!(
        "rms       : cand {:.2} dBFS  ref {:.2} dBFS",
        q.rms_candidate_db, q.rms_reference_db
    );
    println!(
        "peak      : cand {:.2} dBFS  ref {:.2} dBFS",
        q.peak_candidate_db, q.peak_reference_db
    );
    println!(
        "max |err| : {:.3e} @ {}",
        q.delta.max_abs, q.delta.max_abs_index
    );
    println!("ESR       : {:.3e}", q.delta.esr);
    println!("SNR       : {:.1} dB", q.delta.snr_db);
    println!("corr      : {:.6}", q.delta.correlation);
    println!(
        "LSD       : {:.2} dB (20 Hz-20 kHz)  {:.2} dB (200 Hz-6 kHz)",
        q.log_spectral_distance_db, q.band_spectral_distance_db
    );
    if let Some(path) = &args.json {
        write_json(path, &q)?;
    }
    Ok(true)
}
