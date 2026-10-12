use super::{stimulus, write_json};
use crate::metrics::{delta, Delta};
use crate::runners::{Impl, LoadedFile, SizeArg};
use clap::Args as ClapArgs;
use robine_nam::Size;
use serde::Serialize;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    /// .nam model(s).
    #[arg(long, short, required = true)]
    model: Vec<PathBuf>,
    /// Input WAV; a synthetic DI is used when omitted.
    #[arg(long, short)]
    input: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Seconds of audio to use (0 = whole file; synthetic default 5 s).
    #[arg(long, default_value_t = 0.0)]
    seconds: f32,
    /// Sizes to compare (all available when omitted).
    #[arg(long, value_enum)]
    size: Vec<SizeArg>,
    /// Block sizes to run (the delta must hold for every one).
    #[arg(long, value_delimiter = ',', default_value = "1,17,64,256")]
    blocks: Vec<usize>,
    /// Maximum tolerated absolute error against ref-generic.
    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,
    /// Write the full report as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    model: String,
    stimulus: String,
    samples: usize,
    sample_rate: f32,
    results: Vec<PairResult>,
}

#[derive(Serialize)]
struct PairResult {
    size: String,
    block: usize,
    candidate: String,
    reference: String,
    pass: bool,
    delta: Delta,
}

pub fn run(args: Args) -> Result<bool, String> {
    let mut all_pass = true;
    let mut reports = Vec::new();
    for model_path in &args.model {
        let loaded = LoadedFile::read(model_path)?;
        let sizes: Vec<Size> = if args.size.is_empty() {
            loaded.file.sizes().collect()
        } else {
            args.size.iter().map(|&s| s.into()).collect()
        };
        let model_rate = loaded.file.sample_rate;
        let stim = stimulus(args.input.as_ref(), args.channel, args.seconds, model_rate)?;
        println!(
            "== {}  |  {}  |  {} samples @ {} Hz",
            model_path.display(),
            stim.name,
            stim.samples.len(),
            stim.sample_rate
        );
        println!(
            "{:<6} {:>5} {:<12} {:<12} {:>10} {:>9} {:>10} {:>9} {:>8} {:>7}",
            "size",
            "block",
            "candidate",
            "reference",
            "max_abs",
            "@index",
            "rms_err",
            "ESR",
            "SNR dB",
            "pass"
        );
        let mut results = Vec::new();
        for &size in &sizes {
            let size = loaded.pick(Some(size))?;
            for &block in &args.blocks {
                let mut outputs = Vec::new();
                for imp in [Impl::RefGeneric, Impl::RefFast, Impl::Robine] {
                    let mut runner = loaded.build(imp, size, block)?;
                    let mut out = vec![0.0f32; stim.samples.len()];
                    runner.process(&stim.samples, &mut out);
                    outputs.push((imp, out));
                }
                let generic = &outputs[0].1;
                let fast = &outputs[1].1;
                let robine = &outputs[2].1;
                let pairs = [
                    (Impl::Robine, robine, Impl::RefGeneric, generic, true),
                    (Impl::Robine, robine, Impl::RefFast, fast, false),
                    (Impl::RefFast, fast, Impl::RefGeneric, generic, false),
                ];
                for (cand_imp, cand, ref_imp, reference, gating) in pairs {
                    let d = delta(cand, reference, args.tolerance);
                    let pass = d.max_abs <= args.tolerance;
                    if gating && !pass {
                        all_pass = false;
                    }
                    println!(
                        "{:<6} {:>5} {:<12} {:<12} {:>10.3e} {:>9} {:>10.3e} {:>9.2e} {:>8.1} {:>7}",
                        size.to_string(),
                        block,
                        cand_imp.label(),
                        ref_imp.label(),
                        d.max_abs,
                        d.max_abs_index,
                        d.rms_error,
                        d.esr,
                        d.snr_db,
                        if pass { "ok" } else if gating { "FAIL" } else { "warn" }
                    );
                    results.push(PairResult {
                        size: size.to_string(),
                        block,
                        candidate: cand_imp.label().into(),
                        reference: ref_imp.label().into(),
                        pass,
                        delta: d,
                    });
                }
            }
        }
        reports.push(Report {
            model: model_path.display().to_string(),
            stimulus: stim.name.clone(),
            samples: stim.samples.len(),
            sample_rate: stim.sample_rate,
            results,
        });
    }
    if let Some(path) = &args.json {
        write_json(path, &reports)?;
    }
    println!("{}", if all_pass { "PASS" } else { "FAIL" });
    Ok(all_pass)
}
