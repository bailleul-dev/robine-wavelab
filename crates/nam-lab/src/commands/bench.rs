use super::{stimulus, write_json};
use crate::runners::{Impl, LoadedFile, SizeArg};
use clap::Args as ClapArgs;
use robine_nam::Size;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long, short, required = true)]
    model: Vec<PathBuf>,
    /// Implementations to measure.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "robine,ref-fast,ref-generic"
    )]
    r#impl: Vec<Impl>,
    #[arg(long, value_enum)]
    size: Vec<SizeArg>,
    #[arg(long, value_delimiter = ',', default_value = "16,32,64,128,256")]
    blocks: Vec<usize>,
    /// Seconds of audio per iteration.
    #[arg(long, default_value_t = 5.0)]
    seconds: f32,
    /// Timed iterations (one untimed warm-up is always run).
    #[arg(long, default_value_t = 5)]
    iters: usize,
    /// Input WAV instead of the synthetic DI.
    #[arg(long, short)]
    input: Option<PathBuf>,
    #[arg(long)]
    json: Option<PathBuf>,
    /// One compact line per configuration: best and median ns/sample only.
    #[arg(long)]
    brief: bool,
}

#[derive(Serialize, Clone)]
pub struct BenchResult {
    model: String,
    r#impl: String,
    size: String,
    block: usize,
    sample_rate: f32,
    blocks_timed: usize,
    ns_per_sample: f64,
    /// ns/sample of the fastest iteration (least disturbed by the OS).
    ns_per_sample_best: f64,
    /// ns/sample of the median iteration.
    ns_per_sample_median: f64,
    /// Audio seconds processed per CPU second.
    realtime_factor: f64,
    /// CPU share of one core needed at the model sample rate.
    cpu_percent: f64,
    block_us_p50: f64,
    block_us_p99: f64,
    block_us_p999: f64,
    block_us_max: f64,
    block_deadline_us: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
}

pub fn run(args: Args) -> Result<bool, String> {
    let mut results = Vec::new();
    println!(
        "{:<28} {:<12} {:<6} {:>5} {:>9} {:>8} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "model",
        "impl",
        "size",
        "block",
        "ns/sample",
        "RTF",
        "CPU%",
        "p50 us",
        "p99 us",
        "p99.9 us",
        "max us",
        "deadline"
    );
    for model_path in &args.model {
        let loaded = LoadedFile::read(model_path)?;
        let sizes: Vec<Size> = if args.size.is_empty() {
            loaded.file.sizes().collect()
        } else {
            args.size.iter().map(|&s| s.into()).collect()
        };
        let rate = loaded.file.sample_rate;
        let stim = stimulus(args.input.as_ref(), 0, args.seconds, rate)?;
        let model_name = model_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        for &size in &sizes {
            let size = loaded.pick(Some(size))?;
            for &block in &args.blocks {
                for &imp in &args.r#impl {
                    let mut runner = loaded.build(imp, size, block)?;
                    let mut out = vec![0.0f32; stim.samples.len()];
                    // Warm-up.
                    runner.process(&stim.samples, &mut out);
                    let mut block_times: Vec<f64> = Vec::new();
                    let mut iter_ns: Vec<f64> = Vec::new();
                    let mut total = 0.0f64;
                    for _ in 0..args.iters {
                        let mut iter_total = 0.0f64;
                        for (i, o) in stim.samples.chunks(block).zip(out.chunks_mut(block)) {
                            let t0 = Instant::now();
                            runner.process(i, o);
                            let dt = t0.elapsed().as_secs_f64();
                            block_times.push(dt * 1e6);
                            iter_total += dt;
                        }
                        total += iter_total;
                        iter_ns.push(iter_total * 1e9 / stim.samples.len() as f64);
                    }
                    block_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    iter_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let samples = stim.samples.len() * args.iters;
                    let audio_seconds = samples as f64 / rate as f64;
                    let r = BenchResult {
                        model: model_name.clone(),
                        r#impl: imp.label().into(),
                        size: size.to_string(),
                        block,
                        sample_rate: rate,
                        blocks_timed: block_times.len(),
                        ns_per_sample: total * 1e9 / samples as f64,
                        ns_per_sample_best: iter_ns[0],
                        ns_per_sample_median: percentile(&iter_ns, 0.5),
                        realtime_factor: audio_seconds / total,
                        cpu_percent: 100.0 * total / audio_seconds,
                        block_us_p50: percentile(&block_times, 0.5),
                        block_us_p99: percentile(&block_times, 0.99),
                        block_us_p999: percentile(&block_times, 0.999),
                        block_us_max: *block_times.last().unwrap(),
                        block_deadline_us: block as f64 * 1e6 / rate as f64,
                    };
                    if args.brief {
                        println!(
                            "{:<12} {:<6} block {:>4}  best {:>7.1} ns/sample  median {:>7.1}  p50 {:>7.1} us",
                            r.r#impl, r.size, r.block, r.ns_per_sample_best, r.ns_per_sample_median, r.block_us_p50
                        );
                        results.push(r);
                        continue;
                    }
                    println!(
                        "{:<28} {:<12} {:<6} {:>5} {:>9.1} {:>9.1} {:>9.1} {:>8.1} {:>7.2} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                        truncate(&r.model, 28),
                        r.r#impl,
                        r.size,
                        r.block,
                        r.ns_per_sample,
                        r.ns_per_sample_best,
                        r.ns_per_sample_median,
                        r.realtime_factor,
                        r.cpu_percent,
                        r.block_us_p50,
                        r.block_us_p99,
                        r.block_us_p999,
                        r.block_us_max,
                        r.block_deadline_us
                    );
                    results.push(r);
                }
            }
        }
    }
    if let Some(path) = &args.json {
        write_json(path, &results)?;
    }
    Ok(true)
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}
