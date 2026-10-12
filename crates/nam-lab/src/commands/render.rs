use crate::runners::{Impl, LoadedFile, SizeArg};
use crate::wav;
use clap::Args as ClapArgs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(ClapArgs)]
pub struct Args {
    /// .nam model.
    #[arg(long, short)]
    model: PathBuf,
    /// Input WAV (mono, or the channel selected with --channel).
    #[arg(long, short)]
    input: PathBuf,
    /// Output WAV (32-bit float mono).
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "robine")]
    r#impl: Impl,
    /// Model size (largest in the file when omitted).
    #[arg(long, value_enum)]
    size: Option<SizeArg>,
    /// Processing block size.
    #[arg(long, default_value_t = 64)]
    block: usize,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// Input gain in dB applied before the model.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    input_db: f32,
    /// Output gain in dB applied after the model.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    output_db: f32,
    /// Render even if the WAV sample rate differs from the model's.
    #[arg(long)]
    force: bool,
    /// Only render the first N seconds of the input (0 = all).
    #[arg(long, default_value_t = 0.0)]
    seconds: f32,
}

pub fn run(args: Args) -> Result<bool, String> {
    let loaded = LoadedFile::read(&args.model)?;
    let size = loaded.pick(args.size.map(Into::into))?;
    let audio = wav::read_mono(&args.input, args.channel)?;
    let model_rate = loaded.file.variant(size).unwrap().sample_rate;
    if (audio.sample_rate as f32 - model_rate).abs() > 0.5 {
        let msg = format!(
            "{} is {} Hz, model expects {} Hz",
            args.input.display(),
            audio.sample_rate,
            model_rate
        );
        if args.force {
            eprintln!("warning: {msg}");
        } else {
            return Err(format!("{msg} (use --force to render anyway)"));
        }
    }
    let mut runner = loaded.build(args.r#impl, size, args.block)?;
    let in_gain = 10f32.powf(args.input_db / 20.0);
    let out_gain = 10f32.powf(args.output_db / 20.0);
    let take = if args.seconds > 0.0 {
        ((args.seconds * audio.sample_rate as f32) as usize).min(audio.samples.len())
    } else {
        audio.samples.len()
    };
    let input: Vec<f32> = audio.samples[..take].iter().map(|s| s * in_gain).collect();
    let mut output = vec![0.0f32; input.len()];
    let t0 = Instant::now();
    runner.process(&input, &mut output);
    let elapsed = t0.elapsed();
    for s in output.iter_mut() {
        *s *= out_gain;
    }
    wav::write_mono_f32(&args.output, &output, audio.sample_rate)?;
    let audio_s = input.len() as f64 / audio.sample_rate as f64;
    eprintln!(
        "{} | {} | {} samples in {:.1} ms ({:.1}x real time) -> {}",
        runner.description(),
        size,
        input.len(),
        elapsed.as_secs_f64() * 1e3,
        audio_s / elapsed.as_secs_f64(),
        args.output.display()
    );
    Ok(true)
}
