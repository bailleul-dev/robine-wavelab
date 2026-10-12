use super::write_json;
use crate::runners::{Impl, LoadedFile};
use crate::signal;
use clap::Args as ClapArgs;
use serde::Serialize;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long, short)]
    model: PathBuf,
    /// Output directory for `<name>-<size>.{input,output}.f32` and `<name>.golden.json`.
    #[arg(long, short)]
    output: PathBuf,
    /// Fixture name (defaults to the model file stem).
    #[arg(long)]
    name: Option<String>,
    #[arg(long, default_value_t = 8192)]
    samples: usize,
    #[arg(long, default_value_t = 64)]
    block: usize,
    #[arg(long, value_enum, default_value = "ref-generic")]
    r#impl: Impl,
}

#[derive(Serialize)]
struct Manifest {
    model: String,
    generator: String,
    samples: usize,
    block: usize,
    sample_rate: f32,
    sizes: Vec<String>,
    format: &'static str,
}

pub fn run(args: Args) -> Result<bool, String> {
    let loaded = LoadedFile::read(&args.model)?;
    let name = args.name.clone().unwrap_or_else(|| {
        args.model
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    });
    std::fs::create_dir_all(&args.output).map_err(|e| e.to_string())?;
    let rate = loaded.file.sample_rate;
    let input = {
        let mut s = signal::synthetic_di(rate, args.samples as f32 / rate + 0.1, 0xA2, 0.6);
        s.truncate(args.samples);
        s
    };
    write_f32(&args.output.join(format!("{name}.input.f32")), &input)?;
    let mut sizes = Vec::new();
    for size in loaded.file.sizes().collect::<Vec<_>>() {
        let mut runner = loaded.build(args.r#impl, size, args.block)?;
        let mut out = vec![0.0f32; input.len()];
        runner.process(&input, &mut out);
        write_f32(&args.output.join(format!("{name}-{size}.output.f32")), &out)?;
        println!("{size}: {} via {}", out.len(), runner.description());
        sizes.push(size.to_string());
    }
    write_json(
        &args.output.join(format!("{name}.golden.json")),
        &Manifest {
            model: args
                .model
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            generator: args.r#impl.label().into(),
            samples: input.len(),
            block: args.block,
            sample_rate: rate,
            sizes,
            format: "little-endian f32",
        },
    )?;
    Ok(true)
}

fn write_f32(path: &std::path::Path, data: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}
