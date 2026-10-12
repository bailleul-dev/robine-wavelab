use crate::runners::{Impl, LoadedFile};
use clap::Args as ClapArgs;
use robine_nam::Size;
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    /// .nam files to inspect.
    #[arg(required = true)]
    models: Vec<PathBuf>,
    /// Also instantiate every implementation to confirm they accept the file.
    #[arg(long)]
    build: bool,
}

pub fn run(args: Args) -> Result<bool, String> {
    let mut all_ok = true;
    for path in &args.models {
        println!("== {}", path.display());
        let loaded = match LoadedFile::read(path) {
            Ok(l) => l,
            Err(e) => {
                println!("   unsupported: {e}");
                all_ok = false;
                continue;
            }
        };
        let f = &loaded.file;
        println!(
            "   architecture: {}  version: {}  sample_rate: {} Hz",
            f.architecture, f.version, f.sample_rate
        );
        let m = &f.metadata;
        println!(
            "   name: {}  by: {}  gear: {} {} ({})  tone: {}",
            m.display_name("-"),
            m.modeled_by.as_deref().unwrap_or("-"),
            m.gear_make.as_deref().unwrap_or("-"),
            m.gear_model.as_deref().unwrap_or("-"),
            m.gear_type.as_deref().unwrap_or("-"),
            m.tone_type.as_deref().unwrap_or("-"),
        );
        println!(
            "   levels: in {} dBu  out {} dBu  loudness {} dB  gain {}  validation ESR {}",
            opt(m.input_level_dbu),
            opt(m.output_level_dbu),
            opt(m.loudness_db),
            opt(m.gain),
            opt(m.validation_esr),
        );
        for v in &f.variants {
            println!(
                "   variant {:<5} channels {}  max_value {}  weights {}  name {}",
                v.size.to_string(),
                v.size.channels(),
                v.max_value,
                v.weights.len(),
                v.metadata.display_name("-")
            );
        }
        for s in &f.skipped {
            println!("   skipped: {s}");
        }
        if args.build {
            for size in f.sizes().collect::<Vec<Size>>() {
                for imp in [Impl::Robine, Impl::RefGeneric, Impl::RefFast, Impl::RefAuto] {
                    match loaded.build(imp, size, 64) {
                        Ok(r) => {
                            println!("   {:<5} {:<12} ok: {}", size, imp.label(), r.description())
                        }
                        Err(e) => {
                            println!("   {:<5} {:<12} FAILED: {e}", size, imp.label());
                            all_ok = false;
                        }
                    }
                }
            }
        }
    }
    Ok(all_ok)
}

fn opt(v: Option<f32>) -> String {
    v.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into())
}
