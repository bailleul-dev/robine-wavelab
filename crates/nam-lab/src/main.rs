mod commands;
mod runners;

use clap::{Parser, Subcommand};
use std::process::ExitCode;
pub(crate) use wavelab::{metrics, signal, wav};

/// Measurement tools for NAM A2 runtimes (robine-nam vs upstream
/// NeuralAmpModelerCore). Always run from a release build.
#[derive(Parser)]
#[command(name = "nam-lab", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Describe a .nam file and which implementations accept it.
    Inspect(commands::inspect::Args),
    /// Render a WAV through one implementation.
    Render(commands::render::Args),
    /// Sample-exact delta between robine-nam and the upstream reference.
    Compare(commands::compare::Args),
    /// Quality metrics between two renders (any source).
    CompareWav(commands::compare_wav::Args),
    /// Throughput and per-block latency of each implementation.
    Bench(commands::bench::Args),
    /// Generate golden input/output fixtures from the upstream reference.
    Golden(commands::golden::Args),
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: nam-lab is a debug build; timings and real-time claims are meaningless"
        );
    }
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Inspect(a) => commands::inspect::run(a),
        Command::Render(a) => commands::render::run(a),
        Command::Compare(a) => commands::compare::run(a),
        Command::CompareWav(a) => commands::compare_wav::run(a),
        Command::Bench(a) => commands::bench::run(a),
        Command::Golden(a) => commands::golden::run(a),
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
