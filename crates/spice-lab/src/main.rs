mod alias;
mod commands;
mod dk;
mod grade;
mod model;
mod ngspice;
mod rating;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// Tools around robine-spice: schematic conversion, comparison against
/// ngspice, benchmarks. Always run from a release build. Measures on audio
/// files (probe, judge, null) are in `wavelab`.
#[derive(Parser)]
#[command(name = "spice-lab", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert a LiveSPICE schematic (.schx) to a netlist.
    Convert(commands::ConvertArgs),
    /// Describe a netlist: size, controls, warnings, operating point.
    Inspect(commands::InspectArgs),
    /// Render a WAV through a circuit.
    Render(commands::RenderArgs),
    /// Compare robine-spice against ngspice on the same circuit and signal.
    Compare(commands::CompareArgs),
    /// Throughput of a circuit and cost of moving a control.
    Bench(commands::BenchArgs),
    /// Run every circuit of a directory and compare each with ngspice.
    Survey(commands::SurveyArgs),
    /// Write golden fixtures (input and ngspice output) for `cargo test`.
    Golden(commands::GoldenArgs),
    /// How much aliasing a circuit makes, against a run at a higher rate.
    Alias(alias::AliasArgs),
    /// Grade the models from F to A+: measure, record, rate, tabulate.
    Grade(grade::GradeArgs),
    /// Export the reduced linear system, optionally with a traced run.
    Dk(dk::DkArgs),
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: spice-lab is a debug build; timings and real-time claims are meaningless"
        );
    }
    let result = match Cli::parse().command {
        Command::Convert(a) => commands::convert(a),
        Command::Inspect(a) => commands::inspect(a),
        Command::Render(a) => commands::render(a),
        Command::Compare(a) => commands::compare(a),
        Command::Bench(a) => commands::bench(a),
        Command::Survey(a) => commands::survey(a),
        Command::Golden(a) => commands::golden(a),
        Command::Alias(a) => alias::alias(a),
        Command::Grade(a) => grade::grade(a),
        Command::Dk(a) => dk::dk(a),
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
