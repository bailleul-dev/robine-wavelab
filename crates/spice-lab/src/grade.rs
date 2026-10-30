//! `spice-lab grade`: how far each circuit is known to be right, a medal
//! from tin to platinum, or none (unknown, broken).
//!
//! The grades live in `grades.json` next to the netlists, as the evidence
//! they were rated from and the rating itself (`rating.rs`). `measure`
//! collects what a machine can: the solver's health, the distance from
//! ngspice, the controls moved to their ends. `device` records a comparison
//! with a real device. `rate` rates the evidence again, or checks that the
//! stored grades are what the evidence gives. `table` writes them for the
//! documentation.

use crate::commands::{build, delta, netlists, positions, run_ours, CmdResult};
use crate::ngspice;
use crate::rating::{self, Controls, Device, Evidence, Grade, Ngspice, Solver, RUBRIC};
use clap::{Args, Subcommand};
use robine_spice::Circuit;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use wavelab::signal;

const RATE: f64 = 48_000.0;
const HIGH_RATE: f64 = 192_000.0;
/// Where along its travel a control is tried.
const POSITIONS: [f64; 3] = [0.0, 0.5, 1.0];
/// A control is verified when ngspice agrees at least this well at its
/// ends and middle (the point where the ngspice axis starts to score).
const CONTROL_DB: f64 = -20.0;

#[derive(Args)]
pub struct GradeArgs {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Run circuits and compare them with ngspice; record the evidence and
    /// rate it. Needs ngspice on PATH, unless `--no-ngspice`.
    Measure(MeasureArgs),
    /// Record a comparison with a real device, from `wavelab null --json`.
    Device(DeviceArgs),
    /// Rate the stored evidence again; with `--check`, only verify that the
    /// stored grades are what it gives.
    Rate(RateArgs),
    /// The grades as a Markdown table, optionally into a README.
    Table(TableArgs),
}

#[derive(Args)]
struct Where {
    /// Directory of netlists (searched one level deep).
    #[arg(long, default_value = "circuits")]
    circuits: PathBuf,
    /// The grades file; `grades.json` in the circuits directory by default.
    #[arg(long)]
    grades: Option<PathBuf>,
}

impl Where {
    fn file(&self) -> PathBuf {
        self.grades
            .clone()
            .unwrap_or_else(|| self.circuits.join("grades.json"))
    }
}

#[derive(Args)]
struct MeasureArgs {
    #[command(flatten)]
    at: Where,
    /// Only these circuits, by name as the table prints them
    /// (`big-muff-pi`, `basics/diode-rectifier`). Repeatable.
    #[arg(long)]
    only: Vec<String>,
    /// Peak input voltage.
    #[arg(long, default_value_t = 0.5)]
    peak: f64,
    /// Skip ngspice: only the solver is measured, and the grade stays
    /// unknown where it was.
    #[arg(long)]
    no_ngspice: bool,
    /// Do not move the controls.
    #[arg(long)]
    no_controls: bool,
    /// Length of the comparison with ngspice, in seconds.
    #[arg(long, default_value_t = 0.02)]
    seconds: f64,
    /// Tightest ngspice `reltol` to try first.
    #[arg(long, default_value_t = 1e-5)]
    ngspice_reltol: f64,
    #[arg(long, default_value = "target/grade")]
    work_dir: PathBuf,
}

#[derive(Args)]
struct DeviceArgs {
    #[command(flatten)]
    at: Where,
    /// The netlist that was compared.
    #[arg(short, long)]
    circuit: PathBuf,
    /// Which capture of the device, as a reader can find it again (a file
    /// name and where it comes from).
    #[arg(long)]
    capture: String,
    /// Control position the circuit was set to, and the pedal when it was
    /// captured, `name=value`. Repeatable; leave out the ones at their
    /// default.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// The output of `wavelab null --json --peaq` (without `--peaq` the
    /// PEAQ grade is left out), a file or `-` for standard input.
    #[arg(long)]
    null: PathBuf,
}

#[derive(Args)]
struct RateArgs {
    #[command(flatten)]
    at: Where,
    /// Change nothing; fail if a stored grade is not the one its evidence
    /// gives.
    #[arg(long)]
    check: bool,
}

#[derive(Args)]
struct TableArgs {
    #[command(flatten)]
    at: Where,
    /// Replace what is between `<!-- grades:begin -->` and
    /// `<!-- grades:end -->` in this file.
    #[arg(long)]
    readme: Option<PathBuf>,
}

pub fn grade(args: GradeArgs) -> CmdResult {
    match args.action {
        Action::Measure(a) => measure(a),
        Action::Device(a) => device(a),
        Action::Rate(a) => rate(a),
        Action::Table(a) => table(a),
    }
}

// ------------------------------------------------------------------ the file

#[derive(Serialize, Deserialize)]
struct Entry {
    grade: Grade,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    reason: String,
    evidence: Evidence,
}

#[derive(Serialize, Deserialize)]
struct Store {
    rubric: u32,
    circuits: BTreeMap<String, Entry>,
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

impl Entry {
    fn rated(evidence: Evidence) -> Entry {
        let r = rating::rate(&evidence);
        Entry {
            grade: r.grade,
            score: r.score.map(round1),
            reason: r.reason,
            evidence,
        }
    }

    /// Whether what is stored is what the evidence gives.
    fn consistent(&self) -> bool {
        let r = rating::rate(&self.evidence);
        r.grade == self.grade && r.score.map(round1) == self.score && r.reason == self.reason
    }

    /// Whether the netlist changed since any of the evidence was taken.
    fn stale(&self, hash: &str) -> bool {
        let differs = |h: &Option<String>| h.as_deref().is_some_and(|h| h != hash);
        differs(&self.evidence.netlist_hash)
            || self
                .evidence
                .devices
                .iter()
                .any(|d| differs(&d.netlist_hash))
    }
}

impl Store {
    fn read(path: &Path) -> Result<Store, String> {
        if !path.exists() {
            return Ok(Store {
                rubric: RUBRIC,
                circuits: BTreeMap::new(),
            });
        }
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn write(&self, path: &Path) -> Result<(), String> {
        let mut text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        text.push('\n');
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ------------------------------------------------------------ names and dates

/// A circuit as the table names it: its path under the directory, without
/// the extension.
fn name_of(directory: &Path, path: &Path) -> String {
    let relative = match (directory.canonicalize(), path.canonicalize()) {
        (Ok(d), Ok(p)) => p.strip_prefix(&d).map(Path::to_path_buf).ok(),
        _ => None,
    }
    .or_else(|| path.strip_prefix(directory).ok().map(Path::to_path_buf))
    .unwrap_or_else(|| path.file_name().map(PathBuf::from).unwrap_or_default());
    relative
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// FNV-1a over the bytes of the netlist: not a secret, a change detector
/// that gives the same answer on every machine and every toolchain.
///
/// The two comment lines that carry the grade are not part of what is
/// hashed: writing a grade does not change the circuit it is about.
fn hash_file(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hash = without_grade(&text)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    Ok(format!("{hash:016x}"))
}

const GRADE_LINE: &str = "* grade: ";
const WHY_LINE: &str = "* grade-why: ";
const FACTS_LINE: &str = "* grade-facts: ";

/// The netlist without the lines that carry its grade.
fn without_grade(text: &str) -> String {
    text.split_inclusive('\n')
        .filter(|l| ![GRADE_LINE, WHY_LINE, FACTS_LINE].iter().any(|p| l.starts_with(p)))
        .collect()
}

/// The netlist with its grade written after the title line.
fn with_grade(text: &str, entry: Option<&Entry>) -> String {
    let (grade, why, facts) = entry.map_or(
        (Grade::Unknown, "not measured", "measured=no".to_string()),
        |e| (e.grade, e.reason.as_str(), rating::facts(&e.evidence, e.score)),
    );
    let plain = without_grade(text);
    let (title, rest) = plain.split_once('\n').unwrap_or((plain.as_str(), ""));
    format!(
        "{title}\n{GRADE_LINE}{}\n{WHY_LINE}{why}\n{FACTS_LINE}{facts}\n{rest}",
        grade.label()
    )
}

/// Write the grade into a netlist; whether the file changed.
fn write_grade(path: &Path, entry: Option<&Entry>) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let new = with_grade(&text, entry);
    if new == text {
        return Ok(false);
    }
    std::fs::write(path, new).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(true)
}

/// Whether a netlist carries the grade its entry gives.
fn carries_grade(path: &Path, entry: Option<&Entry>) -> bool {
    std::fs::read_to_string(path).is_ok_and(|t| with_grade(&t, entry) == t)
}

/// Today in UTC as `YYYY-MM-DD`.
fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Days since 1970-01-01 to a civil date (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

// ------------------------------------------------------------------- measure

struct Run {
    ns_per_sample: f64,
    finite: bool,
    unconverged: u64,
}

/// The circuit on a test signal at the given control positions.
fn run(circuit: &Circuit, positions: &[f64], seconds: f64, peak: f64) -> Result<Run, String> {
    let mut processor = build(circuit, RATE, positions)?;
    let input: Vec<f32> = signal::sample(RATE, seconds, peak)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let mut output = vec![0.0f32; input.len()];
    let started = Instant::now();
    processor.process(&input, &mut output);
    Ok(Run {
        ns_per_sample: started.elapsed().as_secs_f64() * 1e9 / input.len() as f64,
        finite: output.iter().all(|v| v.is_finite()),
        unconverged: processor.solver_stats().failures,
    })
}

/// Distance from ngspice in dB, or `None` where ngspice did not finish or
/// the output has no AC part to compare.
fn versus_ngspice(
    circuit: &Circuit,
    positions: &[f64],
    rate: f64,
    args: &MeasureArgs,
    dir: &Path,
) -> Option<f64> {
    let input: Vec<f64> = signal::sample(rate, args.seconds, args.peak)
        .into_iter()
        .map(|v| f64::from(v as f32))
        .collect();
    let reference =
        ngspice::run(circuit, positions, &input, rate, dir, args.ngspice_reltol).ok()?;
    let mut processor = build(circuit, rate, positions).ok()?;
    let d = delta(&run_ours(&mut processor, &input), &reference.output);
    (d.signal_rms >= 1e-9).then(|| round1(d.relative_db))
}

fn unbuildable(evidence: &mut Evidence, error: String) {
    evidence.solver = Some(Solver {
        builds: false,
        finite: false,
        unconverged: 0,
        ns_per_sample: None,
        error: Some(error),
    });
}

/// What a machine can find out about a circuit.
fn evidence_of(path: &Path, name: &str, args: &MeasureArgs) -> Result<Evidence, String> {
    let mut evidence = Evidence {
        measured: Some(today()),
        netlist_hash: Some(hash_file(path)?),
        ..Evidence::default()
    };
    let circuit = match Circuit::read(path) {
        Ok(c) => c,
        Err(e) => {
            unbuildable(&mut evidence, e.to_string());
            return Ok(evidence);
        }
    };
    let rest: Vec<f64> = circuit.controls().iter().map(|c| c.value).collect();
    match run(&circuit, &rest, 1.0, args.peak) {
        Ok(r) => {
            evidence.solver = Some(Solver {
                builds: true,
                finite: r.finite,
                unconverged: r.unconverged,
                ns_per_sample: Some(r.ns_per_sample.round()),
                error: None,
            })
        }
        Err(e) => {
            unbuildable(&mut evidence, e);
            return Ok(evidence);
        }
    }
    // A deck that leaves part of the circuit out (a swept delay line: SPICE's
    // has one length) is another circuit, and ngspice says nothing of this
    // one: the axis is left unmeasured rather than counted as a failure.
    if args.no_ngspice || circuit.to_spice(&rest).contains("is not written") {
        return Ok(evidence);
    }
    let dir = args.work_dir.join(name);
    evidence.ngspice = Some(Ngspice {
        db_48k: versus_ngspice(&circuit, &rest, RATE, args, &dir.join("48k")),
        db_192k: versus_ngspice(&circuit, &rest, HIGH_RATE, args, &dir.join("192k")),
    });
    if !args.no_controls {
        // Each control is verified, failed, or unmeasured where ngspice could
        // not run it: only the first two count.
        let checked: Vec<bool> = (0..circuit.controls().len())
            .filter_map(|control| {
                let mut unmeasured = false;
                for &at in &POSITIONS {
                    let mut positions = rest.clone();
                    positions[control] = at;
                    let healthy = run(&circuit, &positions, args.seconds, args.peak)
                        .is_ok_and(|r| r.finite && r.unconverged == 0);
                    if !healthy {
                        return Some(false);
                    }
                    match versus_ngspice(
                        &circuit,
                        &positions,
                        RATE,
                        args,
                        &dir.join(format!("control{control}-{at}")),
                    ) {
                        Some(db) if db > CONTROL_DB => return Some(false),
                        Some(_) => {}
                        None => unmeasured = true,
                    }
                }
                (!unmeasured).then_some(true)
            })
            .collect();
        if !checked.is_empty() {
            evidence.controls = Some(Controls {
                verified: checked.iter().filter(|&&v| v).count(),
                total: checked.len(),
            });
        }
    }
    Ok(evidence)
}

fn wanted(name: &str, only: &[String]) -> bool {
    only.is_empty() || only.iter().any(|o| o == name)
}

fn measure(args: MeasureArgs) -> CmdResult {
    let file = args.at.file();
    let mut store = Store::read(&file)?;
    let all = netlists(&args.at.circuits)?;
    let mut matched = 0;
    for path in all {
        let name = name_of(&args.at.circuits, &path);
        if !wanted(&name, &args.only) {
            continue;
        }
        matched += 1;
        let mut evidence = evidence_of(&path, &name, &args)?;
        // A run without ngspice keeps what ngspice said before, and what the
        // real device said.
        let before = store.circuits.remove(&name).map(|e| e.evidence);
        if let Some(before) = before {
            evidence.devices = before.devices;
            if args.no_ngspice {
                evidence.ngspice = before.ngspice;
                evidence.controls = before.controls;
                evidence.source = before.source;
            }
        }
        let entry = Entry::rated(evidence);
        println!("{name}: {} ({})", entry.grade.label(), entry.reason);
        write_grade(&path, Some(&entry))?;
        store.circuits.insert(name, entry);
        store.rubric = RUBRIC;
        store.write(&file)?;
    }
    if matched == 0 {
        return Err(format!(
            "no circuit of {} matches {:?}",
            args.at.circuits.display(),
            args.only
        ));
    }
    Ok(true)
}

// -------------------------------------------------------------------- device

fn number(json: &serde_json::Value, key: &str) -> Result<f64, String> {
    json.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("the `wavelab null --json` output has no number `{key}`"))
}

fn device(args: DeviceArgs) -> CmdResult {
    let text = if args.null == Path::new("-") {
        std::io::read_to_string(std::io::stdin()).map_err(|e| e.to_string())?
    } else {
        std::fs::read_to_string(&args.null).map_err(|e| format!("{}: {e}", args.null.display()))?
    };
    let json: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| format!("not the JSON of `wavelab null --json`: {e}"))?;
    let circuit =
        Circuit::read(&args.circuit).map_err(|e| format!("{}: {e}", args.circuit.display()))?;
    let values = positions(&circuit, &args.sets)?;
    // The control names as the circuit spells them, only the ones set.
    let mut setting = BTreeMap::new();
    for set in &args.sets {
        let (given, _) = set.split_once('=').unwrap_or((set, ""));
        let index = circuit
            .controls()
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(given))
            .ok_or_else(|| format!("no control `{given}`"))?;
        setting.insert(circuit.controls()[index].name.clone(), values[index]);
    }
    let name = name_of(&args.at.circuits, &args.circuit);
    let file = args.at.file();
    let mut store = Store::read(&file)?;
    let mut evidence = store
        .circuits
        .remove(&name)
        .map(|e| e.evidence)
        .filter(|e| e.solver.is_some())
        .ok_or_else(|| {
            format!("{name} has not been measured: run `grade measure --only {name}`")
        })?;
    let record = Device {
        capture: args.capture,
        setting,
        esr_preemphasis_db: number(&json, "esr_preemphasis_db")?,
        nmr_db: number(&json, "nmr_db")?,
        disturbed_frames: number(&json, "disturbed_frames")?,
        peaq_odg: json.get("peaq_odg").and_then(serde_json::Value::as_f64),
        measured: Some(today()),
        netlist_hash: Some(hash_file(&args.circuit)?),
    };
    record_device(&mut evidence.devices, record);
    let entry = Entry::rated(evidence);
    println!("{name}: {} ({})", entry.grade.label(), entry.reason);
    write_grade(&args.circuit, Some(&entry))?;
    store.circuits.insert(name, entry);
    store.rubric = RUBRIC;
    store.write(&file)?;
    Ok(true)
}

/// A capture is one device at one setting: measuring it again replaces what
/// it said before, also when the circuit is now set otherwise for it (a
/// position read again through another taper), so that the old comparison
/// does not stay counted beside the new one.
fn record_device(devices: &mut Vec<Device>, record: Device) {
    devices.retain(|d| d.capture != record.capture);
    devices.push(record);
}

// ---------------------------------------------------------------------- rate

fn rate(args: RateArgs) -> CmdResult {
    let file = args.at.file();
    let mut store = Store::read(&file)?;
    let mut clean = true;
    if store.rubric != RUBRIC {
        println!(
            "rubric: the grades were rated under version {}, this is version {RUBRIC}",
            store.rubric
        );
        clean = false;
    }
    let present: Vec<String> = netlists(&args.at.circuits)?
        .iter()
        .map(|p| name_of(&args.at.circuits, p))
        .collect();
    for (name, entry) in &mut store.circuits {
        if !entry.consistent() {
            let was = entry.grade;
            if !args.check {
                *entry = Entry::rated(entry.evidence.clone());
            }
            println!(
                "{name}: stored {}, evidence gives {}",
                was.label(),
                rating::rate(&entry.evidence).grade.label()
            );
            clean = false;
        }
        if !present.contains(name) {
            println!("{name}: no such netlist any more");
            continue;
        }
        let path = args.at.circuits.join(format!("{name}.cir"));
        if hash_file(&path).is_ok_and(|h| entry.stale(&h)) {
            println!("{name}: the netlist changed since it was measured");
        }
    }
    for name in &present {
        let path = args.at.circuits.join(format!("{name}.cir"));
        let entry = store.circuits.get(name);
        if entry.is_none() {
            println!("{name}: unknown (never measured)");
        }
        if args.check {
            if !carries_grade(&path, entry) {
                println!("{name}: the netlist does not carry its grade");
                clean = false;
            }
        } else {
            write_grade(&path, entry)?;
        }
    }
    if args.check {
        return Ok(clean);
    }
    store.rubric = RUBRIC;
    store.write(&file)?;
    Ok(true)
}

// --------------------------------------------------------------------- table

fn table(args: TableArgs) -> CmdResult {
    let store = Store::read(&args.at.file())?;
    let mut rows =
        String::from("| circuit | grade | score | why |\n|---------|:-----:|------:|-----|\n");
    let mut counts: BTreeMap<Grade, usize> = BTreeMap::new();
    let mut total = 0;
    for path in netlists(&args.at.circuits)? {
        let name = name_of(&args.at.circuits, &path);
        let entry = store.circuits.get(&name);
        let grade = entry.map_or(Grade::Unknown, |e| e.grade);
        let stale = entry
            .zip(hash_file(&path).ok())
            .is_some_and(|(e, h)| e.stale(&h));
        let score = entry
            .and_then(|e| e.score)
            .map_or(String::new(), |s| format!("{s:.0}"));
        let why = entry.map_or("not measured", |e| e.reason.as_str());
        rows.push_str(&format!(
            "| {name} | {}{} | {score} | {why} |\n",
            grade.label(),
            if stale { " (stale)" } else { "" },
        ));
        *counts.entry(grade).or_default() += 1;
        total += 1;
    }
    let summary: Vec<String> = counts
        .iter()
        .rev()
        .map(|(g, n)| format!("{n} {}", g.label()))
        .collect();
    let text = format!("{total} circuits: {}.\n\n{rows}", summary.join(", "));
    match args.readme {
        None => print!("{text}"),
        Some(readme) => {
            let old = std::fs::read_to_string(&readme)
                .map_err(|e| format!("{}: {e}", readme.display()))?;
            let (begin, end) = ("<!-- grades:begin -->", "<!-- grades:end -->");
            let (Some(a), Some(b)) = (old.find(begin), old.find(end)) else {
                return Err(format!(
                    "{} has no {begin} … {end} to fill",
                    readme.display()
                ));
            };
            if b < a {
                return Err(format!("{}: {end} comes before {begin}", readme.display()));
            }
            let new = format!("{}{begin}\n{text}{}", &old[..a], &old[b..]);
            std::fs::write(&readme, new).map_err(|e| format!("{}: {e}", readme.display()))?;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_capture_measured_again_replaces_its_comparison_at_any_setting() {
        let device = |capture: &str, gain: f64, esr: f64| Device {
            capture: capture.into(),
            setting: BTreeMap::from([("Gain".to_string(), gain)]),
            esr_preemphasis_db: esr,
            nmr_db: 0.0,
            disturbed_frames: 0.5,
            peaq_odg: None,
            measured: None,
            netlist_hash: None,
        };
        let mut devices = vec![device("a.nam", 0.247, -5.0), device("b.nam", 0.5, -6.0)];
        record_device(&mut devices, device("a.nam", 0.5, -7.0));
        assert_eq!(devices.len(), 2);
        let a = devices.iter().find(|d| d.capture == "a.nam").unwrap();
        assert_eq!((a.setting["Gain"], a.esr_preemphasis_db), (0.5, -7.0));
    }

    #[test]
    fn names_are_paths_without_extension() {
        let dir = Path::new("circuits");
        assert_eq!(
            name_of(dir, Path::new("circuits/big-muff-pi.cir")),
            "big-muff-pi"
        );
        assert_eq!(
            name_of(dir, Path::new("circuits/basics/diode-rectifier.cir")),
            "basics/diode-rectifier"
        );
    }

    #[test]
    fn the_hash_is_stable() {
        // FNV-1a 64 of "a", from the reference implementation.
        let dir = std::env::temp_dir().join("spice-lab-grade-hash");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.cir");
        std::fs::write(&file, "a").unwrap();
        assert_eq!(hash_file(&file).unwrap(), "af63dc4c8601ec8c");
    }

    #[test]
    fn a_grade_is_written_after_the_title_and_replaced_not_stacked() {
        let netlist = "Fuzz Face\n* Converted by someone\nR1 a b 1\n";
        let mut entry = Entry::rated(Evidence::default());
        let once = with_grade(netlist, Some(&entry));
        assert_eq!(
            once,
            "Fuzz Face\n* grade: unknown\n* grade-why: not measured\n* grade-facts: measured=no\n* Converted by someone\nR1 a b 1\n"
        );
        assert_eq!(with_grade(&once, Some(&entry)), once);
        entry.grade = Grade::Silver;
        entry.reason = "because".into();
        let twice = with_grade(&once, Some(&entry));
        assert!(twice.contains("* grade: silver\n* grade-why: because\n"));
        assert_eq!(twice.matches("* grade").count(), 3);
        assert_eq!(without_grade(&twice), netlist);
    }

    #[test]
    fn writing_a_grade_does_not_change_the_hash() {
        let dir = std::env::temp_dir().join("spice-lab-grade-header");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("c.cir");
        std::fs::write(&file, "Title\nR1 a b 1\n").unwrap();
        let before = hash_file(&file).unwrap();
        assert!(write_grade(&file, None).unwrap());
        assert!(!write_grade(&file, None).unwrap());
        assert!(carries_grade(&file, None));
        assert_eq!(hash_file(&file).unwrap(), before);
    }

    #[test]
    fn dates_are_civil() {
        let date = today();
        assert_eq!(date.len(), 10);
        assert!(date.starts_with("20"), "{date}");
    }

    #[test]
    fn a_stored_grade_is_consistent_with_its_evidence_until_edited() {
        let evidence = Evidence {
            solver: Some(Solver {
                builds: true,
                finite: true,
                unconverged: 0,
                ns_per_sample: None,
                error: None,
            }),
            ngspice: Some(Ngspice {
                db_48k: Some(-45.0),
                db_192k: None,
            }),
            ..Evidence::default()
        };
        let mut entry = Entry::rated(evidence);
        assert!(entry.consistent());
        entry.grade = Grade::Gold;
        assert!(!entry.consistent());
    }

    #[test]
    fn a_changed_netlist_makes_evidence_stale() {
        let mut entry = Entry::rated(Evidence::default());
        entry.evidence.netlist_hash = Some("aaaa".into());
        assert!(!entry.stale("aaaa"));
        assert!(entry.stale("bbbb"));
        entry.evidence.netlist_hash = None;
        assert!(!entry.stale("bbbb"));
    }
}
