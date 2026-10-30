//! ngspice as the reference: the same flattened circuit, driven by the same
//! samples, solved with tight tolerances and a much finer time step.

use robine_spice::Circuit;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// Internal steps ngspice may take per input sample, at most.
const SUBSTEPS: f64 = 8.0;

/// Tolerance sets, tightest first: `(reltol, abstol, vntol)`. ngspice gives
/// up with "timestep too small" when asked for more than a circuit's
/// conditioning allows, so the runner walks down this list.
const TOLERANCES: [(f64, f64, f64); 4] = [
    (1e-7, 1e-15, 1e-10),
    (1e-6, 1e-14, 1e-9),
    (1e-5, 1e-13, 1e-8),
    (1e-4, 1e-12, 1e-7),
];

/// How long one ngspice run may take: `NGSPICE_TIMEOUT` seconds, 300 by
/// default.
fn timeout() -> Duration {
    let seconds = std::env::var("NGSPICE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    Duration::from_secs(seconds)
}

/// What ngspice returned and how hard it was pushed.
pub struct Reference {
    /// Output voltage at each input sample time.
    pub output: Vec<f64>,
    /// The `reltol` of the tolerance set that completed.
    pub reltol: f64,
}

/// The deck ngspice is given for `input` sampled at `rate`.
pub fn deck(
    circuit: &Circuit,
    positions: &[f64],
    input: &[f64],
    rate: f64,
    data: &Path,
    (reltol, abstol, vntol): (f64, f64, f64),
    method: &str,
) -> String {
    let source = circuit.input_source();
    let mut deck = String::new();
    for line in circuit.to_spice(positions).lines() {
        let mut tokens = line.split_whitespace();
        if tokens
            .next()
            .is_some_and(|t| t.eq_ignore_ascii_case(source))
        {
            let (plus, minus) = (tokens.next().unwrap(), tokens.next().unwrap());
            let _ = writeln!(deck, "{source} {plus} {minus} PWL(");
            for (k, v) in input.iter().enumerate() {
                let _ = writeln!(deck, "+ {:e} {v:e}", k as f64 / rate);
            }
            deck.push_str("+ )\n");
        } else {
            deck.push_str(line);
            deck.push('\n');
        }
    }
    let (out, reference) = circuit.output_nodes();
    // ngspice has no `v(0)`.
    let probe = match (out, reference) {
        (out, "0") => format!("v({out})"),
        ("0", reference) => format!("0-v({reference})"),
        (out, reference) => format!("v({out})-v({reference})"),
    };
    let step = 1.0 / rate;
    let _ = write!(
        deck,
        ".options reltol={reltol:e} abstol={abstol:e} vntol={vntol:e} method={method}\n\
         .control\n\
         set numdgt=15\n\
         set wr_singlescale\n\
         tran {:e} {:e} 0 {:e}\n\
         wrdata {} {probe}\n\
         .endc\n\
         .end\n",
        step,
        step * (input.len() - 1) as f64,
        step / SUBSTEPS,
        data.display(),
    );
    deck
}

/// Integration methods, the more accurate first. Gear damps what trap lets
/// ring, and completes some decks trap gives up on at every tolerance (a
/// pentode plate, nearly a current source, into a transformer's leakage).
const METHODS: [&str; 2] = ["trap", "gear"];

/// Run ngspice with the tightest tolerances it completes with, by trap if it
/// can.
pub fn run(
    circuit: &Circuit,
    positions: &[f64],
    input: &[f64],
    rate: f64,
    dir: &Path,
    tightest: f64,
) -> Result<Reference, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut last = format!("no ngspice tolerance set is as loose as reltol={tightest:e}");
    // A set that fails can take a long time to fail: `tightest` lets the
    // caller skip the ones known not to complete for a circuit.
    for method in METHODS {
        for tolerances in TOLERANCES.into_iter().filter(|t| t.0 >= tightest * 0.999) {
            match attempt(circuit, positions, input, rate, dir, tolerances, method) {
                Ok(output) => {
                    return Ok(Reference {
                        output,
                        reltol: tolerances.0,
                    })
                }
                // Looser tolerances rarely rescue a run that was merely slow.
                Err(e) if e.contains("did not finish") => return Err(e),
                Err(e) => last = e,
            }
        }
    }
    Err(last)
}

fn attempt(
    circuit: &Circuit,
    positions: &[f64],
    input: &[f64],
    rate: f64,
    dir: &Path,
    tolerances: (f64, f64, f64),
    method: &str,
) -> Result<Vec<f64>, String> {
    let deck_path = dir.join("deck.cir");
    let data_path = dir.join("out.data");
    let _ = std::fs::remove_file(&data_path);
    let text = deck(circuit, positions, input, rate, &data_path, tolerances, method);
    std::fs::write(&deck_path, text).map_err(|e| format!("{}: {e}", deck_path.display()))?;
    let log_path = dir.join("ngspice.log");
    let log =
        std::fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let mut child = Command::new("ngspice")
        .arg("-b")
        .arg(&deck_path)
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .spawn()
        .map_err(|e| format!("cannot run ngspice (is it installed and on PATH?): {e}"))?;
    let started = Instant::now();
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(_) => break,
            None if started.elapsed() > timeout() => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "ngspice did not finish in {} s (deck kept at {})",
                    timeout().as_secs(),
                    deck_path.display()
                ));
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let text = std::fs::read_to_string(&data_path).map_err(|_| {
        format!(
            "ngspice produced no data (deck {}, log {})",
            deck_path.display(),
            log_path.display()
        )
    })?;

    // ngspice writes every internal time point. The input's breakpoints
    // force one exactly at each sample time: keep those.
    let points: Vec<(f64, f64)> = text
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace().map(str::parse::<f64>);
            match (fields.next(), fields.next()) {
                (Some(Ok(t)), Some(Ok(v))) => Some((t, v)),
                _ => None,
            }
        })
        .collect();
    if points.is_empty() {
        return Err(format!(
            "ngspice wrote no samples (deck kept at {})",
            deck_path.display()
        ));
    }
    let step = 1.0 / rate;
    let mut out = Vec::with_capacity(input.len());
    let mut j = 0;
    for k in 0..input.len() {
        let target = k as f64 * step;
        while j + 1 < points.len()
            && (points[j + 1].0 - target).abs() <= (points[j].0 - target).abs()
        {
            j += 1;
        }
        if (points[j].0 - target).abs() > 1e-3 * step {
            return Err(format!(
                "ngspice has no point at t = {target:e} s (deck kept at {})",
                deck_path.display()
            ));
        }
        out.push(points[j].1);
    }
    Ok(out)
}
