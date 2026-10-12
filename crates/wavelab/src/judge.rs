//! Judging whether two devices behave alike, from what each does to the
//! same probe signal.
//!
//! The probe is made of steady tones at known frequencies and levels, so
//! every measurement is a correlation against a known sinusoid over a whole
//! number of periods: no spectrum estimation, no windowing artefacts. The
//! devices can be anything that turns a WAV into a WAV (a circuit, a NAM
//! capture, a real pedal reamped).

use std::f64::consts::TAU;

/// Peak of the loudest tones, relative to full scale.
const TOP: f64 = 0.9;
/// Every measurement window, in seconds. Tone frequencies are multiples of
/// 4 Hz so that the window always holds a whole number of periods.
const WINDOW: f64 = 0.25;
/// Silence before each tone, so that it starts from rest.
const GAP: f64 = 0.4;

const RESPONSE_HZ: [f64; 12] = [
    80.0, 120.0, 180.0, 272.0, 408.0, 612.0, 920.0, 1380.0, 2072.0, 3108.0, 4660.0, 6992.0,
];
const RESPONSE_DB: f64 = -36.0;
const TONE_HZ: [f64; 2] = [220.0, 880.0];
const TONE_DB: [f64; 5] = [-36.0, -24.0, -12.0, -6.0, 0.0];
const HARMONICS: usize = 8;
/// Quiet tone, loud burst, quiet tone again: seconds of each.
const RECOVERY: [f64; 3] = [0.6, 0.4, 1.0];
const RECOVERY_AFTER: [f64; 3] = [0.02, 0.1, 0.5];

#[derive(Clone, Copy)]
enum Kind {
    Response,
    Tone,
    Recovery,
}

struct Section {
    kind: Kind,
    /// Start of the sound, in seconds.
    start: f64,
    seconds: f64,
    hertz: f64,
    db: f64,
}

fn sections() -> Vec<Section> {
    let mut list = Vec::new();
    let mut t = 0.0;
    let mut push = |kind, seconds: f64, hertz, db, t: &mut f64| {
        *t += GAP;
        list.push(Section {
            kind,
            start: *t,
            seconds,
            hertz,
            db,
        });
        *t += seconds;
    };
    for hertz in RESPONSE_HZ {
        push(Kind::Response, 0.4, hertz, RESPONSE_DB, &mut t);
    }
    for hertz in TONE_HZ {
        for db in TONE_DB {
            push(Kind::Tone, 0.6, hertz, db, &mut t);
        }
    }
    push(
        Kind::Recovery,
        RECOVERY.iter().sum(),
        TONE_HZ[0],
        0.0,
        &mut t,
    );
    list
}

fn amplitude(db: f64) -> f64 {
    TOP * 10f64.powf(db / 20.0)
}

/// The probe signal at `rate`.
pub fn probe(rate: u32) -> Vec<f32> {
    let rate = f64::from(rate);
    let list = sections();
    let end = list.last().map_or(0.0, |s| s.start + s.seconds) + GAP;
    let mut out = vec![0.0f32; (end * rate).round() as usize];
    for s in &list {
        let first = (s.start * rate).round() as usize;
        let count = (s.seconds * rate).round() as usize;
        for k in 0..count {
            let t = k as f64 / rate;
            let level = match s.kind {
                Kind::Recovery if t < RECOVERY[0] || t >= RECOVERY[0] + RECOVERY[1] => {
                    amplitude(RESPONSE_DB)
                }
                _ => amplitude(s.db),
            };
            out[first + k] = (level * (TAU * s.hertz * t).sin()) as f32;
        }
    }
    out
}

/// Amplitude of the `hertz` component of `x` over one window starting at
/// `start` seconds.
fn component(x: &[f32], rate: f64, start: f64, hertz: f64) -> f64 {
    let first = (start * rate).round() as usize;
    let count = (WINDOW * rate).round() as usize;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, &v) in x[first..first + count].iter().enumerate() {
        let phase = TAU * hertz * k as f64 / rate;
        re += f64::from(v) * phase.cos();
        im += f64::from(v) * phase.sin();
    }
    2.0 * (re * re + im * im).sqrt() / count as f64
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

fn db(ratio: f64) -> f64 {
    20.0 * ratio.max(1e-12).log10()
}

/// What one device did to the probe.
pub struct Measurement {
    /// Level of each small tone of the response sweep, dB.
    response: Vec<f64>,
    /// Per tone (frequency then level): level of the whole output, dB.
    level: Vec<f64>,
    /// Per tone: harmonics 2..=HARMONICS relative to the fundamental, dB.
    harmonics: Vec<Vec<f64>>,
    /// Per loudest tone of each frequency: the first 10 ms and the next
    /// 40 ms relative to the settled tone, dB.
    attack: Vec<[f64; 2]>,
    /// Gain of the quiet tone after the burst relative to before it, dB.
    recovery: Vec<f64>,
}

pub fn measure(x: &[f32], rate: u32) -> Result<Measurement, String> {
    let rate = f64::from(rate);
    let list = sections();
    let needed = ((list.last().unwrap().start + list.last().unwrap().seconds) * rate) as usize;
    if x.len() < needed {
        return Err(format!(
            "the file is {:.1} s long, the probe is {:.1} s: was it rendered from `spice-lab probe`?",
            x.len() as f64 / rate,
            needed as f64 / rate
        ));
    }
    let at = |seconds: f64| (seconds * rate).round() as usize;
    let mut m = Measurement {
        response: Vec::new(),
        level: Vec::new(),
        harmonics: Vec::new(),
        attack: Vec::new(),
        recovery: Vec::new(),
    };
    for s in &list {
        let settled = s.start + s.seconds - WINDOW;
        match s.kind {
            Kind::Response => m.response.push(db(component(x, rate, settled, s.hertz))),
            Kind::Tone => {
                let window = &x[at(settled)..at(settled + WINDOW)];
                let mean = window.iter().map(|&v| f64::from(v)).sum::<f64>() / window.len() as f64;
                let ac: Vec<f32> = window
                    .iter()
                    .map(|&v| (f64::from(v) - mean) as f32)
                    .collect();
                m.level.push(db(rms(&ac)));
                let fundamental = component(x, rate, settled, s.hertz);
                m.harmonics.push(
                    (2..=HARMONICS)
                        .map(|h| db(component(x, rate, settled, s.hertz * h as f64) / fundamental))
                        .collect(),
                );
                if s.db == 0.0 {
                    // The device may delay the signal: find where the tone
                    // actually starts.
                    let peak = window.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                    let search = at(s.start - 0.01)..at(s.start + 0.1);
                    let onset = search
                        .clone()
                        .find(|&i| x[i].abs() > 0.1 * peak)
                        .unwrap_or(at(s.start));
                    let steady = rms(window);
                    m.attack.push([
                        db(rms(&x[onset..onset + at(0.01)]) / steady),
                        db(rms(&x[onset + at(0.01)..onset + at(0.05)]) / steady),
                    ]);
                }
            }
            Kind::Recovery => {
                let before = component(x, rate, s.start + RECOVERY[0] - WINDOW - 0.02, s.hertz);
                let end = s.start + RECOVERY[0] + RECOVERY[1];
                // A shorter window here: the point is how the gain moves.
                let short = |start: f64| {
                    let (first, count) = (at(start), at(0.05));
                    let (mut re, mut im) = (0.0, 0.0);
                    for (k, &v) in x[first..first + count].iter().enumerate() {
                        let phase = TAU * s.hertz * k as f64 / rate;
                        re += f64::from(v) * phase.cos();
                        im += f64::from(v) * phase.sin();
                    }
                    2.0 * (re * re + im * im).sqrt() / count as f64
                };
                let reference = short(s.start + RECOVERY[0] - 0.1).max(before * 1e-6);
                m.recovery = RECOVERY_AFTER
                    .iter()
                    .map(|&after| db(short(end + after) / reference))
                    .collect();
            }
        }
    }
    Ok(m)
}

/// How far apart two measurements may be and still count as the same.
pub struct Tolerance {
    pub response_db: f64,
    pub dynamics_db: f64,
    pub harmonics_db: f64,
    pub attack_db: f64,
    pub recovery_db: f64,
    /// Harmonics weaker than this in both devices are not compared, dB
    /// relative to the fundamental.
    pub harmonic_floor_db: f64,
}

impl Default for Tolerance {
    fn default() -> Tolerance {
        Tolerance {
            response_db: 1.0,
            dynamics_db: 1.0,
            harmonics_db: 2.0,
            attack_db: 1.5,
            recovery_db: 1.5,
            harmonic_floor_db: -60.0,
        }
    }
}

/// Print the comparison and return whether the candidate is within
/// tolerance of the reference on every criterion.
pub fn report(reference: &Measurement, candidate: &Measurement, tol: &Tolerance) -> bool {
    // Devices differ in overall gain, and captures are normalised: compare
    // shapes, and report the gain separately.
    let diffs: Vec<f64> = reference
        .level
        .iter()
        .zip(&candidate.level)
        .map(|(r, c)| c - r)
        .collect();
    let offset = diffs.iter().sum::<f64>() / diffs.len() as f64;
    println!("Overall level: the candidate is {offset:+.1} dB from the reference (removed from everything below).\n");
    let mut verdicts: Vec<(&str, f64, f64)> = Vec::new();

    println!("Frequency response, small signal ({RESPONSE_DB} dB), relative to its own average:");
    println!("| Hz | reference dB | candidate dB | difference |");
    println!("|---:|-------------:|-------------:|-----------:|");
    let centre = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (rc, cc) = (centre(&reference.response), centre(&candidate.response));
    let mut worst = 0.0f64;
    for ((hz, r), c) in RESPONSE_HZ
        .iter()
        .zip(&reference.response)
        .zip(&candidate.response)
    {
        let (r, c) = (r - rc, c - cc);
        worst = worst.max((c - r).abs());
        println!("| {hz:.0} | {r:+.1} | {c:+.1} | {:+.1} |", c - r);
    }
    verdicts.push(("frequency response", worst, tol.response_db));

    println!("\nDynamics: output level for each input level (reference's loudest tone = 0 dB):");
    println!("| tone | input dB | reference dB | candidate dB | difference |");
    println!("|-----:|---------:|-------------:|-------------:|-----------:|");
    let top = reference.level.iter().cloned().fold(f64::MIN, f64::max);
    let mut worst = 0.0f64;
    let tones: Vec<(f64, f64)> = TONE_HZ
        .iter()
        .flat_map(|&hz| TONE_DB.iter().map(move |&db| (hz, db)))
        .collect();
    for (((hz, input), r), c) in tones.iter().zip(&reference.level).zip(&candidate.level) {
        let (r, c) = (r - top, c - offset - top);
        worst = worst.max((c - r).abs());
        println!(
            "| {hz:.0} Hz | {input:.0} | {r:+.1} | {c:+.1} | {:+.1} |",
            c - r
        );
    }
    verdicts.push(("dynamics", worst, tol.dynamics_db));

    println!(
        "\nHarmonics, dB relative to the fundamental (only those above {} dB in either):",
        tol.harmonic_floor_db
    );
    println!("| tone | input dB | harmonic | reference | candidate | difference |");
    println!("|-----:|---------:|---------:|----------:|----------:|-----------:|");
    let mut worst = 0.0f64;
    let mut shown = 0;
    for (((hz, input), r), c) in tones
        .iter()
        .zip(&reference.harmonics)
        .zip(&candidate.harmonics)
    {
        for (h, (r, c)) in r.iter().zip(c).enumerate() {
            if r.max(*c) < tol.harmonic_floor_db {
                continue;
            }
            // A harmonic absent from one device counts from the floor.
            let (r, c) = (r.max(tol.harmonic_floor_db), c.max(tol.harmonic_floor_db));
            worst = worst.max((c - r).abs());
            if *input >= -12.0 && h < 4 {
                shown += 1;
                println!(
                    "| {hz:.0} Hz | {input:.0} | H{} | {r:.1} | {c:.1} | {:+.1} |",
                    h + 2,
                    c - r
                );
            }
        }
    }
    if shown == 0 {
        println!("| | | none | | | |");
    }
    println!("(Harmonics 2 to 5 of the three loudest levels are listed; the verdict covers all {} up to H{HARMONICS}.)", tones.len());
    verdicts.push(("harmonics", worst, tol.harmonics_db));

    println!("\nAttack of the loudest tone, relative to the settled tone:");
    println!("| tone | span | reference dB | candidate dB | difference |");
    println!("|-----:|------|-------------:|-------------:|-----------:|");
    let mut worst = 0.0f64;
    for ((hz, r), c) in TONE_HZ.iter().zip(&reference.attack).zip(&candidate.attack) {
        for (span, (r, c)) in ["first 10 ms", "10 to 50 ms"].iter().zip(r.iter().zip(c)) {
            worst = worst.max((c - r).abs());
            println!(
                "| {hz:.0} Hz | {span} | {r:+.1} | {c:+.1} | {:+.1} |",
                c - r
            );
        }
    }
    verdicts.push(("attack", worst, tol.attack_db));

    println!(
        "\nRecovery: gain of a quiet tone after a burst {:.0} dB louder, relative to before:",
        -RESPONSE_DB
    );
    println!("| after the burst | reference dB | candidate dB | difference |");
    println!("|----------------:|-------------:|-------------:|-----------:|");
    let mut worst = 0.0f64;
    for ((after, r), c) in RECOVERY_AFTER
        .iter()
        .zip(&reference.recovery)
        .zip(&candidate.recovery)
    {
        // Below -40 dB the tone is gone in both: do not compare depths.
        let (r, c) = (r.max(-40.0), c.max(-40.0));
        worst = worst.max((c - r).abs());
        println!(
            "| {:.0} ms | {r:+.1} | {c:+.1} | {:+.1} |",
            after * 1e3,
            c - r
        );
    }
    verdicts.push(("recovery", worst, tol.recovery_db));

    println!("\n| criterion | largest difference | tolerance | verdict |");
    println!("|-----------|-------------------:|----------:|---------|");
    let mut same = true;
    for (name, worst, limit) in verdicts {
        let ok = worst <= limit;
        same &= ok;
        println!(
            "| {name} | {worst:.1} dB | {limit:.1} dB | {} |",
            if ok { "same" } else { "DIFFERENT" }
        );
    }
    same
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wire_measures_as_a_wire() {
        let x = probe(48_000);
        let m = measure(&x, 48_000).unwrap();
        for (r, hz) in m.response.iter().zip(RESPONSE_HZ) {
            assert!(
                (r - db(amplitude(RESPONSE_DB))).abs() < 0.01,
                "{hz} Hz: {r}"
            );
        }
        // Each level step shows up as itself, and a sine has no harmonics.
        assert!((m.level[4] - m.level[0] - 36.0).abs() < 0.01);
        assert!(m.harmonics.iter().flatten().all(|&h| h < -100.0));
        assert!(
            m.attack.iter().flatten().all(|a| a.abs() < 0.5),
            "{:?}",
            m.attack
        );
        assert!(
            m.recovery.iter().all(|r| r.abs() < 0.05),
            "{:?}",
            m.recovery
        );
    }

    #[test]
    fn a_clipper_is_told_from_a_wire_and_from_itself_louder() {
        let x = probe(48_000);
        let clipped: Vec<f32> = x.iter().map(|v| (3.0 * v).tanh()).collect();
        let (wire, clip) = (
            measure(&x, 48_000).unwrap(),
            measure(&clipped, 48_000).unwrap(),
        );
        // Odd harmonics appear, even ones do not.
        let loud = &clip.harmonics[4];
        assert!(loud[1] > -20.0 && loud[0] < -80.0, "{loud:?}");
        assert!(!report(&wire, &clip, &Tolerance::default()));
        // The same device 6 dB louder is the same device.
        let louder: Vec<f32> = clipped.iter().map(|v| 2.0 * v).collect();
        assert!(report(
            &clip,
            &measure(&louder, 48_000).unwrap(),
            &Tolerance::default()
        ));
    }
}
