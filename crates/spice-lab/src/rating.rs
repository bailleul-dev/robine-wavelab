//! From evidence to a grade. Nothing is measured here: `grade measure` and
//! `grade device` collect the evidence, this turns it into a letter, so that
//! the same evidence always gives the same grade and the rubric can be
//! tested without ngspice.
//!
//! The rubric is written out for readers in `robine-spice/circuits/GRADES.md`;
//! change one, change the other, and bump [`RUBRIC`].

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version of the rubric. A stored grade made under another version is
/// stale until it is rated again.
pub const RUBRIC: u32 = 1;

/// How far a model is known to be right.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Grade {
    /// Nothing has proved or disproved it.
    #[serde(rename = "unknown")]
    Unknown,
    F,
    D,
    C,
    B,
    A,
    #[serde(rename = "A+")]
    APlus,
}

impl Grade {
    pub fn label(self) -> &'static str {
        match self {
            Grade::Unknown => "unknown",
            Grade::F => "F",
            Grade::D => "D",
            Grade::C => "C",
            Grade::B => "B",
            Grade::A => "A",
            Grade::APlus => "A+",
        }
    }
}

/// What the solver did on the circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Solver {
    /// The netlist parsed and the processor built.
    pub builds: bool,
    /// Every output sample was finite.
    pub finite: bool,
    /// Samples for which Newton hit its iteration cap.
    pub unconverged: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ns_per_sample: Option<f64>,
    /// Why it does not build.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Distance from ngspice on the same flattened deck, in dB relative to the
/// AC part of the output (negative: closer). `None` where ngspice did not
/// finish or the circuit has nothing to compare.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ngspice {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub db_48k: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub db_192k: Option<f64>,
}

/// Controls moved to their ends and middle, against ngspice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Controls {
    pub verified: usize,
    pub total: usize,
}

/// One comparison with a real device (a capture of it, reamped through
/// `nam-lab`) at one setting, as `wavelab null --json` reports it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Device {
    /// Which capture, as a reader can find it again.
    pub capture: String,
    /// Control positions the circuit was set to, and the pedal was set to
    /// when it was captured.
    #[serde(default)]
    pub setting: BTreeMap<String, f64>,
    pub esr_preemphasis_db: f64,
    pub nmr_db: f64,
    /// Fraction of frames where the difference rises above the mask.
    pub disturbed_frames: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peaq_odg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<String>,
    /// Hash of the netlist it was measured on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netlist_hash: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// When the machine-made parts below were measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<String>,
    /// Hash of the netlist they were measured on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netlist_hash: Option<String>,
    /// Where the numbers come from, when it is not `grade measure`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver: Option<Solver>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ngspice: Option<Ngspice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controls: Option<Controls>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<Device>,
}

/// The outcome of rating some evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Rating {
    pub grade: Grade,
    /// 0 to 100, from the axes measured; `None` when no axis was.
    pub score: Option<f64>,
    /// Why this grade, in a line: what failed, what caps it, what is missing.
    pub reason: String,
}

// Weights of the axes. Only the axes with evidence count: the sum is
// renormalised over them.
const W_NGSPICE: f64 = 0.25;
const W_CONTROLS: f64 = 0.15;
const W_DEVICE: f64 = 0.60;

/// Score at which each grade starts.
const D_FROM: f64 = 0.0;
const C_FROM: f64 = 40.0;
const B_FROM: f64 = 60.0;
const A_FROM: f64 = 80.0;
const A_PLUS_FROM: f64 = 92.0;

/// Distinct settings compared with a real device before each grade is
/// possible. Without any, a model is shown to be solved right, not to be
/// the pedal.
const SETTINGS_FOR_B: usize = 1;
const SETTINGS_FOR_A: usize = 3;
const SETTINGS_FOR_A_PLUS: usize = 5;

/// The best grade the comparisons with a real device allow.
fn ceiling(settings: usize) -> (Grade, usize) {
    if settings >= SETTINGS_FOR_A_PLUS {
        (Grade::APlus, SETTINGS_FOR_A_PLUS)
    } else if settings >= SETTINGS_FOR_A {
        (Grade::A, SETTINGS_FOR_A_PLUS)
    } else if settings >= SETTINGS_FOR_B {
        (Grade::B, SETTINGS_FOR_A)
    } else {
        (Grade::C, SETTINGS_FOR_B)
    }
}

/// 0 at `zero`, 1 at `one`, straight between, held outside.
fn ramp(x: f64, zero: f64, one: f64) -> f64 {
    ((x - zero) / (one - zero)).clamp(0.0, 1.0)
}

/// -20 dB from ngspice is 0, -60 dB is 1.
fn ngspice_score(db: f64) -> f64 {
    ramp(db, -20.0, -60.0)
}

fn ngspice_axis(n: &Ngspice) -> Option<f64> {
    let scores: Vec<f64> = [n.db_48k, n.db_192k]
        .into_iter()
        .flatten()
        .map(ngspice_score)
        .collect();
    (!scores.is_empty()).then(|| scores.iter().sum::<f64>() / scores.len() as f64)
}

fn controls_axis(c: &Controls) -> Option<f64> {
    (c.total > 0).then(|| c.verified.min(c.total) as f64 / c.total as f64)
}

/// A comparison with a device: the error-to-signal ratio from -10 dB (0) to
/// -35 dB (1), the share of frames under the masking threshold, and the
/// PEAQ grade from -3.5 (0) to -0.5 (1) when there is one.
fn device_score(d: &Device) -> f64 {
    let mut parts = vec![
        ramp(d.esr_preemphasis_db, -10.0, -35.0),
        1.0 - d.disturbed_frames.clamp(0.0, 1.0),
    ];
    parts.extend(d.peaq_odg.map(|odg| ramp(odg, -3.5, -0.5)));
    parts.iter().sum::<f64>() / parts.len() as f64
}

fn device_axis(devices: &[Device]) -> Option<f64> {
    (!devices.is_empty())
        .then(|| devices.iter().map(device_score).sum::<f64>() / devices.len() as f64)
}

fn from_score(score: f64) -> Grade {
    if score >= A_PLUS_FROM {
        Grade::APlus
    } else if score >= A_FROM {
        Grade::A
    } else if score >= B_FROM {
        Grade::B
    } else if score >= C_FROM {
        Grade::C
    } else {
        debug_assert!(score >= D_FROM);
        Grade::D
    }
}

/// Distinct settings among the comparisons with a device.
fn distinct_settings(devices: &[Device]) -> usize {
    let mut seen: Vec<&BTreeMap<String, f64>> = Vec::new();
    for d in devices {
        if !seen.contains(&&d.setting) {
            seen.push(&d.setting);
        }
    }
    seen.len()
}

/// Rate some evidence.
///
/// * No solver evidence, or a solver that ran clean but nothing compared
///   with ngspice or a device: `unknown`.
/// * A circuit that does not build, gives a sample that is not finite or
///   leaves Newton unconverged: `F`, whatever else is known.
/// * Otherwise the score of the axes with evidence gives the grade, and the
///   grade is capped by what has been compared with: without a real device
///   at most `C` (the solver is right, the pedal is not shown to be), with
///   one setting of one at most `B`, with three at most `A`, and `A+` takes
///   five.
pub fn rate(e: &Evidence) -> Rating {
    let Some(solver) = &e.solver else {
        return Rating {
            grade: Grade::Unknown,
            score: None,
            reason: "not measured".into(),
        };
    };
    if !solver.builds {
        return fail(&match &solver.error {
            Some(error) => format!("does not build: {error}"),
            None => "does not build".into(),
        });
    }
    if !solver.finite {
        return fail("the output is not finite");
    }
    if solver.unconverged > 0 {
        return fail(&format!(
            "{} sample(s) left Newton unconverged",
            solver.unconverged
        ));
    }

    let axes = [
        (e.ngspice.as_ref().and_then(ngspice_axis), W_NGSPICE),
        (e.controls.as_ref().and_then(controls_axis), W_CONTROLS),
        (device_axis(&e.devices), W_DEVICE),
    ];
    let (sum, weight) = axes
        .iter()
        .filter_map(|(a, w)| a.map(|a| (a * w, *w)))
        .fold((0.0, 0.0), |(s, w), (a, b)| (s + a, w + b));
    if weight == 0.0 {
        return Rating {
            grade: Grade::Unknown,
            score: None,
            reason: "runs clean, but never compared with ngspice or a real device".into(),
        };
    }
    let score = 100.0 * sum / weight;
    let mut grade = from_score(score);
    let mut reason = format!("score {score:.0}");
    let settings = distinct_settings(&e.devices);
    let (best, next) = ceiling(settings);
    if grade > best {
        grade = best;
        reason.push_str(&format!(", held at {}", best.label()));
    }
    if best != Grade::APlus {
        reason.push_str(&format!(
            ": {settings} of {next} settings against a real device"
        ));
    }
    Rating {
        grade,
        score: Some(score),
        reason,
    }
}

fn fail(why: &str) -> Rating {
    Rating {
        grade: Grade::F,
        score: Some(0.0),
        reason: why.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> Solver {
        Solver {
            builds: true,
            finite: true,
            unconverged: 0,
            ns_per_sample: None,
            error: None,
        }
    }

    fn ngspice(db_48k: f64, db_192k: f64) -> Ngspice {
        Ngspice {
            db_48k: Some(db_48k),
            db_192k: Some(db_192k),
        }
    }

    fn device(esr: f64, disturbed: f64, odg: Option<f64>, drive: f64) -> Device {
        Device {
            capture: "capture".into(),
            setting: BTreeMap::from([("Drive".to_string(), drive)]),
            esr_preemphasis_db: esr,
            nmr_db: -20.0,
            disturbed_frames: disturbed,
            peaq_odg: odg,
            measured: None,
            netlist_hash: None,
        }
    }

    fn with(ng: Option<Ngspice>, controls: Option<Controls>, devices: Vec<Device>) -> Evidence {
        Evidence {
            solver: Some(clean()),
            ngspice: ng,
            controls,
            devices,
            ..Evidence::default()
        }
    }

    #[test]
    fn nothing_measured_is_unknown() {
        let r = rate(&Evidence::default());
        assert_eq!(r.grade, Grade::Unknown);
        assert_eq!(r.score, None);
    }

    #[test]
    fn clean_solver_alone_is_unknown() {
        assert_eq!(rate(&with(None, None, vec![])).grade, Grade::Unknown);
        // An ngspice record with no number in it compares nothing either.
        let r = rate(&with(Some(Ngspice::default()), None, vec![]));
        assert_eq!(r.grade, Grade::Unknown);
    }

    #[test]
    fn a_broken_solver_is_f_whatever_else_is_known() {
        let good = ngspice(-80.0, -80.0);
        for solver in [
            Solver {
                builds: false,
                ..clean()
            },
            Solver {
                finite: false,
                ..clean()
            },
            Solver {
                unconverged: 1,
                ..clean()
            },
        ] {
            let e = Evidence {
                solver: Some(solver),
                ngspice: Some(good.clone()),
                devices: vec![device(-40.0, 0.0, Some(0.0), 0.5)],
                ..Evidence::default()
            };
            let r = rate(&e);
            assert_eq!(r.grade, Grade::F, "{}", r.reason);
        }
    }

    #[test]
    fn the_ngspice_ramp_runs_from_20_to_60_db() {
        assert_eq!(ngspice_score(-10.0), 0.0);
        assert_eq!(ngspice_score(-20.0), 0.0);
        assert!((ngspice_score(-40.0) - 0.5).abs() < 1e-12);
        assert_eq!(ngspice_score(-60.0), 1.0);
        assert_eq!(ngspice_score(-150.0), 1.0);
    }

    #[test]
    fn ngspice_alone_gives_c_at_most() {
        // Perfect against ngspice, nothing to say about the pedal.
        let r = rate(&with(Some(ngspice(-100.0, -100.0)), None, vec![]));
        assert_eq!(r.grade, Grade::C);
        assert_eq!(r.score, Some(100.0));
        assert_eq!(
            r.reason,
            "score 100, held at C: 0 of 1 settings against a real device"
        );
    }

    #[test]
    fn ngspice_alone_ranks_below_the_cap() {
        // -40 dB both ways is a score of 50.
        let r = rate(&with(Some(ngspice(-40.0, -40.0)), None, vec![]));
        assert_eq!(r.grade, Grade::C);
        assert_eq!(r.score, Some(50.0));
        // Below a score of 40, a D.
        let r = rate(&with(Some(ngspice(-30.0, -30.0)), None, vec![]));
        assert_eq!(r.grade, Grade::D);
        // -20 dB or worse everywhere is a D, not an F: it runs.
        let r = rate(&with(Some(ngspice(-5.0, -10.0)), None, vec![]));
        assert_eq!(r.grade, Grade::D);
    }

    #[test]
    fn a_missing_rate_does_not_count_against_the_circuit() {
        let one = Ngspice {
            db_48k: Some(-60.0),
            db_192k: None,
        };
        assert_eq!(rate(&with(Some(one), None, vec![])).score, Some(100.0));
    }

    #[test]
    fn unverified_controls_pull_the_score_down() {
        let ng = Some(ngspice(-60.0, -60.0));
        let all = Controls {
            verified: 3,
            total: 3,
        };
        let some = Controls {
            verified: 1,
            total: 3,
        };
        let a = rate(&with(ng.clone(), Some(all), vec![])).score.unwrap();
        let b = rate(&with(ng, Some(some), vec![])).score.unwrap();
        assert_eq!(a, 100.0);
        // (0.25 * 1 + 0.15 * 1/3) / 0.40
        assert!((b - 75.0).abs() < 1e-9, "{b}");
    }

    #[test]
    fn a_circuit_without_controls_has_no_controls_axis() {
        let none = Controls {
            verified: 0,
            total: 0,
        };
        let r = rate(&with(Some(ngspice(-60.0, -60.0)), Some(none), vec![]));
        assert_eq!(r.score, Some(100.0));
    }

    #[test]
    fn a_device_outweighs_ngspice() {
        let ng = Some(ngspice(-60.0, -60.0));
        let poor = device(-10.0, 1.0, Some(-3.5), 0.5);
        let r = rate(&with(ng, None, vec![poor]));
        // (0.25 * 1 + 0.60 * 0) / 0.85
        assert!((r.score.unwrap() - 29.41).abs() < 0.01, "{:?}", r.score);
        assert_eq!(r.grade, Grade::D);
    }

    #[test]
    fn the_settings_against_a_real_device_raise_the_ceiling() {
        let ng = Some(ngspice(-60.0, -60.0));
        let at = |n: usize| -> Vec<Device> {
            (0..n)
                .map(|k| device(-35.0, 0.0, Some(-0.5), k as f64 / 10.0))
                .collect()
        };
        let grade = |n: usize| {
            let r = rate(&with(ng.clone(), None, at(n)));
            assert_eq!(r.score, Some(100.0));
            r.grade
        };
        assert_eq!(grade(0), Grade::C);
        assert_eq!(grade(1), Grade::B);
        assert_eq!(grade(2), Grade::B);
        assert_eq!(grade(3), Grade::A);
        assert_eq!(grade(4), Grade::A);
        assert_eq!(grade(5), Grade::APlus);
    }

    #[test]
    fn the_same_setting_five_times_is_one_setting() {
        let ng = Some(ngspice(-60.0, -60.0));
        let devices = vec![device(-35.0, 0.0, Some(-0.5), 0.5); 5];
        assert_eq!(rate(&with(ng, None, devices)).grade, Grade::B);
    }

    #[test]
    fn a_poor_match_with_the_device_is_not_lifted_by_the_ceiling() {
        let ng = Some(ngspice(-60.0, -60.0));
        let devices: Vec<Device> = (0..5)
            .map(|k| device(-12.0, 0.8, Some(-3.0), k as f64 / 10.0))
            .collect();
        assert!(rate(&with(ng, None, devices)).grade <= Grade::D);
    }

    #[test]
    fn the_peaq_grade_is_optional() {
        let without = device(-35.0, 0.0, None, 0.5);
        assert_eq!(device_score(&without), 1.0);
        let worse = device(-35.0, 0.0, Some(-3.5), 0.5);
        assert!((device_score(&worse) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn grades_order_from_unknown_to_a_plus() {
        let order = [
            Grade::Unknown,
            Grade::F,
            Grade::D,
            Grade::C,
            Grade::B,
            Grade::A,
            Grade::APlus,
        ];
        assert!(order.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn grades_are_written_as_their_label() {
        for g in [Grade::Unknown, Grade::F, Grade::B, Grade::APlus] {
            let text = serde_json::to_string(&g).unwrap();
            assert_eq!(text, format!("\"{}\"", g.label()));
            assert_eq!(serde_json::from_str::<Grade>(&text).unwrap(), g);
        }
    }

    #[test]
    fn the_boundaries_belong_to_the_higher_grade() {
        assert_eq!(from_score(39.99), Grade::D);
        assert_eq!(from_score(40.0), Grade::C);
        assert_eq!(from_score(60.0), Grade::B);
        assert_eq!(from_score(80.0), Grade::A);
        assert_eq!(from_score(92.0), Grade::APlus);
    }
}
