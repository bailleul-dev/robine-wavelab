//! A network that stands in for a circuit's nonlinear solve: training it
//! against Newton, and judging it in the regimes where the circuit is worth
//! simulating.
//!
//! `train` plays a circuit with its own solver through guitar-like
//! excitations, its controls moved at random, and records every sample the
//! drive of the nonlinear ports, the positions of the controls that shape
//! the solve, and the currents Newton found. A perceptron learns that map.
//!
//! `eval` is the judge, and what it asks matters more than the loss does:
//! the network is run *in the loop*, its currents feeding the states that
//! make the next drive, on soft and hard notes, chords, bright loud notes
//! and a long loud low burst that sags the supply, and the output is
//! compared with Newton's. A network that is right on average and wrong
//! when the amp collapses is not the amp.

use crate::commands::{build, load_with, positions, CmdResult};
use clap::{Args, Subcommand};
use robine_spice::{activation, activation_slope, Processor, Solver, Surrogate};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args)]
pub struct NnArgs {
    #[command(subcommand)]
    command: NnCommand,
}

#[derive(Subcommand)]
enum NnCommand {
    /// Train a network on a circuit, against its own solver.
    Train(TrainArgs),
    /// Run a network in the loop against Newton, in the hard regimes.
    Eval(EvalArgs),
    /// List a circuit's devices, the numbers `micro-train` takes.
    MicroList(crate::micro::MicroListArgs),
    /// Learn the law of each chosen component with a small network.
    MicroTrain(crate::micro::MicroTrainArgs),
    /// Run the circuit with those networks in the loop against the analytic laws.
    MicroEval(crate::micro::MicroEvalArgs),
}

#[derive(Args)]
pub struct TrainArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Where the network goes.
    #[arg(short, long)]
    output: PathBuf,
    /// Controls the network reads, by name, comma separated. By default
    /// every control that changes the nonlinear problem.
    #[arg(long, value_delimiter = ',')]
    controls: Vec<String>,
    /// Sizes of the hidden layers, comma separated.
    #[arg(long, value_delimiter = ',', default_value = "64,64")]
    hidden: Vec<usize>,
    /// Samples recorded, in all.
    #[arg(long, default_value_t = 2_000_000)]
    samples: usize,
    /// Passes over the samples.
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    /// Rows in a step.
    #[arg(long, default_value_t = 2048)]
    batch: usize,
    /// First learning rate, which falls to a twentieth of itself.
    #[arg(long, default_value_t = 3e-3)]
    rate: f32,
    /// Largest volts at the circuit input in the excitations.
    #[arg(long, default_value_t = 2.0)]
    volts: f64,
    /// Sample rate.
    #[arg(long, default_value_t = 48_000.0)]
    sample_rate: f64,
    /// Seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Threads.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Args)]
pub struct EvalArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// The network.
    #[arg(short, long)]
    model: PathBuf,
    /// Sample rate.
    #[arg(long, default_value_t = 48_000.0)]
    sample_rate: f64,
    /// Largest volts at the circuit input.
    #[arg(long, default_value_t = 2.0)]
    volts: f64,
}

pub fn nn(args: NnArgs) -> CmdResult {
    match args.command {
        NnCommand::Train(a) => train(a),
        NnCommand::Eval(a) => eval(a),
        NnCommand::MicroList(a) => crate::micro::list(a),
        NnCommand::MicroTrain(a) => crate::micro::train(a),
        NnCommand::MicroEval(a) => crate::micro::eval(a),
    }
}

// ---------------------------------------------------------------- random

/// SplitMix64: small, and the same everywhere.
#[derive(Clone)]
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub(crate) fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }
    /// Log-uniform between two positive numbers.
    pub(crate) fn log_between(&mut self, low: f64, high: f64) -> f64 {
        (low.ln() + (high.ln() - low.ln()) * self.uniform()).exp()
    }
    fn gauss(&mut self) -> f64 {
        let (a, b) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

// ----------------------------------------------------------- excitations

/// What is played.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    Sine,
    Pluck,
    Chord,
    Noise,
    /// A long, loud, low note: the supply sags under it.
    Burst,
    Silence,
}

const KINDS: [(Kind, f64); 6] = [
    (Kind::Sine, 0.18),
    (Kind::Pluck, 0.30),
    (Kind::Chord, 0.20),
    (Kind::Noise, 0.08),
    (Kind::Burst, 0.17),
    (Kind::Silence, 0.07),
];

pub(crate) fn pick(rng: &mut Rng) -> Kind {
    let mut x = rng.uniform();
    for (kind, share) in KINDS {
        if x < share {
            return kind;
        }
        x -= share;
    }
    Kind::Pluck
}

/// A plucked string: harmonics falling as `1/n^a`, the high ones faster.
fn pluck(out: &mut [f32], rate: f64, f0: f64, peak: f64, decay: f64, bright: f64, at: usize) {
    for (n, o) in out.iter_mut().enumerate().skip(at) {
        let t = (n - at) as f64 / rate;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for h in 1..=8usize {
            let a = 1.0 / (h as f64).powf(bright);
            let fade = (-t * (1.0 + 0.7 * (h - 1) as f64) / decay).exp();
            sum += a * fade * (std::f64::consts::TAU * f0 * h as f64 * t).sin();
            norm += a;
        }
        // A little attack so there is no click.
        let attack = (t * 1000.0).min(1.0);
        *o += (peak * attack * sum / norm) as f32;
    }
}

/// `seconds` of a kind of excitation, `volts` being the largest peak.
pub(crate) fn excitation(
    kind: Kind,
    rng: &mut Rng,
    rate: f64,
    seconds: f64,
    volts: f64,
) -> Vec<f32> {
    let n = (seconds * rate) as usize;
    let mut out = vec![0.0f32; n];
    match kind {
        Kind::Silence => {
            // A little hum and hiss, as an input at rest has.
            let level = rng.log_between(1e-5, 1e-3);
            for o in &mut out {
                *o = (level * rng.gauss()) as f32;
            }
        }
        Kind::Sine => {
            let f = rng.log_between(60.0, 6000.0);
            let peak = rng.log_between(0.003, volts);
            let on = rng.between(0.2, 1.0);
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f64 / rate;
                let gate = if (t / seconds) < on { 1.0 } else { 0.0 };
                *o = (gate * peak * (std::f64::consts::TAU * f * t).sin()) as f32;
            }
        }
        Kind::Pluck => {
            let f0 = rng.log_between(80.0, 700.0);
            let peak = rng.log_between(0.01, volts);
            let at = (rng.between(0.0, 0.2) * rate) as usize;
            pluck(
                &mut out,
                rate,
                f0,
                peak,
                rng.between(0.2, 1.4),
                rng.between(0.6, 1.6),
                at.min(n),
            );
        }
        Kind::Chord => {
            let strings = 3 + (rng.next() % 3) as usize;
            let peak = rng.log_between(0.01, volts) / (strings as f64).sqrt();
            let root = rng.log_between(80.0, 330.0);
            let bright = rng.between(0.6, 1.5);
            let decay = rng.between(0.3, 1.5);
            for s in 0..strings {
                let f0 = root * 2f64.powf((s as f64 * 4.0 + rng.between(0.0, 3.0)) / 12.0);
                let at = (rng.between(0.0, 0.04) * rate) as usize + s * 200;
                pluck(
                    &mut out,
                    rate,
                    f0.min(1200.0),
                    peak,
                    decay,
                    bright,
                    at.min(n),
                );
            }
        }
        Kind::Noise => {
            let peak = rng.log_between(0.01, volts);
            let cutoff = rng.log_between(300.0, 8000.0);
            let a = 1.0 - (-std::f64::consts::TAU * cutoff / rate).exp();
            let (mut lp, on) = (0.0, rng.between(0.2, 0.9));
            for (i, o) in out.iter_mut().enumerate() {
                lp += a * (rng.gauss() - lp);
                if (i as f64 / n as f64) < on {
                    *o = (peak * lp * 1.5) as f32;
                }
            }
        }
        Kind::Burst => {
            // Loud and low, held: it draws on the supply and the amp sags.
            let f = rng.log_between(70.0, 160.0);
            let peak = rng.between(0.5, 1.0) * volts * 1.3;
            let on = rng.between(0.4, 0.9);
            for (i, o) in out.iter_mut().enumerate() {
                let t = i as f64 / rate;
                if (t / seconds) < on {
                    let ramp = (t * 200.0).min(1.0);
                    *o = (ramp * peak * (std::f64::consts::TAU * f * t).sin()) as f32;
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------ data

/// The circuit, played with its own solver: what the network learns from.
struct Data {
    inputs: usize,
    /// Rows of `inputs` values (the drive, then the controls), and rows of
    /// `ports` currents.
    x: Vec<f32>,
    i: Vec<f64>,
}

fn record(
    template: &Processor,
    controls: &[usize],
    samples: usize,
    rate: f64,
    volts: f64,
    seed: u64,
) -> Data {
    let mut processor = template.clone();
    let ports = processor.nonlinear_port_count();
    // Inputs: the drive of the ports, the controls, and the previous
    // sample's solved port voltages. The previous solution is already an
    // excellent start on most samples, so with it the network has only
    // the jump between two samples to learn, not the whole map: trained
    // without it, a guide made Newton slower (3 to 11 steps a sample).
    let inputs = 2 * ports + controls.len();
    let mut rng = Rng(seed);
    let mut data = Data {
        inputs,
        x: Vec::with_capacity(samples * inputs),
        i: Vec::with_capacity(samples * ports),
    };
    while data.i.len() < samples * ports {
        // A new setting of the controls, and a new thing played. Three
        // times in ten, every control at one level together, most often
        // near the top: drawn one by one they are almost never all up at
        // once, and a player turns everything up (the micro trainer
        // learnt this the hard way, see `micro::trajectory`).
        let together = (rng.uniform() < 0.3).then(|| {
            let r = rng.uniform();
            if r < 0.6 {
                rng.between(0.8, 1.0)
            } else if r < 0.8 {
                rng.between(0.0, 0.2)
            } else {
                rng.uniform()
            }
        });
        let mut values = Vec::with_capacity(controls.len());
        for &c in controls {
            let v = match together {
                Some(level) => level,
                // The ends of the travel are where things break.
                None if rng.uniform() < 0.15 => f64::from(u8::from(rng.uniform() < 0.5)),
                None => rng.uniform(),
            };
            processor.set_control(c, v);
            values.push(v);
        }
        let seconds = rng.between(0.4, 2.0);
        // A third of the time, the hard things at full level: the random
        // levels of the other excitations are spread over decades and
        // rarely reach it, and a guide never shown the amp driven past
        // cutoff cannot start Newton there.
        let signal = if rng.uniform() < 1.0 / 3.0 {
            let peak = volts * rng.between(0.7, 1.0);
            if rng.uniform() < 0.5 {
                let f = rng.log_between(1000.0, 6000.0);
                (0..(seconds * rate) as usize)
                    .map(|i| (peak * (std::f64::consts::TAU * f * i as f64 / rate).sin()) as f32)
                    .collect()
            } else {
                let kind = if rng.uniform() < 0.5 { Kind::Burst } else { Kind::Chord };
                let mut s = excitation(kind, &mut rng, rate, seconds, volts);
                let top = s.iter().fold(0.0f32, |t, v| t.max(v.abs())).max(1e-9);
                let gain = (peak / f64::from(top)) as f32;
                s.iter_mut().for_each(|v| *v *= gain);
                s
            }
        } else {
            excitation(pick(&mut rng), &mut rng, rate, seconds, volts)
        };
        for &x in &signal {
            let previous: Vec<f64> = processor.port_voltages().to_vec();
            let drive: Vec<f64> = processor.port_drive(x).to_vec();
            processor.process_sample(x);
            data.x.extend(drive.iter().map(|&d| d as f32));
            data.x.extend(values.iter().map(|&v| v as f32));
            data.x.extend(previous.iter().map(|&v| v as f32));
            data.i.extend_from_slice(processor.port_currents());
        }
        if processor.solver_stats().failures > 0 {
            // Newton gave up somewhere; do not learn from a blown state.
            processor.reset();
        }
    }
    data
}

// -------------------------------------------------------------- training

pub(crate) struct Net {
    pub(crate) sizes: Vec<usize>,
    /// Per layer, `out x in` row-major.
    pub(crate) w: Vec<Vec<f32>>,
    pub(crate) b: Vec<Vec<f32>>,
}

impl Net {
    pub(crate) fn new(sizes: Vec<usize>, rng: &mut Rng) -> Net {
        let mut w = Vec::new();
        let mut b = Vec::new();
        for pair in sizes.windows(2) {
            let (i, o) = (pair[0], pair[1]);
            let limit = (6.0 / (i + o) as f64).sqrt();
            w.push(
                (0..i * o)
                    .map(|_| rng.between(-limit, limit) as f32)
                    .collect(),
            );
            b.push(vec![0.0; o]);
        }
        Net { sizes, w, b }
    }

    pub(crate) fn layers(&self) -> usize {
        self.w.len()
    }

    /// The forward pass of one row; fills `z` (pre-activations) and `a`
    /// (activations, `a[0]` being the input).
    fn forward(&self, x: &[f32], z: &mut [Vec<f32>], a: &mut [Vec<f32>]) {
        a[0].copy_from_slice(x);
        let last = self.layers() - 1;
        for l in 0..=last {
            let (i, o) = (self.sizes[l], self.sizes[l + 1]);
            let (input, rest) = a.split_at_mut(l + 1);
            let input = &input[l];
            for k in 0..o {
                let row = &self.w[l][k * i..(k + 1) * i];
                let mut s = self.b[l][k];
                for (w, x) in row.iter().zip(input) {
                    s += w * x;
                }
                z[l][k] = s;
                rest[0][k] = if l < last { activation(s) } else { s };
            }
        }
    }
}

pub(crate) struct Grads {
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
}

impl Grads {
    pub(crate) fn zeros(net: &Net) -> Grads {
        Grads {
            w: net.w.iter().map(|w| vec![0.0; w.len()]).collect(),
            b: net.b.iter().map(|b| vec![0.0; b.len()]).collect(),
        }
    }
    pub(crate) fn add(&mut self, other: &Grads) {
        for (a, b) in self
            .w
            .iter_mut()
            .zip(&other.w)
            .chain(self.b.iter_mut().zip(&other.b))
        {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

/// Loss and gradient over rows `from..to` of the normalised data.
pub(crate) fn gradient(
    net: &Net,
    x: &[f32],
    y: &[f32],
    rows: &[usize],
    scale: f32,
) -> (f64, Grads) {
    weighted_gradient(net, x, y, None, rows, scale)
}

/// The same, each output of each row weighing as `weights` says (laid out
/// like `y`).
pub(crate) fn weighted_gradient(
    net: &Net,
    x: &[f32],
    y: &[f32],
    weights: Option<&[f32]>,
    rows: &[usize],
    scale: f32,
) -> (f64, Grads) {
    let inputs = net.sizes[0];
    let outputs = *net.sizes.last().unwrap();
    let mut grads = Grads::zeros(net);
    let mut z: Vec<Vec<f32>> = net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
    let mut a: Vec<Vec<f32>> = net.sizes.iter().map(|&s| vec![0.0; s]).collect();
    let mut loss = 0.0f64;
    let last = net.layers() - 1;
    let mut delta: Vec<Vec<f32>> = net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
    for &r in rows {
        net.forward(&x[r * inputs..(r + 1) * inputs], &mut z, &mut a);
        let target = &y[r * outputs..(r + 1) * outputs];
        for k in 0..outputs {
            let e = a[last + 1][k] - target[k];
            let weight = weights.map_or(1.0, |w| w[r * outputs + k]);
            loss += f64::from(weight * e * e);
            delta[last][k] = 2.0 * weight * e * scale;
        }
        for l in (0..=last).rev() {
            let (i, o) = (net.sizes[l], net.sizes[l + 1]);
            for k in 0..o {
                let d = delta[l][k];
                grads.b[l][k] += d;
                let g = &mut grads.w[l][k * i..(k + 1) * i];
                for (g, x) in g.iter_mut().zip(&a[l]) {
                    *g += d * x;
                }
            }
            if l > 0 {
                let (before, now) = delta.split_at_mut(l);
                for j in 0..i {
                    let mut s = 0.0;
                    for k in 0..o {
                        s += net.w[l][k * i + j] * now[0][k];
                    }
                    before[l - 1][j] = s * activation_slope(z[l - 1][j]);
                }
            }
        }
    }
    (loss, grads)
}

/// The activation's second derivative, `-16x(9-x^2) / (3(3+x^2)^3)` inside
/// its knees and nothing past them (its slope is `((9-x^2) / (3(3+x^2)))^2`).
fn activation_curve(x: f32) -> f32 {
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let x2 = x * x;
    let d = 3.0 + x2;
    -16.0 * x * (9.0 - x2) / (3.0 * d * d * d)
}

/// Loss and gradient when the network is taught the slopes of what it
/// learns as well as its values (Sobolev training): `slopes` holds, per
/// row, `d y[c] / d x[j]` at `c * inputs + j`, and each is matched with
/// a weight of `slope_weight / (1 + slope^2)`, absolute where the target
/// is flat and relative where it is steep. A solver that takes a
/// network's Jacobian for the law's needs it right, not only its values.
/// Returns the value loss, the slope loss and the gradient of their sum.
pub(crate) fn sobolev_gradient(
    net: &Net,
    x: &[f32],
    y: &[f32],
    slopes: &[f32],
    slope_weight: f32,
    rows: &[usize],
    scale: f32,
) -> (f64, f64, Grads) {
    let inputs = net.sizes[0];
    let outputs = *net.sizes.last().unwrap();
    let mut grads = Grads::zeros(net);
    let last = net.layers() - 1;
    let mut z: Vec<Vec<f32>> = net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
    let mut a: Vec<Vec<f32>> = net.sizes.iter().map(|&s| vec![0.0; s]).collect();
    // Tangents along each input: of z[l] and of a[l].
    let mut tz: Vec<Vec<Vec<f32>>> = (0..inputs)
        .map(|_| net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect())
        .collect();
    let mut ta: Vec<Vec<Vec<f32>>> = (0..inputs)
        .map(|_| net.sizes.iter().map(|&s| vec![0.0; s]).collect())
        .collect();
    // Adjoints of z[l] and of each tangent of z[l].
    let mut gz: Vec<Vec<f32>> = net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
    let mut gtz: Vec<Vec<Vec<f32>>> = (0..inputs)
        .map(|_| net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect())
        .collect();
    let mut ga = vec![0.0f32; *net.sizes.iter().max().unwrap()];
    let mut gta = vec![vec![0.0f32; ga.len()]; inputs];
    let (mut value_loss, mut slope_loss) = (0.0f64, 0.0f64);
    for &r in rows {
        net.forward(&x[r * inputs..(r + 1) * inputs], &mut z, &mut a);
        for j in 0..inputs {
            ta[j][0].iter_mut().enumerate().for_each(|(q, t)| *t = f32::from(u8::from(q == j)));
            for l in 0..=last {
                let (i, o) = (net.sizes[l], net.sizes[l + 1]);
                for k in 0..o {
                    let row = &net.w[l][k * i..(k + 1) * i];
                    let s: f32 = row.iter().zip(&ta[j][l]).map(|(w, t)| w * t).sum();
                    tz[j][l][k] = s;
                    if l < last {
                        ta[j][l + 1][k] = activation_slope(z[l][k]) * s;
                    }
                }
            }
        }
        let target = &y[r * outputs..(r + 1) * outputs];
        let want = &slopes[r * outputs * inputs..(r + 1) * outputs * inputs];
        for c in 0..outputs {
            let e = a[last + 1][c] - target[c];
            value_loss += f64::from(e * e);
            gz[last][c] = 2.0 * e * scale;
            for j in 0..inputs {
                let t = want[c * inputs + j];
                let w = slope_weight / (1.0 + t * t);
                let e = tz[j][last][c] - t;
                slope_loss += f64::from(w * e * e);
                gtz[j][last][c] = 2.0 * w * e * scale;
            }
        }
        for l in (0..=last).rev() {
            let (i, o) = (net.sizes[l], net.sizes[l + 1]);
            for k in 0..o {
                let d = gz[l][k];
                grads.b[l][k] += d;
                let g = &mut grads.w[l][k * i..(k + 1) * i];
                for (q, g) in g.iter_mut().enumerate() {
                    let mut s = d * a[l][q];
                    for j in 0..inputs {
                        s += gtz[j][l][k] * ta[j][l][q];
                    }
                    *g += s;
                }
            }
            if l > 0 {
                for q in 0..i {
                    let mut s = 0.0;
                    for k in 0..o {
                        s += net.w[l][k * i + q] * gz[l][k];
                    }
                    ga[q] = s;
                    for j in 0..inputs {
                        let mut s = 0.0;
                        for k in 0..o {
                            s += net.w[l][k * i + q] * gtz[j][l][k];
                        }
                        gta[j][q] = s;
                    }
                }
                let (before, _) = gz.split_at_mut(l);
                for q in 0..i {
                    let zz = z[l - 1][q];
                    let slope = activation_slope(zz);
                    let curve = activation_curve(zz);
                    let mut s = slope * ga[q];
                    for j in 0..inputs {
                        s += curve * tz[j][l - 1][q] * gta[j][q];
                        gtz[j][l - 1][q] = slope * gta[j][q];
                    }
                    before[l - 1][q] = s;
                }
            }
        }
    }
    (value_loss, slope_loss, grads)
}

pub(crate) struct Adam {
    pub(crate) m: Grads,
    pub(crate) v: Grads,
    pub(crate) step: i32,
}

impl Adam {
    pub(crate) fn update(&mut self, net: &mut Net, g: &Grads, rate: f32) {
        self.step += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let c1 = 1.0 - b1.powi(self.step);
        let c2 = 1.0 - b2.powi(self.step);
        let step = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
            for k in 0..p.len() {
                m[k] = b1 * m[k] + (1.0 - b1) * g[k];
                v[k] = b2 * v[k] + (1.0 - b2) * g[k] * g[k];
                p[k] -= rate * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
            }
        };
        for l in 0..net.layers() {
            step(&mut net.w[l], &g.w[l], &mut self.m.w[l], &mut self.v.w[l]);
            step(&mut net.b[l], &g.b[l], &mut self.m.b[l], &mut self.v.b[l]);
        }
    }
}

pub(crate) fn threads(wanted: Option<usize>) -> usize {
    wanted.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

fn train(args: TrainArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let template = build(&circuit, args.sample_rate, &positions(&circuit, &[])?)?;
    let ports = template.nonlinear_port_count();
    if ports == 0 {
        return Err("the circuit has no nonlinear port: nothing to learn".into());
    }
    let names: Vec<String> = if args.controls.is_empty() {
        template
            .controls()
            .enumerate()
            .filter(|(i, _)| template.control_shapes_nonlinear_solve(*i))
            .map(|(_, c)| c.name.clone())
            .collect()
    } else {
        args.controls.clone()
    };
    let mut indices = Vec::new();
    for name in &names {
        indices.push(
            template
                .control_index(name)
                .ok_or_else(|| format!("the circuit has no control `{name}`"))?,
        );
    }
    let threads = threads(args.threads);
    println!(
        "{}: {ports} nonlinear ports, controls read: {}",
        args.circuit.display(),
        if names.is_empty() {
            "none".into()
        } else {
            names.join(", ")
        }
    );

    // ---- record
    let began = Instant::now();
    let per_thread = args.samples.div_ceil(threads);
    let parts: Vec<Data> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (template, indices) = (&template, &indices);
                let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(t as u64);
                let (rate, volts) = (args.sample_rate, args.volts);
                scope.spawn(move || record(template, indices, per_thread, rate, volts, seed))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let inputs = parts[0].inputs;
    let mut x: Vec<f32> = Vec::new();
    let mut currents: Vec<f64> = Vec::new();
    for p in &parts {
        x.extend_from_slice(&p.x);
        currents.extend_from_slice(&p.i);
    }
    let rows = currents.len() / ports;
    println!(
        "recorded {rows} samples in {:.1} s ({threads} threads)",
        began.elapsed().as_secs_f64()
    );

    // ---- normalise
    let mut mean = vec![0.0f64; inputs];
    let mut var = vec![0.0f64; inputs];
    for row in x.chunks_exact(inputs) {
        for (k, v) in row.iter().enumerate() {
            mean[k] += f64::from(*v);
        }
    }
    mean.iter_mut().for_each(|m| *m /= rows as f64);
    for row in x.chunks_exact(inputs) {
        for (k, v) in row.iter().enumerate() {
            var[k] += (f64::from(*v) - mean[k]).powi(2);
        }
    }
    let input_mean: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
    let input_scale: Vec<f32> = var
        .iter()
        .map(|&v| ((v / rows as f64).sqrt() as f32).max(1e-3))
        .collect();
    // The scale of a current is its median: the level it sits at most of
    // the time, from which the network is asked for the logarithm of how
    // far above it goes.
    let mut output_scale = vec![0.0f32; ports];
    for (k, s) in output_scale.iter_mut().enumerate() {
        let mut magnitudes: Vec<f64> = currents
            .iter()
            .skip(k)
            .step_by(ports)
            .map(|c| c.abs())
            .collect();
        magnitudes.sort_by(|a, b| a.total_cmp(b));
        *s = (magnitudes[magnitudes.len() / 2] as f32).max(1e-18);
    }
    let xn: Vec<f32> = x
        .chunks_exact(inputs)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(k, v)| (v - input_mean[k]) / input_scale[k])
                .collect::<Vec<_>>()
        })
        .collect();
    drop(x);
    // Targets: asinh(i / s), then standardised, the standardising being
    // folded into the last layer on the way out.
    let mut ys: Vec<f32> = currents
        .chunks_exact(ports)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(k, &c)| (c / f64::from(output_scale[k])).asinh() as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    let mut y_mean = vec![0.0f32; ports];
    let mut y_scale = vec![0.0f32; ports];
    for k in 0..ports {
        let m = ys
            .iter()
            .skip(k)
            .step_by(ports)
            .map(|&v| f64::from(v))
            .sum::<f64>()
            / rows as f64;
        let v = ys
            .iter()
            .skip(k)
            .step_by(ports)
            .map(|&y| (f64::from(y) - m).powi(2))
            .sum::<f64>()
            / rows as f64;
        y_mean[k] = m as f32;
        y_scale[k] = (v.sqrt() as f32).max(1e-3);
    }
    for row in ys.chunks_exact_mut(ports) {
        for k in 0..ports {
            row[k] = (row[k] - y_mean[k]) / y_scale[k];
        }
    }
    drop(currents);

    // ---- learn
    let mut sizes = vec![inputs];
    sizes.extend(&args.hidden);
    sizes.push(ports);
    let mut rng = Rng(args.seed ^ 0xABCD);
    let mut net = Net::new(sizes.clone(), &mut rng);
    let mut adam = Adam {
        m: Grads::zeros(&net),
        v: Grads::zeros(&net),
        step: 0,
    };
    // A tenth of the rows are kept back to see that it is not learning them
    // by heart.
    let held = rows / 10;
    let mut order: Vec<usize> = (0..rows - held).collect();
    let held_rows: Vec<usize> = (rows - held..rows).collect();
    println!(
        "network {sizes:?}: {} multiply-adds per sample",
        sizes.windows(2).map(|p| p[0] * p[1]).sum::<usize>()
    );
    let steps_per_epoch = order.len() / args.batch;
    let total = (steps_per_epoch * args.epochs) as f32;
    let mut done = 0f32;
    for epoch in 0..args.epochs {
        // Shuffle.
        for k in (1..order.len()).rev() {
            let j = (rng.next() % (k as u64 + 1)) as usize;
            order.swap(k, j);
        }
        let mut loss = 0.0;
        for step in 0..steps_per_epoch {
            let batch = &order[step * args.batch..(step + 1) * args.batch];
            let chunk = batch.len().div_ceil(threads);
            let scale = 1.0 / (batch.len() * ports) as f32;
            let results: Vec<(f64, Grads)> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .chunks(chunk)
                    .map(|rows| {
                        let (net, xn, ys) = (&net, &xn, &ys);
                        scope.spawn(move || gradient(net, xn, ys, rows, scale))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut total_grads = Grads::zeros(&net);
            for (l, g) in &results {
                loss += l;
                total_grads.add(g);
            }
            // Cosine from the first rate to a twentieth of it.
            let phase = done / total;
            let rate =
                args.rate * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * phase).cos()));
            adam.update(&mut net, &total_grads, rate);
            done += 1.0;
        }
        let (held_loss, _) = {
            let results: Vec<(f64, Grads)> = std::thread::scope(|scope| {
                let handles: Vec<_> = held_rows
                    .chunks(held_rows.len().div_ceil(threads).max(1))
                    .map(|rows| {
                        let (net, xn, ys) = (&net, &xn, &ys);
                        scope.spawn(move || gradient(net, xn, ys, rows, 0.0))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            (results.iter().map(|r| r.0).sum::<f64>(), ())
        };
        println!(
            "epoch {:3}: loss {:.5} held-out {:.5} ({:.0} s)",
            epoch + 1,
            loss / (steps_per_epoch * args.batch * ports) as f64,
            held_loss / (held_rows.len() * ports) as f64,
            began.elapsed().as_secs_f64()
        );
    }

    // ---- export, the standardising of the outputs folded into the last layer
    let mut layers = Vec::new();
    for l in 0..net.layers() {
        let (i, o) = (sizes[l], sizes[l + 1]);
        let mut w = net.w[l].clone();
        let mut b = net.b[l].clone();
        if l == net.layers() - 1 {
            for k in 0..o {
                for j in 0..i {
                    w[k * i + j] *= y_scale[k];
                }
                b[k] = b[k] * y_scale[k] + y_mean[k];
            }
        }
        layers.push((i, o, w, b));
    }
    let surrogate = Surrogate::new(ports, names, true, input_mean, input_scale, output_scale, layers)
        .map_err(|e| e.to_string())?;
    std::fs::write(&args.output, surrogate.to_text())
        .map_err(|e| format!("{}: {e}", args.output.display()))?;
    println!("{}", args.output.display());
    Ok(true)
}

// ------------------------------------------------------------------ eval

pub(crate) fn db(ratio: f64) -> f64 {
    10.0 * ratio.max(1e-300).log10()
}

/// Error-to-signal of the AC part, in dB.
pub(crate) fn esr(reference: &[f32], other: &[f32]) -> f64 {
    let mean = reference.iter().map(|&v| f64::from(v)).sum::<f64>() / reference.len() as f64;
    let (mut err, mut sig) = (0.0, 0.0);
    for (a, b) in reference.iter().zip(other) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        err += (a - b).powi(2);
        sig += (a - mean).powi(2);
    }
    db(err / sig.max(1e-30))
}

fn run(
    base: &Processor,
    solver: Solver,
    setting: &[(usize, f64)],
    input: &[f32],
) -> (Vec<f32>, f64, u64, u64) {
    let mut p = base.clone();
    for &(c, v) in setting {
        p.set_control(c, v);
    }
    p.reset();
    p.set_solver(solver);
    let mut out = vec![0.0f32; input.len()];
    let began = Instant::now();
    p.process(input, &mut out);
    let ns = began.elapsed().as_secs_f64() * 1e9 / input.len() as f64;
    let stats = p.solver_stats();
    (out, ns, stats.iterations, stats.failures)
}

fn eval(args: EvalArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let mut base = build(&circuit, args.sample_rate, &positions(&circuit, &[])?)?;
    let text = std::fs::read_to_string(&args.model)
        .map_err(|e| format!("{}: {e}", args.model.display()))?;
    let surrogate = Surrogate::parse(&text).map_err(|e| e.to_string())?;
    println!(
        "{} multiply-adds per sample, reads: {}",
        surrogate.multiply_adds(),
        surrogate.controls().join(", ")
    );
    base.set_surrogate(surrogate).map_err(|e| e.to_string())?;
    let shaping: Vec<usize> = (0..base.controls().len())
        .filter(|&i| base.control_shapes_nonlinear_solve(i))
        .collect();

    let rate = args.sample_rate;
    let scenarios: [(&str, Kind, f64); 6] = [
        ("soft pluck", Kind::Pluck, 0.08),
        ("hard pluck", Kind::Pluck, args.volts),
        ("chord", Kind::Chord, args.volts * 0.7),
        ("noise", Kind::Noise, args.volts * 0.5),
        ("bright loud", Kind::Sine, args.volts),
        ("sag burst", Kind::Burst, args.volts),
    ];
    let settings: [(&str, f64); 3] = [("default", f64::NAN), ("low", 0.2), ("high", 0.9)];
    println!(
        "\n{:12} {:8} | {:>9} {:>9} {:>9} | {:>8} {:>8} {:>8} | newton steps/sample guided",
        "regime",
        "controls",
        "ESR surr.",
        "ESR guid.",
        "peak err",
        "ns newt.",
        "ns guid.",
        "ns surr."
    );
    let mut worst: f64 = -300.0;
    for (name, kind, volts) in scenarios {
        for (label, value) in settings {
            let setting: Vec<(usize, f64)> = if value.is_nan() {
                Vec::new()
            } else {
                shaping.iter().map(|&c| (c, value)).collect()
            };
            let mut rng = Rng(0xE7A1 + (volts * 1000.0) as u64);
            let mut input = excitation(kind, &mut rng, rate, 2.0, volts);
            if name == "bright loud" {
                // A fixed 3 kHz tone is the hardest thing a clamp is asked.
                for (i, x) in input.iter_mut().enumerate() {
                    *x = (volts * (std::f64::consts::TAU * 3000.0 * i as f64 / rate).sin()) as f32
                        * if i < (rate * 1.5) as usize { 1.0 } else { 0.0 };
                }
            }
            let (reference, ns_n, it_n, f_n) = run(&base, Solver::Newton, &setting, &input);
            let (guided, ns_g, it_g, f_g) = run(&base, Solver::Guided, &setting, &input);
            let (surr, ns_s, _, _) = run(&base, Solver::Surrogate, &setting, &input);
            let peak = reference
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-12);
            let peak_err = surr
                .iter()
                .zip(&reference)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / peak;
            let finite = surr.iter().all(|v| v.is_finite());
            let e_s = if finite {
                esr(&reference, &surr)
            } else {
                f64::INFINITY
            };
            let e_g = esr(&reference, &guided);
            if finite {
                worst = worst.max(e_s);
            }
            println!(
                "{:12} {:8} | {:>8.1}dB {:>8.1}dB {:>8.2}% | {:>8.0} {:>8.0} {:>8.0} | {:.2} -> {:.2}{}",
                name,
                label,
                e_s,
                e_g,
                100.0 * peak_err,
                ns_n,
                ns_g,
                ns_s,
                it_n as f64 / input.len() as f64,
                it_g as f64 / input.len() as f64,
                if f_n + f_g > 0 { format!("  (unconverged: newton {f_n}, guided {f_g})") } else { String::new() },
            );
        }
    }
    println!("\nworst surrogate ESR: {worst:.1} dB (0 dB is as large as the signal)");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loss of one row, slopes included, in f64 by finite differences
    /// of the network's own forward pass.
    fn loss(net: &Net, x: &[f32], y: &[f32], slopes: &[f32], weight: f32) -> f64 {
        let n = net.sizes[0];
        let m = *net.sizes.last().unwrap();
        let mut z: Vec<Vec<f32>> = net.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
        let mut a: Vec<Vec<f32>> = net.sizes.iter().map(|&s| vec![0.0; s]).collect();
        net.forward(x, &mut z, &mut a);
        let out = a.last().unwrap().clone();
        let mut total = 0.0f64;
        for c in 0..m {
            total += f64::from((out[c] - y[c]).powi(2));
        }
        let h = 1e-2f32;
        for j in 0..n {
            let mut up = x.to_vec();
            let mut down = x.to_vec();
            up[j] += h;
            down[j] -= h;
            net.forward(&up, &mut z, &mut a);
            let hi = a.last().unwrap().clone();
            net.forward(&down, &mut z, &mut a);
            let lo = a.last().unwrap().clone();
            for c in 0..m {
                let slope = (hi[c] - lo[c]) / (2.0 * h);
                let t = slopes[c * n + j];
                total += f64::from(weight / (1.0 + t * t) * (slope - t).powi(2));
            }
        }
        total
    }

    #[test]
    fn the_sobolev_gradient_is_the_slope_of_its_loss() {
        let mut rng = Rng(7);
        let mut net = Net::new(vec![3, 5, 4, 3], &mut rng);
        // Biases away from zero, so that every term of the activation counts.
        for b in &mut net.b {
            b.iter_mut().for_each(|b| *b = rng.between(-0.8, 0.8) as f32);
        }
        let x = [0.3f32, -0.7, 0.5];
        let y = [0.1f32, -0.4, 0.9];
        let slopes = [0.5f32, -1.0, 0.2, 3.0, 0.0, -0.3, 0.7, 0.1, -2.0];
        let weight = 0.8;
        let (v, s, g) = sobolev_gradient(&net, &x, &y, &slopes, weight, &[0], 1.0);
        let analytic = loss(&net, &x, &y, &slopes, weight);
        // The value is the same loss (its slopes exact, not differenced).
        assert!((v + s - analytic).abs() < 1e-3 * analytic.max(1.0), "{} {}", v + s, analytic);
        let h = 1e-3f32;
        let mut worst = 0.0f64;
        for l in 0..net.layers() {
            for q in 0..net.w[l].len() {
                let keep = net.w[l][q];
                net.w[l][q] = keep + h;
                let up = sobolev_gradient(&net, &x, &y, &slopes, weight, &[0], 1.0);
                net.w[l][q] = keep - h;
                let down = sobolev_gradient(&net, &x, &y, &slopes, weight, &[0], 1.0);
                net.w[l][q] = keep;
                let numeric = (up.0 + up.1 - down.0 - down.1) / (2.0 * f64::from(h));
                let e = (numeric - f64::from(g.w[l][q])).abs() / numeric.abs().max(0.05);
                worst = worst.max(e);
            }
            for q in 0..net.b[l].len() {
                let keep = net.b[l][q];
                net.b[l][q] = keep + h;
                let up = sobolev_gradient(&net, &x, &y, &slopes, weight, &[0], 1.0);
                net.b[l][q] = keep - h;
                let down = sobolev_gradient(&net, &x, &y, &slopes, weight, &[0], 1.0);
                net.b[l][q] = keep;
                let numeric = (up.0 + up.1 - down.0 - down.1) / (2.0 * f64::from(h));
                let e = (numeric - f64::from(g.b[l][q])).abs() / numeric.abs().max(0.05);
                worst = worst.max(e);
            }
        }
        assert!(worst < 2e-2, "worst relative error of the gradient {worst}");
    }

    #[test]
    fn the_activations_curve_is_the_slope_of_its_slope() {
        for x in [-2.9f32, -1.5, -0.2, 0.0, 0.4, 1.1, 2.7] {
            let h = 1e-3;
            let numeric = (activation_slope(x + h) - activation_slope(x - h)) / (2.0 * h);
            assert!((numeric - activation_curve(x)).abs() < 2e-3, "{x}: {numeric} {}", activation_curve(x));
        }
    }
}
