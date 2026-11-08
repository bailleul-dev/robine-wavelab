//! A small network for each component's law: training it against the law
//! (or against measurements), and judging it in the circuit.
//!
//! A circuit's coupling (the supply that sags, the stage that loads the one
//! before) lives in the linear network, which Newton solves exactly. What a
//! network learns here is only one component's curve of two or three
//! variables, so it can be sampled everywhere and checked everywhere, and
//! the sag is still the circuit's.
//!
//! `micro-train` plays the circuit with its own laws to find the box of
//! voltages each chosen component visits, then samples the law in that box,
//! densely along the trajectory the circuit follows and uniformly over the
//! rest of the box, and learns it. With `--data` the targets are measured
//! currents instead of the law: the way to give a component that has no
//! formula, or one too dear to evaluate.
//!
//! `micro-eval` runs the circuit with the networks in the loop against the
//! analytic laws, in the regimes where the circuit is worth simulating.

use crate::commands::{build, load_with, positions, CmdResult};
use crate::nn::{db, esr, excitation, gradient, pick, threads, Adam, Grads, Kind, Net, Rng};
use clap::Args;
use robine_spice::{write_set, MicroNet, Processor, MAX_WIDTH};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args)]
pub struct MicroTrainArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// Where the set of networks goes.
    #[arg(short, long)]
    output: PathBuf,
    /// Devices to learn, by number (see `micro-list`); every device when
    /// not given.
    #[arg(long, value_delimiter = ',')]
    devices: Vec<usize>,
    /// Only the devices of these kinds (`triode`, `pentode`, `diode`, ...).
    #[arg(long, value_delimiter = ',')]
    kinds: Vec<String>,
    /// Measured currents instead of the law, for one device: a CSV with a
    /// row per measurement, the port voltages then the port currents. Its
    /// range becomes the network's box.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Sizes of the hidden layers, comma separated (at most 16 wide).
    #[arg(long, value_delimiter = ',', default_value = "16,16")]
    hidden: Vec<usize>,
    /// Samples of the circuit played to find where its devices work.
    #[arg(long, default_value_t = 600_000)]
    samples: usize,
    /// Training rows per device, half from the trajectory and half over
    /// the box.
    #[arg(long, default_value_t = 400_000)]
    rows: usize,
    /// Passes over the rows.
    #[arg(long, default_value_t = 60)]
    epochs: usize,
    /// Rows in a step.
    #[arg(long, default_value_t = 1024)]
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
pub struct MicroEvalArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
    /// The set of networks.
    #[arg(short, long)]
    model: PathBuf,
    /// Sample rate.
    #[arg(long, default_value_t = 48_000.0)]
    sample_rate: f64,
    /// Largest volts at the circuit input.
    #[arg(long, default_value_t = 2.0)]
    volts: f64,
}

#[derive(Args)]
pub struct MicroListArgs {
    /// Netlist.
    #[arg(short, long)]
    circuit: PathBuf,
    /// What drives the input (as for `compare`).
    #[arg(long, default_value = "ideal")]
    source: String,
}

/// The device numbers, kinds and ports of a circuit.
pub fn list(args: MicroListArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let p = build(&circuit, 48_000.0, &positions(&circuit, &[])?)?;
    for k in 0..p.device_count() {
        println!(
            "{k:3}  {:10} {} port(s) from port {}",
            p.device_kind(k),
            p.device_ports(k),
            p.device_first_port(k)
        );
    }
    Ok(true)
}

/// The port voltages of the chosen devices, sample after sample, while the
/// circuit is played with its own laws and its controls moved at random.
fn trajectory(
    template: &Processor,
    devices: &[usize],
    samples: usize,
    rate: f64,
    volts: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut processor = template.clone();
    let mut rng = Rng(seed);
    let mut out: Vec<Vec<f64>> = devices.iter().map(|_| Vec::new()).collect();
    let controls = processor.controls().len();
    let mut seen = 0;
    while seen < samples {
        for c in 0..controls {
            let v = if rng.uniform() < 0.15 {
                f64::from(u8::from(rng.uniform() < 0.5))
            } else {
                rng.uniform()
            };
            processor.set_control(c, v);
        }
        let seconds = rng.between(0.4, 2.0);
        let signal = excitation(pick(&mut rng), &mut rng, rate, seconds, volts);
        for &x in &signal {
            processor.process_sample(x);
            for (slot, &k) in out.iter_mut().zip(devices) {
                let at = processor.device_first_port(k);
                slot.extend_from_slice(
                    &processor.port_voltages()[at..at + processor.device_ports(k)],
                );
            }
        }
        seen += signal.len();
        if processor.solver_stats().failures > 0 {
            processor.reset();
        }
    }
    out
}

/// Rows of `ports` voltages, and the currents the targets give there.
struct Table {
    v: Vec<f64>,
    i: Vec<f64>,
}

fn read_csv(path: &PathBuf, ports: usize) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut table = Table {
        v: Vec::new(),
        i: Vec::new(),
    };
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cells: Result<Vec<f64>, _> = line
            .split([',', ';', '\t', ' '])
            .filter(|c| !c.is_empty())
            .map(str::parse::<f64>)
            .collect();
        let Ok(cells) = cells else {
            if n == 0 {
                continue; // a header
            }
            return Err(format!("{}: line {}: not numbers", path.display(), n + 1));
        };
        if cells.len() != 2 * ports {
            return Err(format!(
                "{}: line {}: {} values, expected {} voltages then {} currents",
                path.display(),
                n + 1,
                cells.len(),
                ports,
                ports
            ));
        }
        table.v.extend_from_slice(&cells[..ports]);
        table.i.extend_from_slice(&cells[ports..]);
    }
    if table.v.is_empty() {
        return Err(format!("{}: no measurements", path.display()));
    }
    Ok(table)
}

pub fn train(args: MicroTrainArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let template = build(&circuit, args.sample_rate, &positions(&circuit, &[])?)?;
    if args.hidden.is_empty() || args.hidden.iter().any(|&h| h == 0 || h > MAX_WIDTH) {
        return Err(format!("hidden layers are 1 to {MAX_WIDTH} wide").into());
    }
    let mut devices: Vec<usize> = if args.devices.is_empty() {
        (0..template.device_count()).collect()
    } else {
        args.devices.clone()
    };
    if let Some(&bad) = devices.iter().find(|&&k| k >= template.device_count()) {
        return Err(format!("the circuit has no device {bad}").into());
    }
    if !args.kinds.is_empty() {
        devices.retain(|&k| args.kinds.iter().any(|w| w == template.device_kind(k)));
    }
    if devices.is_empty() {
        return Err("no device to learn".into());
    }
    if args.data.is_some() && devices.len() != 1 {
        return Err("--data is for one device: pick it with --devices".into());
    }
    let threads = threads(args.threads);
    let began = Instant::now();
    println!(
        "{}: learning {} device(s), hidden {:?}",
        args.circuit.display(),
        devices.len(),
        args.hidden
    );

    // ---- where the devices work
    let walked: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (template, devices) = (&template, &devices);
                let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(t as u64);
                let per = args.samples.div_ceil(threads);
                let (rate, volts) = (args.sample_rate, args.volts);
                scope.spawn(move || trajectory(template, devices, per, rate, volts, seed))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!(
        "played the circuit in {:.1} s",
        began.elapsed().as_secs_f64()
    );

    let mut nets = Vec::new();
    for (n, &k) in devices.iter().enumerate() {
        let ports = template.device_ports(k);
        let kind = template.device_kind(k);
        let mut rng = Rng(args.seed ^ (0xABCD + k as u64));
        let mut visited: Vec<f64> = Vec::new();
        for part in &walked {
            visited.extend_from_slice(&part[n]);
        }
        // ---- the table to learn: law or measurements
        let (table, low, high) = if let Some(path) = &args.data {
            let t = read_csv(path, ports)?;
            let (mut low, mut high) = (vec![f64::MAX; ports], vec![f64::MIN; ports]);
            for row in t.v.chunks_exact(ports) {
                for c in 0..ports {
                    low[c] = low[c].min(row[c]);
                    high[c] = high[c].max(row[c]);
                }
            }
            (t, low, high)
        } else {
            let (mut low, mut high) = (vec![f64::MAX; ports], vec![f64::MIN; ports]);
            for row in visited.chunks_exact(ports) {
                for c in 0..ports {
                    low[c] = low[c].min(row[c]);
                    high[c] = high[c].max(row[c]);
                }
            }
            for c in 0..ports {
                let pad = 0.15 * (high[c] - low[c]) + 1e-3;
                low[c] -= pad;
                high[c] += pad;
            }
            let seen = visited.len() / ports;
            let half = args.rows / 2;
            let mut t = Table {
                v: Vec::with_capacity(args.rows * ports),
                i: Vec::with_capacity(args.rows * ports),
            };
            for _ in 0..half {
                let r = (rng.next() % seen as u64) as usize;
                t.v.extend_from_slice(&visited[r * ports..(r + 1) * ports]);
            }
            for _ in 0..args.rows - half {
                for c in 0..ports {
                    t.v.push(rng.between(low[c], high[c]));
                }
            }
            let mut i = vec![0.0; ports];
            for row in t.v.chunks_exact(ports) {
                template.device_law(k, row, &mut i);
                t.i.extend_from_slice(&i);
            }
            (t, low, high)
        };
        let rows = table.v.len() / ports;
        println!(
            "device {k} ({kind}, {ports} port(s)): {rows} rows, box {}",
            (0..ports)
                .map(|c| format!("[{:.3}, {:.3}]", low[c], high[c]))
                .collect::<Vec<_>>()
                .join(" x ")
        );

        // ---- normalise
        let mut mean = vec![0.0f64; ports];
        for row in table.v.chunks_exact(ports) {
            for c in 0..ports {
                mean[c] += row[c] / rows as f64;
            }
        }
        let mut var = vec![0.0f64; ports];
        for row in table.v.chunks_exact(ports) {
            for c in 0..ports {
                var[c] += (row[c] - mean[c]).powi(2) / rows as f64;
            }
        }
        let input_mean: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        let input_scale: Vec<f32> = var.iter().map(|&v| (v.sqrt() as f32).max(1e-3)).collect();
        let mut output_scale = vec![0.0f32; ports];
        for (c, s) in output_scale.iter_mut().enumerate() {
            let mut magnitudes: Vec<f64> = table
                .i
                .iter()
                .skip(c)
                .step_by(ports)
                .map(|v| v.abs())
                .collect();
            magnitudes.sort_by(f64::total_cmp);
            *s = (magnitudes[magnitudes.len() / 2] as f32).max(1e-18);
        }
        let xn: Vec<f32> = table
            .v
            .chunks_exact(ports)
            .flat_map(|row| {
                (0..ports)
                    .map(|c| (row[c] as f32 - input_mean[c]) / input_scale[c])
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut ys: Vec<f32> = table
            .i
            .chunks_exact(ports)
            .flat_map(|row| {
                (0..ports)
                    .map(|c| (row[c] / f64::from(output_scale[c])).asinh() as f32)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut y_mean = vec![0.0f32; ports];
        let mut y_scale = vec![0.0f32; ports];
        for c in 0..ports {
            let m = ys
                .iter()
                .skip(c)
                .step_by(ports)
                .map(|&y| f64::from(y))
                .sum::<f64>()
                / rows as f64;
            let v = ys
                .iter()
                .skip(c)
                .step_by(ports)
                .map(|&y| (f64::from(y) - m).powi(2))
                .sum::<f64>()
                / rows as f64;
            y_mean[c] = m as f32;
            y_scale[c] = (v.sqrt() as f32).max(1e-3);
        }
        for row in ys.chunks_exact_mut(ports) {
            for c in 0..ports {
                row[c] = (row[c] - y_mean[c]) / y_scale[c];
            }
        }

        // ---- learn
        let mut sizes = vec![ports];
        sizes.extend(&args.hidden);
        sizes.push(ports);
        let mut net = Net::new(sizes.clone(), &mut rng);
        let mut adam = Adam {
            m: Grads::zeros(&net),
            v: Grads::zeros(&net),
            step: 0,
        };
        let held = rows / 10;
        let mut order: Vec<usize> = (0..rows - held).collect();
        let held_rows: Vec<usize> = (rows - held..rows).collect();
        let steps_per_epoch = (order.len() / args.batch).max(1);
        let total = (steps_per_epoch * args.epochs) as f32;
        let mut done = 0f32;
        let mut last = (0.0, 0.0);
        for epoch in 0..args.epochs {
            for q in (1..order.len()).rev() {
                let j = (rng.next() % (q as u64 + 1)) as usize;
                order.swap(q, j);
            }
            let mut loss = 0.0;
            for step in 0..steps_per_epoch {
                let batch = &order[step * args.batch..((step + 1) * args.batch).min(order.len())];
                let chunk = batch.len().div_ceil(threads);
                let scale = 1.0 / (batch.len() * ports) as f32;
                let results: Vec<(f64, Grads)> = std::thread::scope(|scope| {
                    let handles: Vec<_> = batch
                        .chunks(chunk)
                        .map(|part| {
                            let (net, xn, ys) = (&net, &xn, &ys);
                            scope.spawn(move || gradient(net, xn, ys, part, scale))
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let mut sum = Grads::zeros(&net);
                for (l, g) in &results {
                    loss += l;
                    sum.add(g);
                }
                let phase = done / total;
                let rate =
                    args.rate * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * phase).cos()));
                adam.update(&mut net, &sum, rate);
                done += 1.0;
            }
            if epoch + 1 == args.epochs || (epoch + 1) % 10 == 0 {
                let (held_loss, _) = gradient(&net, &xn, &ys, &held_rows, 0.0);
                last = (
                    loss / (steps_per_epoch * args.batch * ports) as f64,
                    held_loss / (held_rows.len() * ports) as f64,
                );
                println!(
                    "  epoch {:3}: loss {:.6} held-out {:.6} ({:.0} s)",
                    epoch + 1,
                    last.0,
                    last.1,
                    began.elapsed().as_secs_f64()
                );
            }
        }
        let _ = last;

        // ---- export, the standardising folded into the last layer
        let mut layers = Vec::new();
        for l in 0..net.layers() {
            let (i, o) = (sizes[l], sizes[l + 1]);
            let mut w = net.w[l].clone();
            let mut b = net.b[l].clone();
            if l == net.layers() - 1 {
                for c in 0..o {
                    for j in 0..i {
                        w[c * i + j] *= y_scale[c];
                    }
                    b[c] = b[c] * y_scale[c] + y_mean[c];
                }
            }
            layers.push((i, o, w, b));
        }
        let micro = MicroNet::new(
            format!("{kind} #{k}"),
            ports,
            input_mean,
            input_scale,
            output_scale,
            (low, high),
            layers,
        )
        .map_err(|e| e.to_string())?;

        // ---- how far it is from the target on the points held back
        let mut errors: Vec<f64> = Vec::new();
        let biggest: Vec<f64> = (0..ports)
            .map(|c| {
                table
                    .i
                    .iter()
                    .skip(c)
                    .step_by(ports)
                    .fold(0.0f64, |m, v| m.max(v.abs()))
            })
            .collect();
        let mut got = vec![0.0; ports];
        let mut j = [0.0; 9];
        for r in rows - held..rows {
            micro.eval(&table.v[r * ports..(r + 1) * ports], &mut got, &mut j);
            for c in 0..ports {
                let want = table.i[r * ports + c];
                errors.push((got[c] - want).abs() / (want.abs() + 0.01 * biggest[c]).max(1e-30));
            }
        }
        errors.sort_by(f64::total_cmp);
        let at = |q: f64| errors[((errors.len() - 1) as f64 * q) as usize];
        println!(
            "  error against the target, relative to the current or 1% of its largest: median {:.2e}, 99% {:.2e}, worst {:.2e}",
            at(0.5), at(0.99), at(1.0)
        );
        nets.push(micro);
    }
    std::fs::write(&args.output, write_set(&nets))
        .map_err(|e| format!("{}: {e}", args.output.display()))?;
    println!("{} ({} network(s))", args.output.display(), nets.len());
    Ok(true)
}

/// Which device a network is for, from its signature `kind #k`.
fn device_of(net: &MicroNet) -> Option<usize> {
    net.signature().rsplit('#').next()?.trim().parse().ok()
}

fn run(
    base: &Processor,
    micro: bool,
    setting: &[(usize, f64)],
    input: &[f32],
) -> (Vec<f32>, f64, u64, u64, u64) {
    let mut p = base.clone();
    for &(c, v) in setting {
        p.set_control(c, v);
    }
    for k in 0..p.device_count() {
        p.set_micro_active(k, micro);
    }
    p.reset();
    let mut out = vec![0.0f32; input.len()];
    let began = Instant::now();
    p.process(input, &mut out);
    let ns = began.elapsed().as_secs_f64() * 1e9 / input.len() as f64;
    let stats = p.solver_stats();
    (out, ns, stats.iterations, stats.failures, p.micro_outside())
}

pub fn eval(args: MicroEvalArgs) -> CmdResult {
    let circuit = load_with(&args.circuit, &args.source)?;
    let mut base = build(&circuit, args.sample_rate, &positions(&circuit, &[])?)?;
    let text = std::fs::read_to_string(&args.model)
        .map_err(|e| format!("{}: {e}", args.model.display()))?;
    let nets = MicroNet::parse_set(&text).map_err(|e| e.to_string())?;
    let mut multiply_adds = 0;
    let mut replaced = Vec::new();
    for net in nets {
        let k = device_of(&net).ok_or("a network's signature has no `#device`")?;
        multiply_adds += net.multiply_adds();
        base.set_micro(k, net).map_err(|e| e.to_string())?;
        replaced.push(k);
    }
    println!(
        "{} of {} devices replaced, {multiply_adds} multiply-adds per evaluation of them all",
        replaced.len(),
        base.device_count()
    );
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
        "\n{:12} {:8} | {:>9} {:>9} | {:>8} {:>8} | {:>8} {:>10}",
        "regime", "controls", "ESR", "peak err", "ns exact", "ns nets", "steps", "outside box"
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
                for (i, x) in input.iter_mut().enumerate() {
                    *x = (volts * (std::f64::consts::TAU * 3000.0 * i as f64 / rate).sin()) as f32
                        * if i < (rate * 1.5) as usize { 1.0 } else { 0.0 };
                }
            }
            let (reference, ns_e, it_e, f_e, _) = run(&base, false, &setting, &input);
            let (nets_out, ns_n, it_n, f_n, outside) = run(&base, true, &setting, &input);
            let peak = reference
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-12);
            let peak_err = nets_out
                .iter()
                .zip(&reference)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / peak;
            let finite = nets_out.iter().all(|v| v.is_finite());
            let e = if finite {
                esr(&reference, &nets_out)
            } else {
                f64::INFINITY
            };
            if finite {
                worst = worst.max(e);
            } else {
                worst = f64::INFINITY;
            }
            println!(
                "{:12} {:8} | {:>7.1}dB {:>8.3}% | {:>8.0} {:>8.0} | {:.2}->{:.2} {:>10}{}",
                name,
                label,
                e,
                100.0 * peak_err,
                ns_e,
                ns_n,
                it_e as f64 / input.len() as f64,
                it_n as f64 / input.len() as f64,
                outside,
                if f_e + f_n > 0 {
                    format!("  (unconverged: exact {f_e}, nets {f_n})")
                } else {
                    String::new()
                }
            );
        }
    }
    println!(
        "\nworst ESR with the networks in the loop: {worst:.1} dB ({:.1} dB is a clean tube amp's noise floor, 0 dB is as large as the signal)",
        db(1e-6)
    );
    Ok(true)
}
