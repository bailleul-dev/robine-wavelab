//! LiveSPICE schematic (`.schx`) to netlist.
//!
//! A schematic is a drawing: symbols at grid positions and wire segments.
//! Connectivity follows LiveSPICE's own rule: a terminal belongs to the net
//! of a wire it lies on, and two wires share a net when an end of one lies
//! on the other. Terminals that merely touch each other are not connected.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

type Point = (i32, i32);

struct Symbol {
    kind: String,
    attrs: HashMap<String, String>,
    position: Point,
    rotation: i32,
    flip: bool,
}

impl Symbol {
    fn name(&self) -> &str {
        self.attrs.get("Name").map_or("", String::as_str)
    }

    fn attr(&self, key: &str) -> Result<&str, String> {
        self.attrs
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("{} `{}` has no {key}", self.kind, self.name()))
    }

    fn quantity(&self, key: &str) -> Result<f64, String> {
        let text = self.attr(key)?;
        parse_quantity(text).ok_or_else(|| {
            format!(
                "{} `{}`: cannot read {key} `{text}`",
                self.kind,
                self.name()
            )
        })
    }

    /// LiveSPICE's `Symbol.MapToGlobal`.
    fn global(&self, local: Point) -> Point {
        let x = local.0;
        let y = if self.flip { local.1 } else { -local.1 };
        let cos = quarter_cos(self.rotation);
        let sin = quarter_cos(self.rotation - 1);
        (
            x * cos + y * sin + self.position.0,
            y * cos - x * sin + self.position.1,
        )
    }
}

fn quarter_cos(quarter_turns: i32) -> i32 {
    [1, 0, -1, 0][quarter_turns.rem_euclid(4) as usize]
}

/// Terminal names and local positions, from each component's
/// `LayoutSymbol` in LiveSPICE.
fn terminals(s: &Symbol) -> Result<Vec<(&'static str, Point)>, String> {
    Ok(match s.kind.as_str() {
        "Label" => vec![],
        "Ground" | "Rail" | "NamedWire" => vec![("T", (0, 0))],
        "Resistor" | "Capacitor" | "Inductor" | "Diode" | "Input" | "Speaker" | "VoltageSource"
        | "VariableResistor" => vec![("A", (0, 20)), ("C", (0, -20))],
        "Potentiometer" => vec![("A", (-10, 20)), ("C", (-10, -20)), ("W", (10, 0))],
        "OpAmp" => vec![
            ("+", (-20, -10)),
            ("-", (-20, 10)),
            ("Out", (20, 0)),
            ("Vcc", (0, 20)),
            ("Vee", (0, -20)),
        ],
        "IdealOpAmp" => vec![("+", (-20, -10)), ("-", (-20, 10)), ("Out", (20, 0))],
        "JunctionFieldEffectTransistor" => {
            vec![("D", (10, 20)), ("G", (-20, 0)), ("S", (10, -20))]
        }
        "Triode" => vec![("P", (0, 20)), ("G", (-20, 0)), ("K", (-10, -20))],
        "Pentode" => vec![
            ("P", (0, 25)),
            ("G", (-20, -5)),
            ("G2", (20, 0)),
            ("K", (-10, -25)),
        ],
        "Transformer" => vec![
            ("PA", (-10, 20)),
            ("PC", (-10, -20)),
            ("SA", (10, 20)),
            ("SC", (10, -20)),
        ],
        "CenterTapTransformer" => vec![
            ("PA", (-10, 20)),
            ("PC", (-10, -20)),
            ("SA", (10, 20)),
            ("ST", (10, 0)),
            ("SC", (10, -20)),
        ],
        // Common, then the throws from left to right.
        "SPDT" => vec![("C", (0, -20)), ("T0", (-10, 20)), ("T1", (10, 20))],
        "SP3T" => vec![
            ("C", (0, -20)),
            ("T0", (-20, 20)),
            ("T1", (0, 20)),
            ("T2", (20, 20)),
        ],
        "BipolarJunctionTransistor" => match s.attr("Type")? {
            "NPN" => vec![("C", (10, 20)), ("B", (-20, 0)), ("E", (10, -20))],
            "PNP" => vec![("C", (10, -20)), ("B", (-20, 0)), ("E", (10, 20))],
            other => return Err(format!("unknown BJT type `{other}`")),
        },
        other => {
            return Err(format!(
                "component `{other}` (`{}`) is not supported by the converter yet",
                s.name()
            ))
        }
    })
}

/// `"10 kΩ"`, `"200 nA"`, `"200 k"`, `"1 MHz"`, `"∞ Ω"`.
fn parse_quantity(text: &str) -> Option<f64> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(text.len());
    let (number, mut rest) = (text[..split].trim(), text[split..].trim());
    if number.is_empty() {
        return rest.starts_with('∞').then_some(f64::INFINITY);
    }
    // LiveSPICE writes the ohm sign (U+2126); accept the Greek omega too.
    for unit in ["Hz", "\u{2126}", "\u{3a9}", "Ohm", "F", "H", "V", "A"] {
        if let Some(stripped) = rest.strip_suffix(unit) {
            rest = stripped.trim();
            break;
        }
    }
    let exponent = match rest {
        "" => 0,
        "f" => -15,
        "p" => -12,
        "n" => -9,
        "μ" | "µ" | "u" => -6,
        "m" => -3,
        "k" => 3,
        "M" => 6,
        "G" => 9,
        "T" => 12,
        _ => return None,
    };
    // Let the decimal parser apply the prefix when it can, so that "10 k"
    // is exactly 1e4; a number with its own exponent is scaled instead.
    if number.contains(['e', 'E']) {
        Some(number.parse::<f64>().ok()? * 10f64.powi(exponent))
    } else {
        format!("{number}e{exponent}").parse().ok()
    }
}

fn parse_point(text: &str) -> Result<Point, String> {
    let (x, y) = text
        .split_once(',')
        .ok_or_else(|| format!("bad coordinate `{text}`"))?;
    let parse = |v: &str| {
        v.trim()
            .parse::<i32>()
            .map_err(|_| format!("bad coordinate `{text}`"))
    };
    Ok((parse(x)?, parse(y)?))
}

fn on_segment(p: Point, a: Point, b: Point) -> bool {
    let within = |v: i32, lo: i32, hi: i32| v >= lo.min(hi) && v <= lo.max(hi);
    within(p.0, a.0, b.0)
        && within(p.1, a.1, b.1)
        && (b.0 - a.0) * (p.1 - a.1) == (p.0 - a.0) * (b.1 - a.1)
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Characters a netlist name cannot carry become underscores.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `name` if it already starts with `letter`, otherwise prefixed with it.
fn element_name(letter: char, name: &str) -> String {
    let clean = sanitize(name);
    if clean.chars().next().map(|c| c.to_ascii_uppercase()) == Some(letter) {
        clean
    } else {
        format!("{letter}{clean}")
    }
}

fn model_name(prefix: &str, part: Option<&String>, index: usize) -> String {
    match part.map(|p| p.trim()).filter(|p| !p.is_empty()) {
        Some(part) => format!(
            "{prefix}_{}",
            part.chars()
                .map(|c| if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                })
                .collect::<String>()
        ),
        None => format!("{prefix}_{index}"),
    }
}

/// Convert a schematic to a robine-spice netlist. `source` is only used in
/// the header comment.
pub fn convert(xml: &str, source: &str) -> Result<String, String> {
    let xml = xml.strip_prefix('\u{feff}').unwrap_or(xml);
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("{source}: {e}"))?;

    let mut wires: Vec<(Point, Point)> = Vec::new();
    let mut symbols: Vec<Symbol> = Vec::new();
    for element in doc
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("Element"))
    {
        let class = element.attribute("Type").unwrap_or("");
        if class.starts_with("Circuit.Wire") {
            let a = parse_point(element.attribute("A").unwrap_or(""))?;
            let b = parse_point(element.attribute("B").unwrap_or(""))?;
            wires.push((a, b));
        } else if class.starts_with("Circuit.Symbol") {
            let component = element
                .children()
                .find(|n| n.has_tag_name("Component"))
                .ok_or("symbol without a component")?;
            let full = component.attribute("_Type").unwrap_or("");
            let kind = full
                .split(',')
                .next()
                .unwrap_or("")
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_string();
            symbols.push(Symbol {
                kind,
                attrs: component
                    .attributes()
                    .map(|a| (a.name().to_string(), a.value().to_string()))
                    .collect(),
                position: parse_point(element.attribute("Position").unwrap_or(""))?,
                rotation: element
                    .attribute("Rotation")
                    .and_then(|r| r.parse().ok())
                    .ok_or("symbol without a rotation")?,
                flip: element.attribute("Flip") == Some("true"),
            });
        }
    }

    // Wires that touch form one net.
    let mut parent: Vec<usize> = (0..wires.len()).collect();
    for i in 0..wires.len() {
        for j in 0..i {
            let ((a, b), (c, d)) = (wires[i], wires[j]);
            if on_segment(a, c, d)
                || on_segment(b, c, d)
                || on_segment(c, a, b)
                || on_segment(d, a, b)
            {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                parent[ri] = rj;
            }
        }
    }
    // Named wires with the same name join their nets.
    let wire_at = |p: Point, parent: &mut Vec<usize>| -> Option<usize> {
        let w = wires.iter().position(|&(a, b)| on_segment(p, a, b))?;
        Some(find(parent, w))
    };
    let mut named: HashMap<String, usize> = HashMap::new();
    for s in symbols.iter().filter(|s| s.kind == "NamedWire") {
        if let Some(net) = wire_at(s.global((0, 0)), &mut parent) {
            match named.get(s.attr("WireName")?) {
                Some(&other) => {
                    let other = find(&mut parent, other);
                    parent[net] = other;
                }
                None => {
                    named.insert(s.attr("WireName")?.to_string(), net);
                }
            }
        }
    }

    // A switch joins its common terminal to the selected throw.
    for s in symbols
        .iter()
        .filter(|s| matches!(s.kind.as_str(), "SPDT" | "SP3T"))
    {
        let position: usize = s
            .attrs
            .get("Position")
            .map_or(Ok(0), |p| p.parse())
            .map_err(|_| format!("switch `{}`: bad Position", s.name()))?;
        let pins = terminals(s)?;
        let selected = pins
            .get(1 + position)
            .ok_or_else(|| format!("switch `{}` has no throw {position}", s.name()))?;
        let common = wire_at(s.global(pins[0].1), &mut parent);
        let throw = wire_at(s.global(selected.1), &mut parent);
        if let (Some(a), Some(b)) = (common, throw) {
            parent[a] = b;
        }
    }

    // Net of every terminal: a wire net, or nothing if it hangs free.
    let mut pins: Vec<Vec<Option<usize>>> = Vec::new();
    for s in &symbols {
        let mut mine = Vec::new();
        for (_, local) in terminals(s)? {
            mine.push(wire_at(s.global(local), &mut parent));
        }
        pins.push(mine);
    }

    // Names: ground, named wires, input and output, then n1, n2...
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    for (s, p) in symbols.iter().zip(&pins) {
        if s.kind == "Ground" {
            if let Some(net) = p[0] {
                names.insert(net, "0".to_string());
            }
        }
    }
    for (name, net) in &named {
        let net = find(&mut parent, *net);
        names.entry(net).or_insert_with(|| sanitize(name));
    }
    for (kind, name) in [("Input", "in"), ("Speaker", "out")] {
        let mut of_kind = symbols.iter().zip(&pins).filter(|(s, _)| s.kind == kind);
        if let (Some((_, p)), None) = (of_kind.next(), of_kind.next()) {
            if let Some(net) = p[0] {
                names.entry(net).or_insert_with(|| name.to_string());
            }
        }
    }
    let mut counter = 0;
    let mut floating = 0;
    let mut net_name = |net: Option<usize>, names: &mut BTreeMap<usize, String>| -> String {
        match net {
            Some(net) => names
                .entry(net)
                .or_insert_with(|| {
                    counter += 1;
                    format!("n{counter}")
                })
                .clone(),
            None => {
                floating += 1;
                format!("open{floating}")
            }
        }
    };

    let mut title = String::new();
    let mut body = String::new();
    let mut tail = String::new();
    let mut models: BTreeMap<String, String> = BTreeMap::new();
    let mut subckts: BTreeMap<String, String> = BTreeMap::new();
    let mut inputs = 0;
    let mut outputs = 0;
    let w = &mut body;
    for (index, (s, p)) in symbols.iter().zip(&pins).enumerate() {
        let mut node = |i: usize| net_name(p[i], &mut names);
        match s.kind.as_str() {
            "Ground" | "NamedWire" => {}
            "Label" => {
                if title.is_empty() {
                    title = s.attrs.get("Text").cloned().unwrap_or_default();
                }
            }
            "Resistor" | "Capacitor" | "Inductor" => {
                let (letter, key) = match s.kind.as_str() {
                    "Resistor" => ('R', "Resistance"),
                    "Capacitor" => ('C', "Capacitance"),
                    _ => ('L', "Inductance"),
                };
                let value = s.quantity(key)?;
                let _ = writeln!(
                    w,
                    "{} {} {} {value:e}",
                    element_name(letter, s.name()),
                    node(0),
                    node(1)
                );
            }
            "Input" => {
                inputs += 1;
                let name = element_name('V', s.name());
                let _ = writeln!(w, "{name} {} {} DC 0", node(0), node(1));
                let _ = writeln!(tail, ".input {name}");
            }
            "Speaker" => {
                outputs += 1;
                let (a, c) = (node(0), node(1));
                let impedance = s.quantity("Impedance")?;
                if impedance.is_finite() {
                    let _ = writeln!(w, "{} {a} {c} {impedance:e}", element_name('R', s.name()));
                }
                let _ = writeln!(
                    tail,
                    ".output {a}{}",
                    if c == "0" {
                        String::new()
                    } else {
                        format!(" {c}")
                    }
                );
            }
            "Rail" => {
                let volts = s.quantity("Voltage")?;
                let _ = writeln!(
                    w,
                    "{} {} 0 DC {volts:e}",
                    element_name('V', s.name()),
                    node(0)
                );
            }
            "VoltageSource" => {
                let volts = s
                    .quantity("Voltage")
                    .map_err(|e| format!("{e} (only constant sources convert)"))?;
                let _ = writeln!(
                    w,
                    "{} {} {} DC {volts:e}",
                    element_name('V', s.name()),
                    node(0),
                    node(1)
                );
            }
            "Diode" => {
                let (is, n) = (s.quantity("IS")?, s.quantity("n")?);
                let model = model_name("d", s.attrs.get("PartNumber"), index);
                models.insert(model.clone(), format!("D(IS={is:e} N={n:e})"));
                let _ = writeln!(
                    w,
                    "{} {} {} {model}",
                    element_name('D', s.name()),
                    node(0),
                    node(1)
                );
            }
            "BipolarJunctionTransistor" => {
                let (is, bf, br) = (s.quantity("IS")?, s.quantity("BF")?, s.quantity("BR")?);
                let polarity = s.attr("Type")?;
                let model = model_name("q", s.attrs.get("PartNumber"), index);
                models.insert(
                    model.clone(),
                    format!("{polarity}(IS={is:e} BF={bf:e} BR={br:e})"),
                );
                let _ = writeln!(
                    w,
                    "{} {} {} {} {model}",
                    element_name('Q', s.name()),
                    node(0),
                    node(1),
                    node(2)
                );
            }
            "Potentiometer" | "VariableResistor" => {
                let r = s.quantity("Resistance")?;
                // Attributes left at their default are not written to the file.
                let wipe: f64 = s
                    .attrs
                    .get("Wipe")
                    .map_or(Ok(0.5), |w| w.parse())
                    .map_err(|_| "bad Wipe")?;
                let sweep = s.attrs.get("Sweep").map_or("Linear", String::as_str);
                let (reverse, base) = match sweep.strip_prefix("Reverse") {
                    Some(base) => (true, base),
                    None => (false, sweep),
                };
                // A reversed curve is the complementary curve read from the
                // other end of the track.
                let taper = match (base, reverse) {
                    ("Linear", _) => "lin",
                    ("Logarithmic", false) | ("AntiLogarithmic", true) => "log",
                    ("AntiLogarithmic", false) | ("Logarithmic", true) => "antilog",
                    _ => return Err(format!("`{}`: sweep `{sweep}` is not supported", s.name())),
                };
                let (a, c) = (node(0), node(1));
                let (high, wiper, low) = match (s.kind.as_str(), reverse) {
                    ("Potentiometer", false) => (a, node(2), c),
                    ("Potentiometer", true) => (c, node(2), a),
                    // Two terminals: the resistance is the low half of the
                    // track, or the high half of the complementary one.
                    (_, false) => (a.clone(), a, c),
                    (_, true) => (a, c.clone(), c),
                };
                let _ = writeln!(
                    w,
                    "{} {high} {wiper} {low} {r:e} taper={taper} pos={wipe}",
                    element_name('P', s.name())
                );
            }
            "IdealOpAmp" => {
                // No ideal element in SPICE: a controlled source with a gain
                // large enough to stand for infinity.
                let _ = writeln!(
                    w,
                    "{} {} 0 {} {} 1e6",
                    element_name('E', s.name()),
                    node(2),
                    node(0),
                    node(1)
                );
            }
            "JunctionFieldEffectTransistor" => {
                let polarity = match s.attr("Type")? {
                    "N" => "NJF",
                    "P" => "PJF",
                    other => return Err(format!("unknown JFET type `{other}`")),
                };
                let (is, vto) = (s.quantity("IS")?, s.quantity("Vt0")?);
                let (beta, lambda) = (s.quantity("Beta")?, s.quantity("Lambda")?);
                let model = model_name("j", s.attrs.get("PartNumber"), index);
                models.insert(
                    model.clone(),
                    format!("{polarity}(VTO={vto:e} BETA={beta:e} LAMBDA={lambda:e} IS={is:e})"),
                );
                let _ = writeln!(
                    w,
                    "{} {} {} {} {model}",
                    element_name('J', s.name()),
                    node(0),
                    node(1),
                    node(2)
                );
            }
            "Triode" => {
                let q = |key: &str| s.quantity(key);
                let part = model_name("u", s.attrs.get("PartNumber"), index);
                let (model, definition) = match s.attr("Model")? {
                    "DempwolfZolzer" => (
                        part,
                        format!(
                            "DZTRIODE(G={:e} MU={:e} GAMMA={:e} C={:e} GG={:e} XI={:e} CG={:e} IG0={:e})",
                            q("G")?, q("Mu")?, q("Gamma")?, q("C")?, q("Gg")?, q("Xi")?, q("Cg")?, q("Ig0")?
                        ),
                    ),
                    "ChildLangmuir" => (
                        format!("{part}_cl"),
                        format!("CLTRIODE(MU={:e} K={:e})", q("Mu")?, q("K")?),
                    ),
                    other => {
                        return Err(format!("`{}`: triode model `{other}` is not supported", s.name()))
                    }
                };
                models.insert(model.clone(), definition);
                let name = element_name('U', s.name());
                let (p, g, k) = (node(0), node(1), node(2));
                let _ = writeln!(w, "{name} {p} {g} {k} {model}");
                if s.attrs.get("SimulateCapacitances").map(String::as_str) == Some("True") {
                    let _ = writeln!(w, "C{name}_gp {p} {g} {:e}", q("Cgp")?);
                    let _ = writeln!(w, "C{name}_gk {g} {k} {:e}", q("Cgk")?);
                    let _ = writeln!(w, "C{name}_pk {p} {k} {:e}", q("Cpk")?);
                }
            }
            "Pentode" => {
                let q = |key: &str, default: f64| match s.attrs.contains_key(key) {
                    true => s.quantity(key),
                    false => Ok(default),
                };
                let model = model_name("u", s.attrs.get("PartNumber"), index);
                models.insert(
                    model.clone(),
                    format!(
                        "PENTODE(MU={:e} EX={:e} KG1={:e} KG2={:e} KP={:e} KVB={:e} RGK={:e} VG={:e} KN={:e})",
                        q("Mu", 10.7)?,
                        q("Ex", 1.31)?,
                        q("Kg1", 1672.0)?,
                        q("Kg2", 4500.0)?,
                        q("Kp", 41.16)?,
                        q("Kvb", 12.7)?,
                        q("Rgk", 2e4)?,
                        q("Vg", 13.0)?,
                        q("Kn", 3.0)?
                    ),
                );
                let _ = writeln!(
                    w,
                    "{} {} {} {} {} {model}",
                    element_name('U', s.name()),
                    node(0),
                    node(1),
                    node(2),
                    node(3)
                );
            }
            "Transformer" | "CenterTapTransformer" => {
                // Turns are primary:secondary; an ideal transformer is a
                // voltage copied one way and a current copied back.
                let (n, d) = s
                    .attr("Turns")?
                    .split_once(':')
                    .and_then(|(n, d)| {
                        Some((n.trim().parse::<f64>().ok()?, d.trim().parse::<f64>().ok()?))
                    })
                    .filter(|&(n, d)| n > 0.0 && d > 0.0)
                    .ok_or_else(|| format!("`{}`: bad turns ratio", s.name()))?;
                let name = sanitize(s.name());
                let (pa, pc) = (node(0), node(1));
                if s.kind == "Transformer" {
                    let ratio = d / n;
                    let (sa, sc) = (node(2), node(3));
                    let _ = writeln!(w, "V{name} {sa} {name}_s DC 0");
                    let _ = writeln!(w, "E{name} {name}_s {sc} {pa} {pc} {ratio:e}");
                    let _ = writeln!(w, "F{name} {pa} {pc} V{name} {:e}", -ratio);
                } else {
                    // Each half of the secondary carries half its voltage.
                    let ratio = d / (2.0 * n);
                    let (sa, st, sc) = (node(2), node(3), node(4));
                    let _ = writeln!(w, "V{name}a {sa} {name}_a DC 0");
                    let _ = writeln!(w, "E{name}a {name}_a {st} {pa} {pc} {ratio:e}");
                    let _ = writeln!(w, "F{name}a {pa} {pc} V{name}a {:e}", -ratio);
                    let _ = writeln!(w, "V{name}c {name}_c {sc} DC 0");
                    let _ = writeln!(w, "E{name}c {st} {name}_c {pa} {pc} {ratio:e}");
                    let _ = writeln!(w, "F{name}c {pa} {pc} V{name}c {:e}", -ratio);
                }
            }
            "SPDT" | "SP3T" => {
                // The closed throw was merged into the common net above; the
                // open ones keep LiveSPICE's 1 TOhm so nothing floats.
                let common = node(0);
                let name = sanitize(s.name());
                for throw in 1..p.len() {
                    let other = node(throw);
                    if other != common {
                        let _ = writeln!(w, "R{name}_open{throw} {common} {other} 1e12");
                    }
                }
            }
            "OpAmp" => {
                let (rin, rout) = (s.quantity("Rin")?, s.quantity("Rout")?);
                let (aol, gbp) = (s.quantity("Aol")?, s.quantity("GBP")?);
                let rails = p[3].is_some() && p[4].is_some();
                let name = format!(
                    "{}{}",
                    model_name("opamp", s.attrs.get("PartNumber"), index),
                    if rails { "" } else { "_norails" }
                );
                // LiveSPICE's op-amp: input resistance, one gain stage with
                // a single pole, diode clamps two volts inside the rails,
                // buffered output.
                let mut sub = format!(
                    ".subckt {name} inp inn out{}\n\
                     Rin inp inn {rin:e}\n\
                     Ggain 0 stage inp inn {:e}\n\
                     Rstage stage 0 1e3\n\
                     Cstage stage 0 {:e}\n",
                    if rails { " vcc vee" } else { "" },
                    aol / 1e3,
                    aol / (2.0 * std::f64::consts::PI * 1e3 * gbp),
                );
                if rails {
                    sub.push_str(
                        "Vhigh vcc top DC 2\n\
                         Dhigh stage top opamp_clamp\n\
                         Vlow bottom vee DC 2\n\
                         Dlow bottom stage opamp_clamp\n",
                    );
                    models.insert("opamp_clamp".to_string(), "D(IS=8e-16 N=1)".to_string());
                }
                let _ = writeln!(sub, "Gout 0 out stage out {:e}\n.ends", 1.0 / rout);
                subckts.insert(name.clone(), sub);
                let supplies = if rails {
                    format!(" {} {}", node(3), node(4))
                } else {
                    String::new()
                };
                let _ = writeln!(
                    w,
                    "{} {} {} {}{supplies} {name}",
                    element_name('X', s.name()),
                    node(0),
                    node(1),
                    node(2)
                );
            }
            other => {
                return Err(format!(
                    "component `{other}` is not supported by the converter yet"
                ))
            }
        }
    }
    if inputs != 1 || outputs != 1 {
        return Err(format!(
            "expected one Input and one Speaker, found {inputs} and {outputs}"
        ));
    }

    let mut out = String::new();
    let _ = writeln!(out, "{}", if title.is_empty() { source } else { &title });
    let _ = writeln!(
        out,
        "* Converted by `spice-lab convert` from the LiveSPICE schematic"
    );
    let _ = writeln!(
        out,
        "* \"{source}\" (LiveSPICE is MIT licensed, (c) Dillon Sharlet)."
    );
    out.push_str(&body);
    for sub in subckts.values() {
        out.push_str(sub);
    }
    for (name, model) in &models {
        let _ = writeln!(out, ".model {name} {model}");
    }
    out.push_str(&tail);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantities() {
        for (text, want) in [
            ("10 k\u{2126}", 10e3),
            ("10 k\u{3a9}", 10e3),
            ("1 μF", 1e-6),
            ("200 nA", 200e-9),
            ("200 k", 200e3),
            ("1 MHz", 1e6),
            ("4.5 V", 4.5),
            ("1.3", 1.3),
            ("100 m\u{2126}", 0.1),
            ("10 fA", 10e-15),
            ("1.73E-06", 1.73e-6),
            ("2.4E-12", 2.4e-12),
        ] {
            assert_eq!(parse_quantity(text), Some(want), "{text}");
        }
        assert_eq!(parse_quantity("∞ \u{2126}"), Some(f64::INFINITY));
        assert_eq!(parse_quantity("x"), None);
    }

    #[test]
    fn symbol_transform_matches_livespice() {
        let at = |rotation, flip| {
            Symbol {
                kind: String::new(),
                attrs: HashMap::new(),
                position: (100, 50),
                rotation,
                flip,
            }
            .global((10, 20))
        };
        assert_eq!(at(0, false), (110, 30));
        assert_eq!(at(0, true), (110, 70));
        assert_eq!(at(-1, false), (120, 60));
        assert_eq!(at(1, false), (80, 40));
        assert_eq!(at(-2, false), (90, 70));
    }
}
