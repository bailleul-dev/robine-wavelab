//! A trained stand-in for one nonlinear port's solve: the circuit's own
//! solve of a diode pair, `v = p + K i(v)`, plus a small network's
//! correction, fed the port's last drives and the current it gave last
//! sample. Written by the training scripts as JSON.

use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Layer {
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Model {
    pub port: usize,
    history: usize,
    current_scale: f64,
    voltage_scale: f64,
    is: f64,
    nvt: f64,
    k: f64,
    layers: Vec<Layer>,
}

/// A model with its running memory.
pub struct Running {
    model: Model,
    drives: Vec<f64>,
    previous: f64,
}

impl Model {
    pub fn read(path: &Path) -> Result<Model, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn pair(&self, v: f64) -> (f64, f64) {
        let x = v / self.nvt;
        (
            2.0 * self.is * x.sinh(),
            2.0 * self.is * x.cosh() / self.nvt,
        )
    }

    /// The circuit's own solve for drive `p`.
    fn solve(&self, p: f64) -> f64 {
        let mut v = 0.0f64;
        for _ in 0..60 {
            let (i, g) = self.pair(v);
            let step = (v - p - self.k * i) / (1.0 - self.k * g);
            v -= step.clamp(-0.05, 0.05);
            if step.abs() < 1e-12 {
                break;
            }
        }
        self.pair(v).0
    }

    fn correction(&self, features: &[f64]) -> f64 {
        let mut x = features.to_vec();
        let last = self.layers.len() - 1;
        for (k, layer) in self.layers.iter().enumerate() {
            x = layer
                .weight
                .iter()
                .zip(&layer.bias)
                .map(|(row, b)| {
                    let s = b + row.iter().zip(&x).map(|(w, v)| w * v).sum::<f64>();
                    if k < last {
                        s.tanh()
                    } else {
                        s
                    }
                })
                .collect();
        }
        x[0] * self.current_scale
    }

    pub fn start(self, rest_current: f64) -> Running {
        Running {
            drives: vec![0.0; self.history],
            previous: rest_current,
            model: self,
        }
    }
}

impl Running {
    pub fn port(&self) -> usize {
        self.model.port
    }

    /// The current for this sample, given the port's drive.
    pub fn next(&mut self, drive: f64) -> f64 {
        self.drives.rotate_right(1);
        self.drives[0] = drive;
        let m = &self.model;
        let mut features: Vec<f64> = self.drives.iter().map(|d| d / m.voltage_scale).collect();
        features.push(self.previous / m.current_scale);
        let current = m.solve(drive) + m.correction(&features);
        self.previous = current;
        current
    }
}
