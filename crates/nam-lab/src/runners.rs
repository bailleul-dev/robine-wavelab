//! A single interface over every implementation we measure.

use clap::ValueEnum;
use nam_reference_sys::{Mode, ReferenceModel};
use robine_nam::{Model, NamFile, Size};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Impl {
    /// robine-nam (the library under test).
    Robine,
    /// Upstream NeuralAmpModelerCore, generic Eigen WaveNet (the oracle).
    RefGeneric,
    /// Upstream NeuralAmpModelerCore, hand-optimised A2 fast path.
    RefFast,
    /// Upstream dispatcher (`get_dsp`), whatever it picks.
    RefAuto,
}

impl Impl {
    pub fn label(self) -> &'static str {
        match self {
            Impl::Robine => "robine",
            Impl::RefGeneric => "ref-generic",
            Impl::RefFast => "ref-fast",
            Impl::RefAuto => "ref-auto",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SizeArg {
    Small,
    Large,
}

impl From<SizeArg> for Size {
    fn from(s: SizeArg) -> Size {
        match s {
            SizeArg::Small => Size::Small,
            SizeArg::Large => Size::Large,
        }
    }
}

pub trait Runner {
    fn description(&self) -> String;
    fn process(&mut self, input: &[f32], output: &mut [f32]);
}

impl Runner for Model {
    fn description(&self) -> String {
        format!("robine-nam Model ({})", self.size())
    }
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        Model::process(self, input, output)
    }
}

impl Runner for ReferenceModel {
    fn description(&self) -> String {
        ReferenceModel::description(self)
    }
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        ReferenceModel::process(self, input, output)
    }
}

/// A parsed model file plus its raw text (the reference re-parses it).
pub struct LoadedFile {
    pub text: String,
    pub file: NamFile,
}

impl LoadedFile {
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file = NamFile::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { text, file })
    }

    /// Resolve the requested size (largest when `None`).
    pub fn pick(&self, size: Option<Size>) -> Result<Size, String> {
        match size {
            Some(s) if self.file.variant(s).is_some() => Ok(s),
            Some(s) => Err(format!(
                "size {s} not in file (available: {})",
                self.file
                    .sizes()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            None => Ok(self.file.largest().size),
        }
    }

    /// `slim` value that makes upstream pick the same variant as `size`.
    pub fn slim_for(&self, size: Size) -> f64 {
        let mut previous = 0.0;
        for v in &self.file.variants {
            if v.size == size {
                return previous;
            }
            previous = v.max_value;
        }
        1.0
    }

    pub fn build(
        &self,
        imp: Impl,
        size: Size,
        max_block: usize,
    ) -> Result<Box<dyn Runner>, String> {
        match imp {
            Impl::Robine => Ok(Box::new(
                self.file
                    .build_with(size, max_block)
                    .map_err(|e| e.to_string())?,
            )),
            Impl::RefGeneric | Impl::RefFast | Impl::RefAuto => {
                let mode = match imp {
                    Impl::RefGeneric => Mode::Generic,
                    Impl::RefFast => Mode::A2Fast,
                    _ => Mode::Auto,
                };
                Ok(Box::new(ReferenceModel::from_json(
                    &self.text,
                    mode,
                    self.slim_for(size),
                    max_block,
                )?))
            }
        }
    }
}
