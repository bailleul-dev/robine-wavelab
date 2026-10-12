//! Safe wrapper over the upstream NeuralAmpModelerCore reference build.
//!
//! The C++ sources are compiled from `vendor/NeuralAmpModelerCore`, fetched
//! at the commit pinned in `UPSTREAM` by `vendor.sh`, with
//! `NAM_ENABLE_A2_FAST`, so both the generic WaveNet and the
//! upstream A2 fast path are available for comparison.

use std::ffi::{c_char, c_double, c_int, CStr};

mod ffi {
    use super::*;
    #[repr(C)]
    pub struct nam_ref_model {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn nam_ref_create(
            json: *const c_char,
            json_len: usize,
            mode: c_int,
            slim: c_double,
            max_block: c_int,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut nam_ref_model;
        pub fn nam_ref_destroy(m: *mut nam_ref_model);
        pub fn nam_ref_sample_rate(m: *const nam_ref_model) -> c_double;
        pub fn nam_ref_prewarm_samples(m: *mut nam_ref_model) -> c_int;
        pub fn nam_ref_description(m: *const nam_ref_model) -> *const c_char;
        pub fn nam_ref_reset(m: *mut nam_ref_model);
        pub fn nam_ref_process(
            m: *mut nam_ref_model,
            input: *const f32,
            output: *mut f32,
            n: c_int,
        );
    }
}

/// Which upstream code path to instantiate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Upstream dispatcher: A2 fast path when the shape matches.
    Auto,
    /// Generic Eigen-based WaveNet.
    Generic,
    /// Upstream hand-optimised `A2FastModel`.
    A2Fast,
}

impl Mode {
    fn raw(self) -> c_int {
        match self {
            Mode::Auto => 0,
            Mode::Generic => 1,
            Mode::A2Fast => 2,
        }
    }
}

/// An upstream model instance.
pub struct ReferenceModel {
    ptr: *mut ffi::nam_ref_model,
    max_block: usize,
}

// The C++ object is only touched through &mut self.
unsafe impl Send for ReferenceModel {}

impl ReferenceModel {
    /// Build from `.nam` JSON text. `slim` selects the container submodel
    /// (upstream semantics: first submodel with `slim < max_value`).
    pub fn from_json(json: &str, mode: Mode, slim: f64, max_block: usize) -> Result<Self, String> {
        let mut err = vec![0u8; 1024];
        let ptr = unsafe {
            ffi::nam_ref_create(
                json.as_ptr() as *const c_char,
                json.len(),
                mode.raw(),
                slim,
                max_block as c_int,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if ptr.is_null() {
            let msg = CStr::from_bytes_until_nul(&err)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown error".into());
            return Err(msg);
        }
        Ok(Self { ptr, max_block })
    }

    /// Expected sample rate, if the file carries one.
    pub fn sample_rate(&self) -> Option<f64> {
        let sr = unsafe { ffi::nam_ref_sample_rate(self.ptr) };
        (sr > 0.0).then_some(sr)
    }

    /// Prewarm sample count reported upstream.
    pub fn prewarm_samples(&mut self) -> usize {
        unsafe { ffi::nam_ref_prewarm_samples(self.ptr) }.max(0) as usize
    }

    /// Description of the instantiated upstream class.
    pub fn description(&self) -> String {
        unsafe { CStr::from_ptr(ffi::nam_ref_description(self.ptr)) }
            .to_string_lossy()
            .into_owned()
    }

    /// Largest block accepted by `process_block`.
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Reset and prewarm.
    pub fn reset(&mut self) {
        unsafe { ffi::nam_ref_reset(self.ptr) }
    }

    /// Process one block of at most `max_block` frames.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        assert!(input.len() <= self.max_block);
        unsafe {
            ffi::nam_ref_process(
                self.ptr,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as c_int,
            )
        }
    }

    /// Process slices of any length, chunked by `max_block`.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, o) in input
            .chunks(self.max_block)
            .zip(output.chunks_mut(self.max_block))
        {
            self.process_block(i, o);
        }
    }
}

impl Drop for ReferenceModel {
    fn drop(&mut self) {
        unsafe { ffi::nam_ref_destroy(self.ptr) }
    }
}
