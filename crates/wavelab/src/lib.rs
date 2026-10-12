//! # wavelab
//!
//! Measuring audio devices from the outside: what goes in, what comes out,
//! and how two outputs differ. Nothing here knows what the device is, so
//! the same measures serve a circuit simulation, a neural capture and a real
//! pedal recorded through an interface.
//!
//! * [`judge`]: a probe signal of steady tones and five criteria read from
//!   what a device did to it (frequency response, dynamics, harmonics,
//!   attack, recovery).
//! * [`null`]: align two renders of the same material, subtract, and rate
//!   what is left: error-to-signal ratio, noise-to-mask ratio
//!   ([`perceptual`]), and PEAQ's grade ([`peaq`]).
//! * [`metrics`]: sample-exact deltas and long-term spectral distance.
//! * [`signal`] and [`wav`]: test signals and file input and output.

pub mod judge;
pub mod metrics;
pub mod null;
pub mod peaq;
pub mod perceptual;
pub mod signal;
pub mod wav;
