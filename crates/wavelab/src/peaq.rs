//! PEAQ (ITU-R BS.1387, basic version) through GstPEAQ, an external program
//! that `make peaq` builds.

use crate::wav;
use std::path::PathBuf;

/// PEAQ's verdict on two aligned signals: the objective difference grade
/// and the distortion index it is derived from.
pub fn grade(reference: &[f32], candidate: &[f32], rate: u32) -> Result<(f64, f64), String> {
    if rate != 48_000 {
        return Err(format!(
            "PEAQ is defined at 48 kHz, these files are at {rate} Hz"
        ));
    }
    // GstPEAQ is built in peaq/build at the root of this repository (the
    // executable being in target/<profile>), or wherever $PEAQ_DIR says.
    let dir = match std::env::var_os("PEAQ_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::current_exe()
            .map_err(|e| e.to_string())?
            .ancestors()
            .nth(3)
            .map(|root| root.join("peaq/build"))
            .unwrap_or_default(),
    };
    let binary = dir.join("peaq");
    if !binary.exists() {
        return Err(format!(
            "{} not found: build GstPEAQ with `make peaq` (needs GStreamer)",
            binary.display()
        ));
    }
    let scratch = std::env::temp_dir().join(format!("spice-lab-peaq-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let (ref_path, test_path) = (scratch.join("reference.wav"), scratch.join("candidate.wav"));
    // PEAQ reads 16-bit material: write that, the level being set already.
    wav::write_mono_i16(&ref_path, reference, rate)?;
    wav::write_mono_i16(&test_path, candidate, rate)?;
    let output = std::process::Command::new(&binary)
        .arg(&ref_path)
        .arg(&test_path)
        .env("GST_PLUGIN_PATH", &dir)
        .output()
        .map_err(|e| format!("{}: {e}", binary.display()));
    let _ = std::fs::remove_dir_all(&scratch);
    let output = output?;
    let text = String::from_utf8_lossy(&output.stdout);
    let field = |label: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(label)?.trim().parse::<f64>().ok())
            .ok_or_else(|| {
                format!(
                    "GstPEAQ gave no `{label}`:\n{text}{}",
                    String::from_utf8_lossy(&output.stderr)
                )
            })
    };
    Ok((
        field("Objective Difference Grade:")?,
        field("Distortion Index:")?,
    ))
}
