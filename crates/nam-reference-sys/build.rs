use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor/NeuralAmpModelerCore");
    let nam = vendor.join("NAM");
    if !nam.join("dsp.h").exists() {
        panic!(
            "NeuralAmpModelerCore sources not found at {}.\nRun `make build` (or crates/nam-reference-sys/vendor.sh) to fetch the pinned upstream.",
            vendor.display()
        );
    }

    let mut sources: Vec<PathBuf> = Vec::new();
    for dir in [nam.clone(), nam.join("wavenet")] {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "cpp"))
            .collect();
        entries.sort();
        sources.extend(entries);
    }
    sources.push(manifest.join("shim/nam_ref.cpp"));

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .include(&vendor)
        .include(vendor.join("Dependencies/eigen"))
        .include(vendor.join("Dependencies/nlohmann"))
        .define("NAM_ENABLE_A2_FAST", None)
        .opt_level(3)
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-Wno-everything")
        .flag_if_supported("-w")
        .files(&sources);
    // Match the upstream CMake flags on macOS.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build.flag("-stdlib=libc++");
    }
    build.compile("nam_reference");

    println!("cargo:rerun-if-changed=shim/nam_ref.cpp");
    println!("cargo:rerun-if-changed=shim/nam_ref.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=UPSTREAM");
    for s in &sources {
        println!("cargo:rerun-if-changed={}", s.display());
    }
}
