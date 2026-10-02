// Compiles the vendored Fathom tablebase prober (../third_party/fathom) with the
// system C compiler ($CC, default cc) for the `tbstats` command.
use std::path::PathBuf;
use std::{env, process};

fn main() {
    let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../third_party/fathom");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=CC");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("fathom.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let ok = process::Command::new(&cc)
        .args(["-std=gnu99", "-O2", "-fPIC", "-w", "-c"])
        .arg("-I")
        .arg(&src)
        .arg(src.join("tbprobe.c"))
        .arg("-o")
        .arg(&out)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("error: compiling the Fathom tablebase prober with `{}` failed (set CC to a C compiler)", cc);
        process::exit(1);
    }
    println!("cargo:rustc-link-arg={}", out.display());
}
