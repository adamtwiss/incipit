// Picks the NNUE net to embed: $EVALFILE if set, otherwise the file named in net.txt
// (downloaded by `make net`). Relative EVALFILE paths are relative to this directory.
use std::path::PathBuf;
use std::{env, fs, process};

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=EVALFILE");
    println!("cargo:rerun-if-changed=net.txt");

    let net = match env::var("EVALFILE") {
        Ok(p) if !p.is_empty() => dir.join(p),
        _ => {
            let url = fs::read_to_string(dir.join("net.txt")).unwrap_or_else(|_| {
                eprintln!("error: EVALFILE is not set and net.txt is missing");
                process::exit(1);
            });
            dir.join(url.trim().rsplit('/').next().unwrap())
        }
    };
    if !net.is_file() {
        eprintln!(
            "error: NNUE net {} not found. Run `make net` to download it, or set EVALFILE=/path/to/net.nnue",
            net.display()
        );
        process::exit(1);
    }
    println!("cargo:rerun-if-changed={}", net.display());
    println!("cargo:rustc-env=INCIPIT_NET={}", net.display());

    // AVX-512 intrinsics are stable from Rust 1.89. With an older compiler
    // the AVX2 kernels are used even on AVX-512 hosts.
    println!("cargo:rustc-check-cfg=cfg(avx512_intrinsics)");
    if rustc_minor() >= 89 {
        println!("cargo:rustc-cfg=avx512_intrinsics");
    }
}

/// Minor version of the compiler building us ("rustc 1.89.0 ..." -> 89).
fn rustc_minor() -> u32 {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|v| v.split_whitespace().nth(1)?.split('.').nth(1)?.parse().ok())
        .unwrap_or(0)
}
