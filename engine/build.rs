// Picks the NNUE net to embed: $EVALFILE if set, otherwise the file named in net.txt
// (downloaded by `make net`). Relative EVALFILE paths are relative to this directory.
// Also compiles the vendored Fathom tablebase prober (../third_party/fathom) with
// the system C compiler ($CC, default cc) and links it in.
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
    build_fathom(&dir);
    version(&dir);
}

/// Engine version for `id name`: the Cargo version on a release tag (vX.Y.Z),
/// else with the git position, like `git describe`: 0.1.0-12-gabc1234 (12
/// commits after the last release tag), 0.1.0-gabc1234 before any release
/// tag, and -dirty for uncommitted changes. Without git (e.g. a source zip)
/// just the Cargo version.
fn version(dir: &std::path::Path) {
    let base = env::var("CARGO_PKG_VERSION").unwrap();
    let git = |args: &[&str]| {
        process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let v = match git(&["describe", "--tags", "--match", "v[0-9]*.[0-9]*.[0-9]*", "--dirty"]) {
        Some(d) => d.trim_start_matches('v').to_string(),
        None => match git(&["describe", "--always", "--dirty"]) {
            Some(h) => format!("{base}-g{h}"),
            None => base,
        },
    };
    println!("cargo:rustc-env=INCIPIT_VERSION={v}");
    // Rebuild when HEAD moves (new commit or branch switch).
    if let Some(g) = git(&["rev-parse", "--git-dir"]) {
        let g = dir.join(g);
        println!("cargo:rerun-if-changed={}", g.join("HEAD").display());
        if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={}", g.join(r).display());
        }
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

fn build_fathom(dir: &std::path::Path) {
    let src = dir.join("../third_party/fathom");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=CC");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("fathom.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = process::Command::new(&cc)
        .args(["-std=gnu99", "-O2", "-fPIC", "-w", "-c"])
        .arg("-I")
        .arg(&src)
        .arg(src.join("tbprobe.c"))
        .arg("-o")
        .arg(&out)
        .status();
    match status {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg={}", out.display()),
        _ => {
            eprintln!("error: compiling the Fathom tablebase prober with `{}` failed (set CC to a C compiler)", cc);
            process::exit(1);
        }
    }
}
