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
    build_fathom(&dir);
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
