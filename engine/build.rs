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
}
