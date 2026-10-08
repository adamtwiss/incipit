# Incipit Chess Engine — Makefile
# Supports: manual builds, OpenBench integration, PGO builds
#
# Usage:
#   make                  Build with native CPU optimizations (downloads the net first if needed)
#   make EXE=incipit-v2   Build with custom output name
#   make EVALFILE=x.nnue  Embed a different net
#   make openbench        OpenBench-compatible build target
#   make pgo              PGO-optimized build (profiled with bench)
#   make net              Download the production NNUE net and verify its checksum
#   make clean            Remove build artifacts
#
# Nets are named net-XXXXXXXX.nnue, where XXXXXXXX is the first 8 hex digits
# (upper case) of the file's SHA-256. net.txt holds the release URL of the
# production net.

# Configuration
EXE := incipit
NET_URL := $(strip $(shell cat net.txt 2>/dev/null))
# EVALFILE: defaults to the filename from net.txt (e.g. net-E4722BA0.nnue).
# OpenBench overrides this with an absolute path to the network file.
EVALFILE := $(if $(NET_URL),$(notdir $(NET_URL)),net.nnue)
# Repo and release tag, parsed from https://github.com/OWNER/REPO/releases/download/TAG/FILE
# (used by the GitHub CLI fallback while the repo is private).
NET_REPO := $(word 3,$(subst /, ,$(NET_URL)))/$(word 4,$(subst /, ,$(NET_URL)))
NET_TAG := $(word 7,$(subst /, ,$(NET_URL)))
# Cargo.lock version 4 needs cargo 1.78.
MIN_RUST_VERSION := 1.78.0

# Platform detection
ifeq ($(OS),Windows_NT)
    NAME := $(EXE).exe
    RM := del /q
else
    NAME := $(EXE)
    RM := rm -f
endif

# Rust flags (overrides the portable flags in .cargo/config.toml)
export RUSTFLAGS := -Ctarget-cpu=native

# Default: build with the embedded NNUE net (downloads from net.txt if needed)
rule: check-rust net
	EVALFILE=$(abspath $(EVALFILE)) cargo rustc --release$(if $(TUNE), --features tune) -- --emit link=$(NAME)

# OpenBench build — enables the `tune` feature so the SPSA tunable UCI options are
# advertised (required for OpenBench SPSA to setoption them). Normal `make` keeps
# them hidden. Point OpenBench's Incipit build command at `make openbench`.
openbench: TUNE := 1
openbench: rule

# PGO build (profile-guided optimization): build instrumented, profile with
# `bench`, rebuild with the profile. Uses rustc's own PGO flags and the
# toolchain's llvm-profdata (`rustup component add llvm-tools`; a system
# llvm-profdata must match rustc's LLVM major version). Without llvm-profdata
# it falls back to the normal build. Separate target dirs keep the normal
# build's cache intact.
PROFDATA := $(firstword $(wildcard $(shell rustc --print sysroot 2>/dev/null)/lib/rustlib/*/bin/llvm-profdata) $(shell command -v llvm-profdata 2>/dev/null))
PGO_DIR := $(abspath target/pgo-data)
pgo: check-rust net
ifeq ($(PROFDATA),)
	@echo "Warning: llvm-profdata not found (rustup component add llvm-tools); building without PGO"
	$(MAKE) rule
else
	rm -rf $(PGO_DIR)
	EVALFILE=$(abspath $(EVALFILE)) RUSTFLAGS="$(RUSTFLAGS) -Cprofile-generate=$(PGO_DIR)" cargo rustc --release --target-dir target/pgo-gen$(if $(TUNE), --features tune) -- --emit link=$(NAME)-pgo-gen
	./$(NAME)-pgo-gen bench > /dev/null
	$(PROFDATA) merge -o $(PGO_DIR)/merged.profdata $(PGO_DIR)
	EVALFILE=$(abspath $(EVALFILE)) RUSTFLAGS="$(RUSTFLAGS) -Cprofile-use=$(PGO_DIR)/merged.profdata" cargo rustc --release --target-dir target/pgo-use$(if $(TUNE), --features tune) -- --emit link=$(NAME)
	$(RM) $(NAME)-pgo-gen
endif

# OpenBench-style build with PGO (to test PGO on the fleet, point
# `openbench` at `pgo` on a branch: the server always runs `make openbench`).
openbench-pgo: TUNE := 1
openbench-pgo: pgo

# Download the production NNUE net (uses the filename from net.txt) and check
# that its SHA-256 matches the checksum in its name.
net:
	@if [ ! -s "$(EVALFILE)" ] && [ -n "$(NET_URL)" ]; then \
		set -e; \
		tmp="$(EVALFILE).tmp"; \
		trap 'rm -f "$$tmp"' EXIT; \
		echo "Downloading NNUE net from $(NET_URL)..."; \
		if ! curl -fsSL --retry 3 --retry-delay 2 "$(NET_URL)" -o "$$tmp"; then \
			if command -v gh >/dev/null 2>&1; then \
				echo "Direct download failed; trying the GitHub CLI (private repo)..."; \
				gh release download "$(NET_TAG)" -R "$(NET_REPO)" -p "$(notdir $(NET_URL))" -O "$$tmp" --clobber; \
			else \
				echo "Error: download failed. If the repo is private, install the GitHub CLI and run 'gh auth login', or copy the net here as $(EVALFILE)"; \
				exit 1; \
			fi; \
		fi; \
		test -s "$$tmp"; \
		want=$$(basename "$(EVALFILE)" .nnue | sed -n 's/^net-\([0-9A-F]\{8\}\)$$/\1/p'); \
		got=$$( (sha256sum "$$tmp" 2>/dev/null || shasum -a 256 "$$tmp") | cut -c1-8 | tr a-f A-F); \
		if [ -n "$$want" ] && [ "$$want" != "$$got" ]; then \
			echo "Error: checksum mismatch for $(EVALFILE) (got $$got)"; exit 1; \
		fi; \
		mv "$$tmp" "$(EVALFILE)"; \
		trap - EXIT; \
		echo "Downloaded $(EVALFILE)"; \
	elif [ -s "$(EVALFILE)" ]; then \
		echo "$(EVALFILE) already exists"; \
	else \
		echo "Warning: no net.txt found and no $(EVALFILE) present"; \
	fi

# Check Rust toolchain version.
# Version compare uses awk rather than `sort -V`, which is a GNU coreutils
# extension absent from Git Bash / minimal Windows Unix shells.
check-rust:
	@command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found. Install Rust from https://rustup.rs"; exit 1; }
	@RUST_VERSION=$$(rustc --version | sed 's/rustc \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/'); \
	MIN="$(MIN_RUST_VERSION)"; \
	awk -v min="$$MIN" -v cur="$$RUST_VERSION" ' \
		function cmp(a,b) { \
			split(a,A,"."); split(b,B,"."); \
			for(i=1;i<=3;i++){ \
				if(A[i]+0 < B[i]+0) return -1; \
				if(A[i]+0 > B[i]+0) return 1; \
			} \
			return 0; \
		} \
		BEGIN { \
			if(cmp(cur,min) < 0) { \
				print "Error: Rust " cur " is too old. Need >= " min ". Run: rustup update"; \
				exit 1; \
			} \
		}'

clean:
	cargo clean
	$(RM) $(NAME)

.PHONY: rule openbench openbench-pgo pgo net check-rust clean
