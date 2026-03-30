# HiveClaw — Phase 4: XPC Mach service + IOSurface + PyO3.
.PHONY: build build-release poc clean test-ipc python spike-deps daemon-load daemon-unload daemon-uninstall

ROOT := $(abspath .)
PBIN := $(ROOT)/target/release/pheromoned
GEN_PLIST := $(ROOT)/com.hiveclaw.pheromoned.gen.plist
# launchctl often returns EIO (error 5) when bootstrapping a plist from an arbitrary repo path; install under ~/Library/LaunchAgents/ (Apple convention).
LAUNCH_AGENTS := $(HOME)/Library/LaunchAgents
INSTALLED_PLIST := $(LAUNCH_AGENTS)/com.hiveclaw.pheromoned.plist
# Use `make python PYTHON=.venv/bin/python3` if `python3` is not your venv.
PYTHON ?= python3
# After `cd crates/...`, relative PYTHON paths break; resolve when it is a path to a file.
_PY_ABS := $(abspath $(PYTHON))
MATURIN_PYTHON := $(if $(wildcard $(_PY_ABS)),$(_PY_ABS),$(PYTHON))

build:
	cargo build --workspace

build-release:
	cargo build --release -p hiveclaw-daemon

poc: build
	@echo "Phase 4: use  make daemon-load  then  python scripts/intelligence_spike.py  (see scripts/README.md)"

clean:
	cargo clean
	rm -f /tmp/hiveclaw.sock $(GEN_PLIST)
	-launchctl bootout gui/$$(id -u) com.hiveclaw.pheromoned 2>/dev/null || true

test-ipc:
	cargo test -p hiveclaw-daemon

# MLX / numpy / psutil for scripts/intelligence_spike.py (idempotent).
spike-deps:
	"$(PYTHON)" -m pip install -r "$(ROOT)/scripts/requirements-spike.txt"

# PyO3/maturin: pin interpreter; force CARGO_TARGET_DIR to this repo's target/ so ~/.cargo/config.toml build.target-dir cannot pull another checkout's PyO3 cache.
# Unset VIRTUAL_ENV: if it points at another checkout, maturin can pick that .venv/bin/python and fail cross-compile checks even when PYO3_PYTHON is correct.
python: spike-deps
	cd crates/hiveclaw-python && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CARGO_BUILD_TARGET -u VIRTUAL_ENV CARGO_TARGET_DIR="$(ROOT)/target" PYO3_PYTHON="$(MATURIN_PYTHON)" "$(MATURIN_PYTHON)" -m maturin develop --release

daemon-load: build-release
	@test -f "$(PBIN)" || (echo "missing $(PBIN); run: cargo build --release -p hiveclaw-daemon" && exit 1)
	mkdir -p "$(LAUNCH_AGENTS)"
	sed "s|@PROGRAM@|$(PBIN)|g" "$(ROOT)/com.hiveclaw.pheromoned.plist.in" > "$(INSTALLED_PLIST)"
	chmod 644 "$(INSTALLED_PLIST)"
	cp "$(INSTALLED_PLIST)" "$(GEN_PLIST)"
	-launchctl bootout gui/$$(id -u) com.hiveclaw.pheromoned 2>/dev/null || true
	launchctl bootstrap gui/$$(id -u) "$(INSTALLED_PLIST)" || (echo >&2 "launchctl bootstrap failed (often EIO in IDE terminals). Try: same command from Terminal.app with a GUI login session; see scripts/README.md"; exit 1)

daemon-unload:
	-launchctl bootout gui/$$(id -u) com.hiveclaw.pheromoned 2>/dev/null || true

daemon-uninstall: daemon-unload
	rm -f "$(INSTALLED_PLIST)"
