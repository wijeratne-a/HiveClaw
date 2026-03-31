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
# Prepend interpreter's bin dir so pip-installed `cmake` (see requirements-spike.txt) is on PATH.
_PY_BINDIR := $(shell "$(MATURIN_PYTHON)" -c 'import os,sys; print(os.path.dirname(sys.executable))')

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
	# Single thread: all integration tests register the same Mach label `com.hiveclaw.pheromoned`.
	cargo test -p hiveclaw-daemon -- --test-threads=1

# MLX / numpy / psutil for scripts/intelligence_spike.py (idempotent).
spike-deps:
	"$(PYTHON)" -m pip install -r "$(ROOT)/scripts/requirements-spike.txt"

# PyO3/maturin: pin interpreter; force CARGO_TARGET_DIR to this repo's target/ so ~/.cargo/config.toml build.target-dir cannot pull another checkout's PyO3 cache.
# Unset VIRTUAL_ENV: if it points at another checkout, maturin can pick that .venv/bin/python and fail cross-compile checks even when PYO3_PYTHON is correct.
python: spike-deps
	@PATH="$(_PY_BINDIR):$$PATH" command -v cmake >/dev/null 2>&1 || (echo >&2 "cmake not found (required for crates/hiveclaw-mlx). pip should install it: check scripts/requirements-spike.txt"; exit 1)
	cd crates/hiveclaw-mlx && PATH="$(_PY_BINDIR):$$PATH" CMAKE_ARGS="-DPython_EXECUTABLE=$(MATURIN_PYTHON)" "$(MATURIN_PYTHON)" setup.py build_ext --inplace
	@sh -c 'for f in "$(ROOT)"/crates/hiveclaw-mlx/hiveclaw_mlx_ext*.so "$(ROOT)"/crates/hiveclaw-mlx/hiveclaw_mlx_ext*.dylib; do \
	  if [ -e "$$f" ]; then install -m644 "$$f" "$(ROOT)/crates/hiveclaw-python/python/hiveclaw_python/"; fi; \
	done'
	cd crates/hiveclaw-python && \
	  _py="$(MATURIN_PYTHON)"; \
	  _venv="$$(cd "$$(dirname "$$_py")/.." && pwd)"; \
	  unset VIRTUAL_ENV; \
	  if [ -f "$$_venv/pyvenv.cfg" ]; then export VIRTUAL_ENV="$$_venv"; fi; \
	  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CARGO_BUILD_TARGET \
	    CARGO_TARGET_DIR="$(ROOT)/target" PYO3_PYTHON="$$_py" \
	    "$$_py" -m maturin develop --release

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
