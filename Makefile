# HiveClaw — Phase 4: XPC Mach service + IOSurface + PyO3.
.PHONY: build build-release poc clean test-ipc python spike-deps daemon-load daemon-unload

ROOT := $(abspath .)
PBIN := $(ROOT)/target/release/pheromoned
GEN_PLIST := $(ROOT)/com.hiveclaw.pheromoned.gen.plist
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

python: spike-deps
	cd crates/hiveclaw-python && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV "$(MATURIN_PYTHON)" -m maturin develop --release

daemon-load: build-release
	@test -f "$(PBIN)" || (echo "missing $(PBIN); run: cargo build --release -p hiveclaw-daemon" && exit 1)
	sed "s|@PROGRAM@|$(PBIN)|g" "$(ROOT)/com.hiveclaw.pheromoned.plist.in" > "$(GEN_PLIST)"
	-launchctl bootout gui/$$(id -u) com.hiveclaw.pheromoned 2>/dev/null || true
	launchctl bootstrap gui/$$(id -u) "$(GEN_PLIST)"

daemon-unload:
	-launchctl bootout gui/$$(id -u) com.hiveclaw.pheromoned 2>/dev/null || true
