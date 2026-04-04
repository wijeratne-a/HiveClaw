# HiveClaw — Phase 4: XPC Mach service + IOSurface + PyO3.
.PHONY: build build-release poc clean test-ipc python python-clean python-check-maturin spike-deps daemon-load daemon-unload daemon-uninstall daemon-status

ROOT := $(abspath .)
PBIN := $(ROOT)/target/release/pheromoned
GEN_PLIST := $(ROOT)/com.hiveclaw.pheromoned.gen.plist
# launchctl often returns EIO (error 5) when bootstrapping a plist from an arbitrary repo path; install under ~/Library/LaunchAgents/ (Apple convention).
LAUNCH_AGENTS := $(HOME)/Library/LaunchAgents
INSTALLED_PLIST := $(LAUNCH_AGENTS)/com.hiveclaw.pheromoned.plist
# launchd GUI domain for the user running `make` (set at Make parse time).
HIVECLAW_GUI_DOMAIN := gui/$(shell id -u)
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
	-launchctl bootout $(HIVECLAW_GUI_DOMAIN) com.hiveclaw.pheromoned 2>/dev/null || true

test-ipc:
	# Single thread: all integration tests register the same Mach label `com.hiveclaw.pheromoned`.
	cargo test -p hiveclaw-daemon -- --test-threads=1

# MLX / numpy / psutil for scripts/intelligence_spike.py (idempotent).
spike-deps:
	"$(MATURIN_PYTHON)" -m pip install -r "$(ROOT)/scripts/requirements-spike.txt"

# Fail fast if repo .venv exists but PYTHON points elsewhere — maturin walks parent dirs and
# would still discover .venv, mixing PyO3/mlx builds (wrong arch or CPython version).
python-check-maturin:
	@if [ -f "$(ROOT)/.venv/pyvenv.cfg" ]; then \
	  _venvpy=""; \
	  for _cand in "$(ROOT)/.venv/bin/python3" "$(ROOT)/.venv/bin/python"; do \
	    if [ -x "$$_cand" ]; then _venvpy="$$_cand"; break; fi; \
	  done; \
	  if [ -n "$$_venvpy" ]; then \
	  M="$$( "$(MATURIN_PYTHON)" -c 'import sys; print(sys.executable)' 2>/dev/null )"; \
	  V="$$( "$$_venvpy" -c 'import sys; print(sys.executable)' 2>/dev/null )"; \
	  if [ -n "$$M" ] && [ -n "$$V" ] && [ "$$M" != "$$V" ]; then \
	    echo >&2 ""; \
	    echo >&2 "HiveClaw: PYTHON ($$M) does not match repo .venv ($$V)."; \
	    echo >&2 "maturin discovers $(ROOT)/.venv and may build for the wrong interpreter (std::bad_cast, wrong arch)."; \
	    echo >&2 "Run:  make python PYTHON=$(ROOT)/.venv/bin/python3"; \
	    echo >&2 "Or rename/remove $(ROOT)/.venv if you intentionally use another interpreter."; \
	    echo >&2 ""; \
	    exit 1; \
	  fi; \
	  fi; \
	fi

# Remove native extension artifacts (optional deep clean before make python). Does not delete the whole target/.
python-clean:
	rm -rf "$(ROOT)/crates/hiveclaw-mlx/build"
	rm -f "$(ROOT)/crates/hiveclaw-mlx"/hiveclaw_mlx_ext*.so "$(ROOT)/crates/hiveclaw-mlx"/hiveclaw_mlx_ext*.dylib
	rm -f "$(ROOT)/crates/hiveclaw-python/python/hiveclaw_python"/hiveclaw_mlx_ext*.so
	cargo clean -p hiveclaw-python 2>/dev/null || true

# PyO3/maturin: pin interpreter; force CARGO_TARGET_DIR to this repo's target/ so ~/.cargo/config.toml build.target-dir cannot pull another checkout's PyO3 cache.
# When PYTHON is under a venv, export VIRTUAL_ENV so maturin uses that env (not a different .venv in the repo).
# `make python` compiles/installs extensions only — it does not run MLX integration tests. Avoid running
# MLX slab/LLM scripts (integration_test, test_batched_steering, spikes) while the harvester holds the GPU.
python: spike-deps python-check-maturin
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
	-launchctl bootout $(HIVECLAW_GUI_DOMAIN) com.hiveclaw.pheromoned 2>/dev/null || true
	-launchctl bootout $(HIVECLAW_GUI_DOMAIN) "$(INSTALLED_PLIST)" 2>/dev/null || true
	@sleep 0.5
	@echo "Bootstrapping $(HIVECLAW_GUI_DOMAIN)..."
	plutil -lint "$(INSTALLED_PLIST)" >/dev/null || (echo >&2 "Invalid plist: $(INSTALLED_PLIST)"; exit 1)
	@if launchctl bootstrap $(HIVECLAW_GUI_DOMAIN) "$(INSTALLED_PLIST)"; then \
		echo "OK: com.hiveclaw.pheromoned loaded. Check: make daemon-status"; \
	else \
		_svc="$(HIVECLAW_GUI_DOMAIN)/com.hiveclaw.pheromoned"; \
		if launchctl print "$$_svc" 2>/dev/null | grep -q "state = running" \
			&& launchctl print "$$_svc" 2>/dev/null | grep -Fq "$(PBIN)"; then \
			echo "Note: launchctl bootstrap failed (often EIO 5), but the job is already running with this binary — OK."; \
			echo "OK: com.hiveclaw.pheromoned. Check: make daemon-status"; \
		else \
			echo >&2 ""; \
			echo >&2 "launchctl bootstrap failed and com.hiveclaw.pheromoned is not running."; \
			echo >&2 "Domain: $(HIVECLAW_GUI_DOMAIN)  plist: $(INSTALLED_PLIST)"; \
			echo >&2 "If you see EIO 5 from an IDE terminal, run: cd $(ROOT) && make daemon-load from Terminal.app."; \
			echo >&2 "See scripts/README.md -> \"Bootstrap failed: 5\"."; \
			echo >&2 ""; \
			exit 1; \
		fi; \
	fi

daemon-status:
	@launchctl print "$(HIVECLAW_GUI_DOMAIN)/com.hiveclaw.pheromoned" 2>/dev/null || echo "(not loaded in $(HIVECLAW_GUI_DOMAIN) — run make daemon-load from Terminal.app)"

daemon-unload:
	-launchctl bootout $(HIVECLAW_GUI_DOMAIN) com.hiveclaw.pheromoned 2>/dev/null || true

daemon-uninstall: daemon-unload
	rm -f "$(INSTALLED_PLIST)"
