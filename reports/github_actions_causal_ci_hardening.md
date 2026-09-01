# GitHub Actions causal CI hardening

## Executive result

**PARTIAL** — repository changes are in place and locally validated. GitHub Actions confirmation of the Node.js 20 annotation is recorded after the push of this work (see Validation).

- **Node.js 20 warning:** classified as `resolved_by_SHA_pinned_supported_action`. Causal CI previously used `actions/checkout@v4` and `actions/setup-python@v5` (`runs.using: node20`). Those are upgraded to checkout v7.0.1 and setup-python v7.0.0 (`runs.using: node24`) and pinned to verified commit SHAs. `actions/setup-node` was not added; this repository does not run Node/npm tooling.
- **Causal CI still runs** the same command: `make test-causal PYTHON=python3`.
- **Causal logic / test thresholds / fixtures / expected values:** unchanged.
- **Workflow permissions:** verified and narrowed to `contents: read` (wheel job additionally needs `actions: write` for artifact upload; documented below).
- **Official actions:** upgraded where needed for Node 24 and pinned to 40-character SHAs with version comments.
- **Dependency caching:** none on the causal job (no lockfile for the mypy install). A cache miss/hit therefore cannot change the tested environment.

Do not treat this report as **COMPLETE** until the post-change GitHub Actions run is recorded in Validation and the Node annotation comparison is filled in.

## Baseline

### 1.1 Workflow inventory

No reusable workflow calls and no repository composite actions (`action.yml`) exist. Three workflows:

| Workflow file | Workflow name | Triggers | Jobs | Runner | Actions used (before) | Runtime/dependency setup | Permissions (before) | Caches | Test/build commands |
|---|---|---|---|---|---|---|---|---|---|
| `.github/workflows/causal.yml` | Causal runtime | `push`, `pull_request` (all branches) | `test-causal` | `ubuntu-latest` | `actions/checkout@v4`, `actions/setup-python@v5` | Python 3.11 via setup-python; `pip install mypy` (unpinned) | default GITHUB_TOKEN (unrestricted) | none | `make test-causal PYTHON=python3` |
| `.github/workflows/wheel-macos-arm64.yml` | Wheel macOS arm64 | `workflow_dispatch`; `push` to `main` with path filters | `wheel` | `macos-14` | `actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, `actions/setup-python@v5`, `pypa/cibuildwheel@v2.22.0`, `actions/upload-artifact@v4` | Python 3.11; cargo release daemon; cibuildwheel `cp311-*` arm64 | default GITHUB_TOKEN | none | cargo build, cibuildwheel, upload wheel |
| `.github/workflows/ironclad-burn-in.yml` | Ironclad burn-in | `workflow_dispatch` only | `ironclad` | `macos-14` | `actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, `actions/setup-python@v5` | venv + requirements-spike/server; `make python`; `make daemon-load`; `ci_ironclad_verify.sh` | default GITHUB_TOKEN | none | doctor + burn-in verify |

### 1.2 Causal workflow trace (before)

- **Path / name:** `.github/workflows/causal.yml` / `Causal runtime`
- **Triggers:** every `push` and `pull_request` (no branch filter)
- **Jobs / deps:** single job `test-causal` (no `needs:`)
- **Runner:** `ubuntu-latest` (GitHub-hosted; not pinned)
- **Python:** `python-version: "3.11"` (explicit; matches `crates/hiveclaw-python/pyproject.toml` `requires-python = ">=3.11"`). No `.python-version` file. No causal lockfile.
- **Install:** `python -m pip install --upgrade pip mypy`
- **Cache:** none
- **Test command:** `make test-causal PYTHON=python3` which runs `unittest discover -s tests -p 'test_hiveclaw_causal_*.py'` then mypy on `hiveclaw_causal` and those tests
- **Network:** mypy install uses PyPI. Tests themselves are CPU/SQLite, frozen fixtures, no live network, no secrets, no GPU/daemon.
- **Randomness / locale / system libs:** Rewind tests use explicit seeds; no locale pinning required; no extra system packages
- **Ignored failures:** none (`continue-on-error` absent)

### 1.3 Baseline status (verified via GitHub API, not assumed)

```
baseline commit SHA: 2194fc54a3aa22e9d0a577ce4d5caad7e690c4f1 (HEAD of main at start of this work)
screenshot commit:   368b3304223585feb8d7748cd404f094e17bf488 (still present; not HEAD)
branch:              main
latest causal run:   https://github.com/wijeratne-a/HiveClaw/actions/runs/33479229489
run ID:              33479229489
job:                 test-causal
job status:          success
job duration:        2026-09-01T06:48:48Z → 2026-09-01T06:49:00Z (~12 s)
screenshot run:      https://github.com/wijeratne-a/HiveClaw/actions/runs/33478599893 (368b330, success, ~12 s)
current warning:     Node.js 20 is deprecated. The following actions target Node.js 20 but are being forced to run on Node.js 24: actions/checkout@v4, actions/setup-python@v5. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
current test command: make test-causal PYTHON=python3
current test outcome: success (step "make test-causal")
```

Annotation fetched from check-run `99763010423` on run `33478599893`.

## Root-cause analysis

### Node version distinction

| Runtime | Role in this repo |
|---|---|
| **JavaScript action internal runtime** (`runs.using` in the action’s `action.yml`) | What GitHub warned about. `checkout@v4` and `setup-python@v5` declare `node20`. GitHub hosted runners force those actions onto Node 24 and emit the deprecation annotation. |
| **Project Node** | None. HiveClaw CI does not install Node, npm, pnpm, or yarn. Adding `actions/setup-node` would not change another action’s internal runtime. |
| **Runner preinstalled Node** | Present on GitHub-hosted images; unused by causal tests. |
| **Forced compatibility runtime** | Documented by GitHub: Node 20 actions are deprecated and forced onto Node 24; Node 20 removal is scheduled (changelog 2025-09-19). This is platform-managed *unless* the workflow still references node20 actions. |

### Official action versions inspected

Verified via GitHub Releases API and raw `action.yml` at the release commit:

| Action | Before | Selected release | Commit SHA (40 hex) | `runs.using` |
|---|---|---|---|---|
| actions/checkout | `@v4` (floating major; node20) | **v7.0.1** (latest stable release at inspection) | `3d3c42e5aac5ba805825da76410c181273ba90b1` | `node24` |
| actions/setup-python | `@v5` (floating major; node20) | **v7.0.0** (latest stable release at inspection) | `5fda3b95a4ea91299a34e894583c3862153e4b97` | `node24` |
| actions/upload-artifact | `@v4` (wheel only) | **v7.0.1** | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` | `node24` |
| pypa/cibuildwheel | `@v2.22.0` | **v2.22.0 kept** (no major bump) | `ee63bf16da6cddfb925f542f2c7b59ad50e93969` | composite |
| dtolnay/rust-toolchain | `@stable` (moving tag) | master HEAD at inspection | `6c977a6ca4077a0ceb28ffbe03f59d46e9ac8772` | composite; `toolchain: stable` |

Sources:

- https://github.com/actions/checkout/releases/tag/v7.0.1
- https://github.com/actions/setup-python/releases/tag/v7.0.0
- https://github.com/actions/upload-artifact/releases/tag/v7.0.1
- https://raw.githubusercontent.com/actions/checkout/v4/action.yml (`using: node20`)
- https://raw.githubusercontent.com/actions/checkout/v7.0.1/action.yml (`using: node24`)
- https://raw.githubusercontent.com/actions/setup-python/v5/action.yml (`using: node20`)
- https://raw.githubusercontent.com/actions/setup-python/v7.0.0/action.yml (`using: node24`)
- https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/

Checkout v7 still supports `persist-credentials` and `fetch-depth`. Setup-python v7 still supports `python-version: "3.11"`. No `pip-install` input is used (removed in v7; unused here).

### Verdict

```
resolved_by_SHA_pinned_supported_action
```

Upgrading to the current official node24 majors and pinning SHAs is the repository-owned fix. If GitHub still annotates after that upgrade, treat the leftover as `platform_managed_warning_no_repo_change_available` rather than swapping in third-party checkout/python actions.

**Residual (non-causal):** `pypa/cibuildwheel@v2.22.0` is a composite that still `uses: actions/setup-python@v5` internally. Bumping cibuildwheel to v3/v4 was rejected to avoid changing wheel-build behavior. The wheel job may therefore still see a nested Node 20 notice; that does not affect `test-causal`.

## Changes

| file | change | reason | security/reliability effect | causal-test behavior effect |
|---|---|---|---|---|
| `.github/workflows/causal.yml` | Pin checkout v7.0.1 + setup-python v7.0.0 SHAs; `permissions: contents: read`; `persist-credentials: false`; `fetch-depth: 1`; pin runner `ubuntu-24.04`; echo python/pip/mypy versions; keep `make test-causal PYTHON=python3` | Remove node20 action runtime; least privilege; reproducible runner/Python | Smaller token surface; immutable action refs; explicit Python 3.11 | **none** — same make target, same unittest glob, same mypy scope |
| `.github/workflows/wheel-macos-arm64.yml` | Same official action pins; pin rust-toolchain SHA with `toolchain: stable`; pin cibuildwheel v2.22.0 SHA; pin upload-artifact v7.0.1; `contents: read` + job `actions: write`; persist-credentials false | Durable pins without bumping cibuildwheel major | Artifact upload still works under restricted default token | **none** (not the causal job) |
| `.github/workflows/ironclad-burn-in.yml` | Same checkout/python/rust pins and `contents: read` | Same supply-chain policy on the manual burn-in workflow | Least privilege | **none** |
| `tests/test_hiveclaw_causal_ci_policy.py` | Offline policy + causal-command integrity tests | Prevent silent weakening of CI (unpinned actions, continue-on-error, missing `make test-causal`) | Fails closed on policy regressions | **none** to causal math; adds CI-policy unittests under the existing `test_hiveclaw_causal_*.py` glob |
| `reports/github_actions_causal_ci_hardening.md` | This report | Operator record | none | none |

No changes to `hiveclaw_causal/`, causal fixtures, thresholds, or Makefile test body (the discover pattern is unchanged).

## Validation

### Local static checks

- Ruby `YAML.load_file` on all three workflow files: OK
- `actionlint` 1.7.12 on `.github/workflows/*.yml`: **exit 0, no findings**
- SHA scan: every `uses:` pin is exactly 40 lowercase hex characters and matches the inspected upstream release/commit
- Grep: no `continue-on-error`, no `curl … \| sh` in workflow files, no `echo` of `secrets.*`, no `write-all`

Policy-test limitation: workflows are inspected with stdlib regex/line parsing (no PyYAML in the causal environment). There are no reusable workflow calls or composite actions in this repo; the test reports that limitation by asserting those patterns are absent.

### Local causal tests

```
command:              make test-causal PYTHON=/Users/wijeratne/dev/HiveClaw/.venv/bin/python3
Python version:       3.11.1 (project venv)
dependency tool:      pip 22.3.1; mypy 2.3.1 (installed into the venv for typecheck only; not a production dependency pin)
test count:           26 tests, 0 skipped
passed/failed/skipped: OK / 0 failed / 0 skipped
runtime:              unittest 4.293s; make wall ~9.06s
warnings:             none from unittest/mypy; pip printed a version-check warning during mypy install
mypy:                 Success: no issues found in 22 source files
```

Same pass/fail semantics as CI: unittest discover of `test_hiveclaw_causal_*.py` then mypy. The extra tests are the CI policy guards (9), not weakened causal assertions.

### GitHub Actions validation

Filled in after push of the hardening commit:

```
run URL/ID:     (pending)
commit SHA:     (pending)
job result:     (pending)
duration:       (pending)
annotations:    (pending comparison vs Node 20 warning on 33478599893)
```

## Workflow security posture

- **Permissions:** top-level `permissions: contents: read` on all three workflows. Exception: wheel job also sets `actions: write` so `upload-artifact` can store the macOS wheel after the default token is restricted. No `id-token`, packages, or pull-request write.
- **Action pinning policy:** every `uses:` in `.github/workflows/*.yml` must be `owner/repo@<40-hex-sha>` with the human version in a comment. Enforced by `tests/test_hiveclaw_causal_ci_policy.py`. Nested actions inside third-party composites (cibuildwheel’s internal setup-python@v5) are outside that file-level pin.
- **Checkout credentials:** `persist-credentials: false`; `fetch-depth: 1` (causal tests do not need git history).
- **Dependency pinning/caching:** causal job installs mypy from PyPI with no lockfile and **no cache**. Cache hit/miss cannot substitute an incompatible dependency set because nothing is restored. Residual: mypy version floats with PyPI (pre-existing; not introduced here).
- **Secrets/logging:** workflows print `python3 --version`, `pip --version`, and `mypy --version` only. No `env:` dumps, no tokens.

## Residual risks

- GitHub-hosted runners may still force internal JS runtimes independently of YAML. If a future runner change reintroduces a Node warning on already-node24 official actions, that is platform-managed.
- SHA pinning requires a deliberate bump when GitHub publishes security fixes to checkout/setup-python/upload-artifact.
- `pypa/cibuildwheel@v2.22.0` still nests `actions/setup-python@v5` (node20). Wheel workflow only.
- `dtolnay/rust-toolchain` composite may `curl | sh` rustup when rustup is missing (GitHub-hosted macOS images already have rustup). Not used by causal CI.
- Unpinned `mypy` on PyPI can change typecheck strictness over time. Not changed in this work; a mypy pin would be a separate, tested decision.
- A passing unit suite does not prove causal validity beyond the declared assertions in `tests/test_hiveclaw_causal_*.py`.
- Policy tests parse workflow YAML structurally but not as a full GitHub Actions schema.

## Operator guidance

### Run causal tests locally

```bash
python3 -m pip install mypy   # once, if needed
make test-causal PYTHON="$(pwd)/.venv/bin/python3"
```

Requires Python 3.11+ (CI uses 3.11). No daemon, GPU, or network after mypy is installed.

### Inspect the CI job

1. Open https://github.com/wijeratne-a/HiveClaw/actions/workflows/causal.yml
2. Open the run for the commit SHA.
3. Confirm job `test-causal` and step `make test-causal` succeeded (not merely checkout/setup).
4. Check the Annotations pane for Node runtime warnings.

### Update an action safely

1. Read the official release notes; confirm `runs.using` is a supported Node (currently `node24`).
2. Resolve the release tag to a 40-character commit SHA (`git/ref/tags/vX.Y.Z`, peel annotated tags).
3. Replace the `uses:` SHA and the version comment.
4. Run `actionlint .github/workflows/*.yml` and `make test-causal`.
5. Do not switch to an unverified third-party action to silence a GitHub runtime warning.

### If GitHub warns about an action runtime again

1. Identify whether the listed `uses:` still points at a node20 (or older) `action.yml`.
2. If a newer official major declares `node24`, pin that SHA.
3. If the official action is already on the supported runtime, record `platform_managed_warning_no_repo_change_available` and do not replace GitHub actions with unknown publishers.
4. Do not set `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION` or `continue-on-error` to hide the warning.
