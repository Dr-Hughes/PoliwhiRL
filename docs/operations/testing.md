# Testing

What this page is for: how the test suite is laid out, how to select tests
given that there are no custom markers, which tests need the emulator and ROM
files, what CI actually does, and the gap between the documented development
environment and the CI environment.

**Sources**: `pytest.ini`, `pyproject.toml`, `.github/workflows/run_tests.yml`,
`tests/` (listing), `requirements.txt`.

## Suite layout

Flat: 36 files `tests/test_*.py`, no subdirectories and no project-level
`conftest.py` (the only ones on disk sit inside `.venv/`, third-party). The
suite therefore relies on `pythonpath` for imports.

```bash
uv run pytest                                        # everything
uv run pytest tests/test_extends_config.py        # one file
uv run pytest tests/test_world_map.py::test_x     # one test
```

Both pytest configuration files put the repo root on the import path
(`pytest.ini:2`, `pyproject.toml:34-35`), which is what lets tests import
`main` and `PoliwhiRL` without an install step. Both also suppress the same
warning: `ignore:Using SDL2 binaries from pysdl2-dll 2.28.5:UserWarning`
(`pytest.ini:3-4`, `pyproject.toml:36-38`).

## No custom markers

There are **no registered pytest markers**: `pytest.ini` contains only
`pythonpath` and `filterwarnings` (`pytest.ini:1-4`), and
`[tool.pytest.ini_options]` in `pyproject.toml` likewise contains only those
two keys (`pyproject.toml:34-38`) — no `markers`, no `addopts`.

Consequence: **selection is filename-based**. Use `-k <expression>` or pass
paths; do not write `pytest -m emulator`, which would collect nothing and warn
about an unregistered marker.

> **One skip guard does exist.** Although no *pytest* markers are registered, there is
> a single `unittest` skip guard in the suite: `tests/test_collision_field.py:26`
> decorates the class with `@unittest.skipUnless(_have_rom(), "ROM/save-state not
> available")`, where `_have_rom()` (`tests/test_collision_field.py:22-23`) checks for
> `./emu_files/Pokemon - Crystal Version.gbc` and `./emu_files/states/start.state`
> (`tests/test_collision_field.py:18-19`). Without a ROM that test **self-skips**, so
> a green `pytest` run on a machine with no ROM does not prove the collision field
> works — check the skipped count, not just the pass count.

## Emulator/ROM-backed subset (by naming)

These nine files are named after emulator-, ROM- or save-state-backed
subsystems, so they are the ones most likely to require
`emu_files/Pokemon - Crystal Version.gbc` and
`emu_files/states/start.state` (see
`configs/default_configs/rom_settings.json:2-3`):

| File | Inferred subject |
|---|---|
| `tests/test_pyboy_ram.py` | RAM decoding through a live PyBoy |
| `tests/test_pyboyenv.py` | the Gym environment wrapper |
| `tests/test_collision_field.py` | runtime collision field |
| `tests/test_checkpoint_progress.py` | checkpoint progress accounting |
| `tests/test_snapshot_seed_state.py` | snapshot seeding state |
| `tests/test_regression_probe.py` | regression probes |
| `tests/test_debug_eval.py` | `debug_eval` mode |
| `tests/test_replay_seeding.py` | replay seeding |
| `tests/test_replay_pool_loading.py` | replay/frontier pool loading |

> **This split is inferred from filenames only.** It was not verified by
> executing the suite or by reading each file's imports. Some of the above may
> stub PyBoy; some files *not* listed (for example
> `tests/test_snapshot_seed_state.py` neighbours such as
> `tests/test_stagnation_truncation.py`) may equally touch the emulator. Treat
> the list as a starting hypothesis for "which tests need ROM files", not as
> fact.

CI covers this by decrypting the ROM artefacts before the run
(`.github/workflows/run_tests.yml:31-40`).

## CI

Single workflow: `.github/workflows/run_tests.yml`, job `test` ("Run Tests") on
`ubuntu-latest` (`run_tests.yml:13-16`), triggered on push and pull request to
`main` and `develop` (`run_tests.yml:3-11`).

| Step | Action | Source |
|---|---|---|
| Checkout | `actions/checkout@v2` | `run_tests.yml:18` |
| Python | `actions/setup-python@v2` with `python-version: '3.x'` | `run_tests.yml:20-23` |
| Install | `pip install --upgrade pip`, `pip install -r requirements.txt`, `sudo apt-get install -y gpg` | `run_tests.yml:25-29` |
| Decrypt | `gpg --decrypt` each `emu_files_encrypted/*.gpg` into `emu_files/`, passphrase from `secrets.FILE_DECRYPTION_KEY` | `run_tests.yml:31-40` |
| Test | bare `pytest` | `run_tests.yml:42-43` |
| Cleanup | `rm -rf emu_files/*`, `if: always()` | `run_tests.yml:45-47` |

There is **no dependency caching** (no `cache:` input on setup-python, no
actions/cache step) — every run reinstalls from scratch (`run_tests.yml:20-29`).
The bare `pytest` invocation is governed by `pytest.ini` / `[tool.pytest.ini_options]`
discovered at the workspace root, which is also where `pythonpath` comes from —
that is why the job needs no install step.

Do not transcribe the value of `FILE_DECRYPTION_KEY`; it lives only as a
GitHub Actions secret referenced at `run_tests.yml:40`.

## Local-vs-CI environment gap

Worth flagging, and verified in-tree:

| Aspect | CI | Documented development environment |
|---|---|---|
| Installer | `pip install -r requirements.txt` (`run_tests.yml:28`) | uv lockfile (`[tool.uv] package = false`, `pyproject.toml:31-32`) |
| Python version | `'3.x'` — whatever the runner's default resolves to (`run_tests.yml:23`) | `requires-python = ">=3.14"` (`pyproject.toml:5`) |
| Dependency source | `requirements.txt`, 13 pinned/ranged entries, no black or flake8 (`requirements.txt:1-13`) | `pyproject.toml:6-18` plus the `dev` extra (`pyproject.toml:20-26`) |
| Formatting/lint gates | none in the workflow | black (`line-length = 100`, `pyproject.toml:40-41`), flake8 (`pyproject.toml:43`), pre-commit in the dev extra (`pyproject.toml:24`) |

**CI is therefore not reproducing the documented development environment.** The
most consequential line is `python-version: '3.x'` against
`requires-python = ">=3.14"`: pip is free to install into a Python the project
does not declare compatible, and a green CI run is not evidence the suite passes
on 3.14. Likewise, because CI installs `requirements.txt` rather than
`pyproject.toml`, the two dependency lists can drift apart unnoticed.

## Known gaps

- The unit-versus-integration split above is filename-inferred, not verified.
- The exact Python version CI resolves from `'3.x'` at run time is unknown from
  the workflow file alone (`run_tests.yml:23`).
- `requirements.txt` contents are now known (13 lines, mirroring
  `pyproject.toml:6-18` plus `pre-commit` and `pytest`), but no automation
  keeping it in step with `pyproject.toml` was found.
- Test counts, durations and pass/fail status were not obtained: this pass ran
  no tests, so no per-file coverage claim is made.
- Whether `actions/checkout@v2` and `setup-python@v2` (both deprecated majors)
  still work on current `ubuntu-latest` runners is unverified
  (`run_tests.yml:18,21`).
- Which of the remaining 27 test files are pure unit tests was not determined;
  only the nine emulator-named files were singled out, and no test file
  contents were read in this pass.
