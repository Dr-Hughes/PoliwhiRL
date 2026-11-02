# Getting started

What this page is for: the shortest path from a clean checkout to a running
training session, a recorded inference run, and a passing test suite — the
prerequisites, install, ROM placement, first run, where artefacts land, and how
to run the tests. Read [configuration](configuration.md) for what the flags
mean and [operations/run-triage](operations/run-triage.md) for what to do with
the output.

**Sources**: `pyproject.toml`, `pytest.ini`, `.github/workflows/run_tests.yml`,
`main.py`, `configs/default_configs/*.json`, `configs/stages/open_world.json`,
`configs/inference.json`.

## Prerequisites

| Requirement | Value | Source |
|---|---|---|
| Python | `>=3.14` | `pyproject.toml:5` |
| Emulator binding | `pyboy==2.7.1` (pinned) | `pyproject.toml:12` |
| Torch | `torch>=2.8.0` | `pyproject.toml:15` |
| Core dependencies | 11 pinned/ranged deps (gymnasium, matplotlib, numpy, Pillow, psutil, pyboy, pysdl2-dll, PySDL2, torch, torchviz, tqdm) | `pyproject.toml:6-18` |
| Dev extra | black, flake8, pre-commit, pytest | `pyproject.toml:20-26` |
| Package installable? | **No** — `[tool.uv] package = false`; run from the repo root | `pyproject.toml:31-32` |
| Console script | `poliwhirl = main:main` | `pyproject.toml:28-29` |

The project is uv-managed. Because `package = false`, nothing is built or
installed into site-packages — the repo root *is* the working directory, and
`main.py` is invoked directly.

## Install

```bash
git clone <this repo> && cd PoliwhiRL
uv sync --extra dev          # runtime + dev tools (black, flake8, pre-commit, pytest)
uv run python main.py --help # sanity check that the interpreter resolves
```

`uv.lock`-driven installs satisfy `pyproject.toml:6-18`. `requirements.txt`
also exists as a flat mirror of the dependency list (13 lines: the 11 runtime
deps plus `pre-commit>=4.6.0` and `pytest>=8.4.1`); it is what CI installs
from (`requirements.txt:1-13`, `.github/workflows/run_tests.yml:28`). It omits
black and flake8, so use uv for local work.

## ROM and save-state placement

The emulator is driven headless against a specific ROM and an initial save
state, both resolved by relative path from the repo root
(`configs/default_configs/rom_settings.json:2-3`):

```
emu_files/Pokemon - Crystal Version.gbc
emu_files/states/start.state
```

Plus two peripheral files that PyBoy writes next to the ROM
(`configs/default_configs/rom_settings.json:4-7`):

```
emu_files/Pokemon - Crystal Version.gbc.ram
emu_files/Pokemon - Crystal Version.gbc.rtc
```

These are **not** in version control. CI keeps encrypted copies under
`emu_files_encrypted/*.gpg` and decrypts them into `emu_files/` at run time
using the `FILE_DECRYPTION_KEY` secret
(`.github/workflows/run_tests.yml:31-40`), then wipes them
(`.github/workflows/run_tests.yml:45-47`). Locally you must supply the ROM and
`start.state` yourself; every mode fails at emulator start without them.

## First run

The canonical from-scratch configuration is `configs/stages/open_world.json`:
it extends the shared curriculum parent (`configs/stages/open_world.json:2`),
explicitly sets `load_checkpoint: ""` so nothing is auto-resumed
(`configs/stages/open_world.json:5`), and sets the long open-world horizons
`num_rollouts: 163840` / `episode_length: 24576`
(`configs/stages/open_world.json:13-14`).

```bash
python main.py --use_config configs/stages/open_world.json
```

The entry flag is `--use_config`, **not** `--config` (`main.py:104-110`).
`model` defaults to `"PPO"` (`configs/default_configs/core_settings.json:2`),
which dispatches to `setup_and_train_PPO()` (`main.py:184-185`).

To wipe previous artefacts before starting (`erase` defaults to `false`,
`configs/default_configs/core_settings.json:3`; the flag accepts
`yes|true|t|1`, `main.py:15-17`):

```bash
python main.py --use_config configs/stages/open_world.json --erase yes
```

To run a bounded session rather than the full configured budget, use
`--add_rollouts` (exact unit) or `--add_episodes` (converted to rollouts via
`episode_length / (ppo_update_frequency * num_envs)`) — both are additive on
top of what an auto-resumed checkpoint already completed
(`main.py:136-152`):

```bash
python main.py --use_config configs/stages/open_world.json --add_rollouts 512
```

## What gets written where

Everything lands under `output_base_dir`, which defaults to
`./Training Outputs/` (`configs/default_configs/outputs_settings.json:2`). The
sibling keys in that file are bare folder names
(`configs/default_configs/outputs_settings.json:3-7`) and get the base dir
prepended at load time because the filename contains `"outputs"`
(`main.py:29-30`, `main.py:38-41`):

| Key | Default (pre-remap) | Resolved for the `open_world` run |
|---|---|---|
| `checkpoint` | `Checkpoints` | `./Training Outputs/open_world/Checkpoints` |
| `results_dir` | `Results` | `./Training Outputs/open_world/Results` |
| `record_path` | `Runs` | `./Training Outputs/open_world/Runs` |
| `export_state_loc` | `Training States` | `./Training Outputs/open_world/Training States` |
| `snapshot_dir` | `Snapshots` | `./Training Outputs/open_world/Snapshots` |

The second column is the value produced by `modify_outputs_settings()`
(`main.py:38-41`); the third follows from `merge_configs()` re-pointing every
base-relative default when a config overrides `output_base_dir`
(`main.py:84-96`; the run's base dir is
`./Training Outputs/open_world/`, `configs/stages/open_world.json:3`). This
remap deliberately skips `output_base_dir` itself and any key whose merged
value is no longer base-relative, so input paths such as `load_checkpoint` are
never clobbered (`main.py:89-96`).

Artefacts to expect: `Checkpoints/info.pth`,
`Results/metrics/training_metrics.json`, and first-honest replays under
`Runs/checkpoints/<flag>-<title>/` — see
[operations/run-triage](operations/run-triage.md).

## Inference on a trained checkpoint

```bash
python main.py --use_config configs/inference.json
```

`model: "inference"` dispatches to `run_inference()` (`main.py:190-191`). The
config loads `./Training Outputs/open_world/Checkpoints/best`
(`configs/inference.json:3`), runs autoregressively with `sequence_length: 1`
(`configs/inference.json:4`), `episode_length: 8192` and
`num_inference_runs: 1` (`configs/inference.json:5-6`), and records output
(`configs/inference.json:14`). Point `--load_checkpoint` at any other saved
checkpoint to evaluate it instead.

Other modes, with their minimal CLI, are in the dispatch table in
[configuration](configuration.md).

## Running the tests

```bash
uv run pytest                       # whole suite
uv run pytest tests/test_extends_config.py   # a single file
```

Both `pytest.ini` and `[tool.pytest.ini_options]` put the repo root on
`pythonpath` (`pytest.ini:2`, `pyproject.toml:34-35`), so tests import `main`
and `PoliwhiRL` without an install step. There are no custom pytest markers,
so selection is filename-based — see
[operations/testing](operations/testing.md). Some tests boot PyBoy and need the
ROM files above to be present.

## Known gaps

- `requirements.txt` was read and mirrors the runtime deps plus
  `pre-commit`/`pytest`, but it is not verified to be kept in lock-step with
  `pyproject.toml` by any automation.
- `emu_files/states/start.state` is assumed to be the checkpoint of the game's
  canonical opening; its provenance is not recorded in this tree.
- Whether `poliwhirl` (the console script, `pyproject.toml:28-29`) works under
  `[tool.uv] package = false` was not exercised — the documented path is
  `python main.py`.
- The GPU/MPS expectation is implicit: `device` defaults to `"mps"`
  (`configs/default_configs/core_settings.json:4`); passing
  `--device cuda`/`cpu` was not verified here.
