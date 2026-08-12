# `python scripts/run_chess_training.py` fails with `ModuleNotFoundError: No module named 'alpha_zero.utils.checkpoint'` despite the file existing

## Summary

After a clean editable install (`pip install -e .`), running the chess
training entry point as a script fails to discover `alpha_zero.utils.checkpoint`,
even though the file exists in the working tree and a direct import of the same
name from a fresh Python interpreter succeeds. This affects every contributor
who follows the documented invocation `python scripts/run_chess_training.py`,
not just my machine — it's a function of how the editable install registers
submodules combined with how `python <path>/<script>.py` modifies `sys.path`.

## Environment

- Windows 11, PowerShell 5.1
- Python 3.11.x (system install at `C:\Program Files\Python311\`)
- Branch: `main` at commit `9d151f4` (PR #75 merge)
- Install: `pip install -e .` immediately after the merge
- All other package dependencies (torch, chess, openai, yaml, ...) installed cleanly

## Reproduction

```powershell
git checkout main
git pull                      # at commit 9d151f4 or later
pip install -e .

# Direct submodule import works:
python -c "from alpha_zero.utils.checkpoint import safe_load_checkpoint; print('OK')"
# → OK

# But the script invocation fails:
python scripts/run_chess_training.py --config configs/chess_config.yaml --dry-run
```

Expected: dry-run prints "Dry run OK: env=ChessEnv, ..." and exits 0.

Actual:

```
File "C:\...\Civ6-AlphaZero\scripts\run_chess_training.py", line 180, in _run
    from alpha_zero.envs.env_factory import create_game_env
File "C:\...\Civ6-AlphaZero\alpha_zero\envs\__init__.py", line 13, in <module>
    from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env
File "C:\...\Civ6-AlphaZero\alpha_zero\envs\env_factory.py", line 31, in <module>
    from alpha_zero.envs.microciv import MicroCivEnv
File "C:\...\Civ6-AlphaZero\alpha_zero\envs\microciv.py", line 25, in <module>
    from alpha_zero.core.civ6_actions import MicroCivActionSpace
File "C:\...\Civ6-AlphaZero\alpha_zero\core\__init__.py", line 14, in <module>
    from alpha_zero.core.civ6_network import (
File "C:\...\Civ6-AlphaZero\alpha_zero\core\civ6_network.py", line 20, in <module>
    from alpha_zero.utils.checkpoint import safe_load_checkpoint
ModuleNotFoundError: No module named 'alpha_zero.utils.checkpoint'
```

`alpha_zero/utils/checkpoint.py` is present on disk and `alpha_zero.egg-info/SOURCES.txt`
lists it. The same import works when invoked through `python -c "..."` from the
project root.

## Root cause (diagnosis)

Two interacting factors:

1. **`python <path>/<script>.py` puts the script's directory on `sys.path[0]`,
   not the project root.** Running `python scripts/run_chess_training.py`
   makes `sys.path[0] = "scripts"`. Running `python -c "..."` from the project
   root makes `sys.path[0] = ""` (cwd). With cwd on `sys.path`, `alpha_zero/`
   is discoverable as a namespace alongside the editable install; without it,
   only the editable install's finder is used.

2. **`pyproject.toml` has no `[tool.setuptools.packages]` configuration.**
   Modern setuptools editable installs construct a custom MetaPathFinder when
   packages are auto-discovered. With multiple top-level directories in the
   repo (`alpha_zero/`, `tests/`, `scripts/`, `infra/`, `configs/`, `dockerfiles/`,
   `docs/`, `data/`, ...), auto-discovery's behavior is fragile and can register
   only a partial submodule list. The custom finder then fails to resolve
   submodules added after install (or, in this case, submodules whose discovery
   depends on a transitive import path that isn't excited at install time).

The combination means: the editable install knows about `alpha_zero` and most
of its submodules well enough that direct `from alpha_zero.utils.checkpoint import X`
works, but when the same import is reached through the longer chain
`envs.__init__ → env_factory → microciv → core.__init__ → civ6_network`, the
finder fails to locate `checkpoint` as a child of `utils`.

## Workaround (immediate, zero-risk)

```powershell
$env:PYTHONPATH = "."
python scripts/run_chess_training.py --config configs/chess_config.yaml --dry-run
```

Or run as a module (`scripts/__init__.py` already exists, so this works):

```powershell
python -m scripts.run_chess_training --config configs/chess_config.yaml --dry-run
```

Both restore the project root to `sys.path` and let normal package discovery
resolve every submodule correctly.

## Proposed fix (preferred): explicit `[tool.setuptools.packages.find]` in `pyproject.toml`

Add this block to `pyproject.toml` (after the existing `[project]` and
`[project.optional-dependencies]` sections, before `[tool.ruff]`):

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["alpha_zero*"]
exclude = ["tests*", "scripts*", "infra*", "configs*", "data*", "docs*", "dockerfiles*", "kb*", "examples*", "demos*", "checkpoints*", "logs*", "replays*", "mod*", "k8s*", "spacetimedb_module*", "server-module*"]
```

This declares the package set explicitly, removing setuptools' need to guess
and giving the editable-install finder a deterministic submodule list. Verified
to fix the symptom in local testing.

## Alternative fixes (also acceptable)

### Option B — pin `editable_mode=compat` in install instructions

Update `README.md` and any setup docs to say:

```bash
pip install -e . --config-settings editable_mode=compat
```

`editable_mode=compat` reverts to the older `.pth`-file install style that just
adds the project root to `sys.path` and lets normal Python package discovery
do the rest — no custom finder. The cost is being on a setuptools deprecation
path, but it's fully supported today and unambiguously works.

### Option C — `sys.path` shim at the top of `scripts/run_chess_training.py`

```python
# scripts/run_chess_training.py, after the docstring, before any alpha_zero import
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Smallest scope, but doesn't fix the same problem in any other script under
`scripts/` and doesn't help users who do `from alpha_zero...` interactively
from inside `scripts/`. Acceptable as a stopgap, not a real fix.

## Recommendation

Option A (`packages.find` in `pyproject.toml`) — single source of truth, no
runtime overhead, fixes every script and every install mode, and is the
setuptools-recommended approach for non-trivial package layouts.

## Acceptance criteria

- [ ] `pip install -e .` followed by `python scripts/run_chess_training.py --config configs/chess_config.yaml --dry-run` succeeds without `PYTHONPATH=.`
- [ ] `python -c "from alpha_zero.utils.checkpoint import safe_load_checkpoint"` still succeeds
- [ ] All existing tests still pass: `pytest tests/ -v --tb=short`
- [ ] Editable install behavior is documented in README "Quick Start" / Section 8 of `CLAUDE.md`

## References

- Triggered by validating PR #75's LM Studio integration on Windows.
- Related to setuptools' [editable install behavior](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (changed in setuptools 64+).
- Related to PEP 420 namespace package interactions with multiple top-level directories.
