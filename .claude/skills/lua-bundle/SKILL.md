---
name: lua-bundle
description: Change the Civ6 mod's Lua safely -- edit the module files, regenerate the bundle the game loads with scripts/build_lua_bundle.py, run the Lua suites with the runtime required, and commit module and bundle together. Use whenever a file under mod/AlphaZeroAI/Scripts/ or the .modinfo changes. Gated by build_lua_bundle.py --check.
allowed-tools: Bash, Read, Edit
---

# Lua Bundle Skill

The game loads one Lua file, `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`. It is
generated: the module files beside it (`ActionExecutor.lua`, `StateExporter.lua`,
`Main.lua` and the rest), concatenated in the order the `.modinfo` lists them.

## Why this exists

The bundle used to be edited by hand and drifted from the modules, so an edit to a
module changed nothing in the game, and two fixes that lived only in the bundle were
lost the first time it was regenerated. The generator and its `--check` exist so that
there is exactly one source: the module files.

## When to use

- Any edit to a `.lua` file under `mod/AlphaZeroAI/Scripts/`.
- Any edit to `mod/AlphaZeroAI/AlphaZeroAI.modinfo` (it fixes the module order).
- A red `check-lua-bundle` pre-commit hook, `make validate` or CI governance job.

## Steps

### 1. Edit the module files, never the bundle

Make the change in the module file that owns the code. **Never edit
`AlphaZeroMain.lua` by hand**: the next regeneration overwrites it, and the
PostToolUse hook and `make validate` report it as out of date.

### 2. Regenerate the bundle

```bash
python scripts/build_lua_bundle.py
python scripts/build_lua_bundle.py --check
```

The first command rewrites the bundle; the second must then exit 0 (exit 1 means the
committed bundle is out of date, 2 a layout error such as a module the `.modinfo`
lists but the directory lacks). `make check-lua-bundle` runs the same check.

### 3. Run the Lua suites with the runtime required

```bash
make install-lua      # once: the dev extra plus lupa (Lua 5.1)
make test-lua         # sets CIV6AZ_REQUIRE_LUA=1
CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/scripts/test_lua_bundle_sync.py -q
```

Without `lupa` the Lua tests skip, and **a skip is not a pass**.
`CIV6AZ_REQUIRE_LUA=1` (set by `make test-lua`, `make verify` and CI's test job)
turns a missing runtime into a failure. The tests run the bundle against the stand-in
game in `tests/fixtures/lua/civ6_stub.lua`, so they prove the Lua does what the
contract says, not that the contract matches the game; live behaviour goes on the
maintainer's checklist in `docs/LIVE_CIV6.md`.

### 4. Commit the module and the bundle together

Stage the edited module files and the regenerated `AlphaZeroMain.lua` in the same
commit. A commit with only one of them fails the `check-lua-bundle` pre-commit hook
and leaves the history with a revision whose bundle does not match its modules.

### 5. Negative controls on Lua

Follow the `negative-control` skill, with one addition: a planted defect in a module
reaches the tests only through the bundle, so back up **both** files before you plant
it. Take the backups in a new temporary directory (`mktemp -d`, as below), print its
full path, and name each backup by its repository path:

```bash
BACKUP_DIR="$(mktemp -d)"; echo "backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR/mod/AlphaZeroAI/Scripts"
cp mod/AlphaZeroAI/Scripts/ActionExecutor.lua "$BACKUP_DIR/mod/AlphaZeroAI/Scripts/"
cp mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua "$BACKUP_DIR/mod/AlphaZeroAI/Scripts/"
# plant the defect in the module, then rebuild so the tests can see it
python scripts/build_lua_bundle.py
```

To restore, copy **both** backups back and compare each with `cmp`; do not rebuild
the bundle, so it too equals its backup byte for byte. Never use `git checkout`,
`git restore` or `git stash` to undo a plant: they put back the committed version and
drop uncommitted work. Finish with `python scripts/build_lua_bundle.py --check`.
