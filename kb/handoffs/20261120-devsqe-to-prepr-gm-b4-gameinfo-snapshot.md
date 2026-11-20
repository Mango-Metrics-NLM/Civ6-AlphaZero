# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-20
**Spec:** kb/specs/20261119-game-mechanics-plan.md (slice GM-B4; OpenSpec `openspec/changes/add-bridge-research-parity`, tasks 4.1, 4.2)
**Branch:** `claude/gm-b4-gameinfo-snapshot`, on the default branch after #143, #146, #147, #148, #149, #150 and #151 merged (not stacked on anything open)
**Files Changed:**
- `mod/AlphaZeroAI/Scripts/GameInfoExporter.lua`
- `mod/AlphaZeroAI/Scripts/FileIO.lua`
- `mod/AlphaZeroAI/Scripts/Config.lua`
- `mod/AlphaZeroAI/Scripts/Main.lua`
- `mod/AlphaZeroAI/Scripts/JsonEncoder.lua`
- `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`
- `mod/AlphaZeroAI/AlphaZeroAI.modinfo`
- `alpha_zero/bridge/gameinfo_capture.py`
- `alpha_zero/bridge/Agent.md`
- `tests/fixtures/lua/civ6_stub.lua`
- `tests/fixtures/gameinfo_snapshot.py`
- `docs/LIVE_CIV6.md`
- `docs/MODULE_REFERENCE.md`
- `CHANGELOG.md`
- `CLAUDE.md`
- `openspec/changes/add-bridge-research-parity/tasks.md`

**Tests Added:** `tests/unit/test_lua_gameinfo_export.py` (60), `tests/unit/test_bridge_gameinfo_capture.py` (75), `tests/unit/test_bridge_gameinfo_capture_review.py` (71), `tests/unit/test_lua_json_encoder.py` (144)
**Test Results:** 350 pass in those four files; the 631 tests that execute the mod's Lua pass. `tests/unit` + `tests/integration` (`-m "not slow"`): 12,359 passed, 1 failed (`test_pinned_env.py::test_git_archive_of_head_is_extracted_and_read_only`: the sandbox runs as root; unrelated); that full run was before the capture's review fixes, whose own suites were re-run after. `tests/regression`, `tests/governance`, play-path closure, bundle sync, `tests/unit/test_rules` pass. Coverage not measured locally; CI is the gate.
**Lint Status:** `ruff check .` clean; `ruff format --check` clean on `alpha_zero/` and `tests/`.
**Type Check Status:** `python -m mypy alpha_zero/bridge/gameinfo_capture.py --follow-imports=silent`: no issues. `mypy alpha_zero/` cannot finish in this sandbox; CI is the gate.
**Smoke Tests:** end to end on the stand-in: the mod's printed snapshot -> `gameinfo_capture` -> `import-gameinfo` -> a research registry with technology and civic slots -> encoder and mask. `python scripts/build_lua_bundle.py --check` exits 0.
**Verification not done:** `make verify`; **nothing was run against a real game**. The column names in `GameInfoExporter.lua` are unverified against the current game.
**Play path:** the diff against the default branch over `scripts/release_play_path.txt` is empty.

## What changed
- **Mod:** `GameInfoExporter.lua` copies six GameInfo tables once per game load (identifiers and whole numbers only, exactly the importer's input format), written through `AlphaZeroFileIO.WriteGameInfo` as `gameinfo.json` or as chunked `AZ_GAMEINFO` lines. `ENABLE_GAMEINFO_EXPORT` (default on). No new command, no protocol change; a failing export never stops initialisation.
- **Python:** `bridge/gameinfo_capture.py` reads the snapshot from `Lua.log` or the file, requires all six tables, runs the importer's own check, and files it under `<CIV6AZ_DATA_ROOT>/gameinfo/`; refuses a repository checkout (links resolved), a missing data root, an existing file, and an unfinished newest snapshot unless `--allow-earlier`.
- **Encoder fix (pre-existing defect):** `JsonEncoder.lua` used a pattern with `\x` escapes, which Lua 5.1 lacks. Every message was about four times as long (stand-in state 20,319 -> 5,074 characters) and a string with a quote produced invalid JSON; a newline came back as backslash-n.

## Review of record
- Tests by `sdlc-qa-sdet` against a written contract before the code: 134 of 136 failed or errored; ten planted defects each failed the intended tests. It found the encoder defect while sizing the chunks. After the review, 59 further tests failed on the then-current capture.
- Review by `sdlc-architect` (read-only), findings and what was done:
  1. A `gameinfo/` symlink into a checkout's subdirectory got past the checkout test. **Fixed** (resolve first; the old test passed for the wrong reason and a subdirectory case was added).
  2. `OSError` escaped the CLI as a traceback. **Fixed.**
  3. A partial snapshot was filed silently. **Fixed** (all six tables required).
  4. An earlier game load's snapshot was used silently when the newest was unfinished. **Fixed** (`scan_log`, refusal, `--allow-earlier`).
  5. Checkout detection relied on `.git` alone. **Fixed** (the package's source tree is refused too).
  6. Weak tests (`_publish` fallback, CLI I/O errors). **Fixed.** The encoder commit's message made a false claim about Lua versions; **corrected** in the next commit's message (the code was right).
  7. Device names, over-long names, chunk denominators. **Fixed.** The whole log is read into memory: **not changed**, noted.

## Known issues and open questions
- **Unverified against a game:** the column names (`TechnologyType`, `Cost`, `EraType`, `PrereqTech`, `BaseMoves`, ...), that GameInfo rows return nil for an absent column, the size of the real snapshot (estimated around 40 log lines), and whether the game's `iscntrl` depends on the locale (bytes 0x80-0x9f could then be escaped one byte at a time; the JSON would stay valid).
- The flag can only be turned off through the mod's override file: `Initialize()` rebuilds the config defaults, so a runtime `Set` does not survive it.
- The capture reads the whole `Lua.log` into memory.
- Next in the plan: GM-B5 (`--mcts` refuses until an action translator exists).
