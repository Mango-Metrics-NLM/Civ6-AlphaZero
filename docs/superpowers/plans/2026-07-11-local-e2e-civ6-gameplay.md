# Local E2E Civ6 Gameplay Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a fresh branch synced with origin, install dependencies, prove the test suite is green, then run a full live Civilization VI game locally end-to-end through the file-based Lua bridge.

**Architecture:** The Python agent (`alpha_zero.civ6_local --mode civ6`) talks to a running Civ6 game through file-based IPC: the AlphaZeroAI Lua mod writes `StateMessage` JSON into a shared comm directory and polls for `CommandMessage` replies; `LiveBridge` + `BridgeAgentLoop` decode state, run the policy (greedy network or Gumbel MCTS), and write commands back. No sockets — the Lua sandbox only allows files.

**Tech Stack:** Python 3.11 venv (`.venv`, already created), PyTorch 2.12 CPU, pytest (+pytest-timeout/asyncio), Civ6 via Steam on Windows, Lua mod in `mod/AlphaZeroAI`.

## Global Constraints

- Repo root: `C:\Users\iansh\OneDrive\Documents\Civ6-AlphaZero` (the repo MOVED here — it is no longer under `Civ6 AlphaZero AI agent\`).
- All Python commands use the repo venv interpreter: `.venv\Scripts\python.exe` (Python 3.11.9). Never the system Python.
- Execute in the main checkout, NOT a git worktree — the venv, the installed Civ6 mod, and the bridge comm directory are all tied to this path.
- Current branch `feature/v1.8.0-civ6-steam-demo` is ahead of `origin/main` by 30 and of `origin/feature/v1.8.0-civ6-steam-demo` by 31 commits, behind by 0 on both. "Pull from origin" is expected to be a no-op; verify, don't assume.
- New branch name: `feature/v1.8.1-local-e2e-validation`, created from current HEAD (`0f335e8`).
- Do NOT push to origin — this plan is local-only unless the user later asks.
- Windows pytest: `SIGALRM` timeouts don't exist; keep `--timeout=120` (thread method) and the deselect list below. Two previously-deselected files (`tests/unit/test_batch_inference.py`, `tests/unit/test_civ6_mcts.py`) were DELETED from the repo — do not deselect them.
- Canonical deselect list (4 files that hang under the thread-method timeout on Windows):
  ```
  --deselect tests/unit/test_interfaces.py
  --deselect tests/unit/test_gumbel_mcts.py
  --deselect tests/unit/test_enhanced_training_wandb.py
  --deselect tests/cloud/unit/test_training_muzero_dispatch.py
  ```
- No trained civ6-scale checkpoint exists; `run_civ6_bridge` builds a random-weight network sized to the small-demo action space (`ActionSpaceConfig(board_size, max_units=20, max_cities=5, num_production_items=20)`). That is expected and fine for this validation — do not hunt for a checkpoint.
- Bridge comm dir (auto-resolved): `C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\AlphaZeroAI\`
- Civ6 Lua log (for debugging the mod side): `%LOCALAPPDATA%\Firaxis Games\Sid Meier's Civilization VI\Logs\Lua.log`
- Untracked files at repo root (`pytest_*.txt`, `.claude/scheduled_tasks.lock`) are pre-existing droppings — leave them alone, never `git add` them.

---

### Task 1: Create the branch and sync with origin

**Files:**
- Create: `docs/superpowers/plans/2026-07-11-local-e2e-civ6-gameplay.md` (this file — commit it)
- No source files modified.

**Interfaces:**
- Consumes: existing checkout on `feature/v1.8.0-civ6-steam-demo` at commit `0f335e8`.
- Produces: branch `feature/v1.8.1-local-e2e-validation` checked out, fully up to date with origin, with this plan committed. All later tasks run on this branch.

- [ ] **Step 1: Verify starting state**

Run (Git Bash):
```bash
cd "/c/Users/iansh/OneDrive/Documents/Civ6-AlphaZero"
git status --porcelain
git branch --show-current
```
Expected: branch prints `feature/v1.8.0-civ6-steam-demo`; porcelain output lists ONLY untracked (`??`) files — no modified tracked files. If tracked files are modified, STOP and report; do not stash or reset.

- [ ] **Step 2: Fetch and confirm nothing to pull**

```bash
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git rev-list --left-right --count HEAD...origin/feature/v1.8.0-civ6-steam-demo
```
Expected: first count `30	0` (or higher on the left, always `0` on the right), second `31	0` likewise. Right-hand `0` means origin has nothing we lack. If the right side is non-zero, run `git merge origin/main --no-edit` AFTER creating the branch in Step 3 and report what came in.

- [ ] **Step 3: Create the new branch**

```bash
git checkout -b feature/v1.8.1-local-e2e-validation
git merge origin/main --no-edit
```
Expected: `Switched to a new branch 'feature/v1.8.1-local-e2e-validation'` then `Already up to date.`

- [ ] **Step 4: Commit the plan document**

```bash
git add docs/superpowers/plans/2026-07-11-local-e2e-civ6-gameplay.md
git commit -m "docs: add local e2e civ6 gameplay validation plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
Expected: 1 file changed, commit created on `feature/v1.8.1-local-e2e-validation`.

---

### Task 2: Install and verify dependencies

**Files:**
- No repo files modified (installs into `.venv` only).

**Interfaces:**
- Consumes: branch from Task 1; existing `.venv` (Python 3.11.9, torch 2.12.0+cpu already present, `mlflow` known-missing).
- Produces: a venv where `requirements.txt` is fully satisfied and `import alpha_zero.bridge` works. Tasks 3–6 assume this venv.

- [ ] **Step 1: Install requirements**

Run (PowerShell, from repo root):
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: exit 0. Slow first time (mlflow + friends download); "Requirement already satisfied" for torch/numpy/pytest is normal.

- [ ] **Step 2: Verify the import surface the bridge needs**

```powershell
.venv\Scripts\python.exe -c "import torch, numpy, yaml, mlflow, pytest; from alpha_zero.bridge import BridgeConfig, FileBridge; from alpha_zero.bridge.config import _get_default_comm_dir; print('imports OK'); print('comm dir:', _get_default_comm_dir())"
```
Expected output ends with:
```
imports OK
comm dir: C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\AlphaZeroAI
```

- [ ] **Step 3: Config dry-run of the entry point**

```powershell
.venv\Scripts\python.exe -m alpha_zero.civ6_local --preset civ6 --dry-run
```
Expected: exit 0, logs show the resolved civ6-mode config, no gameplay starts. If this fails, the failure message names the missing piece — fix that before proceeding.

---

### Task 3: Run the test suite

**Files:**
- No repo files modified. Test output goes to the scratchpad, not the repo.

**Interfaces:**
- Consumes: venv from Task 2.
- Produces: evidence (exit code 0) that the suite is green on this branch. Task 6's "when confident" gate depends on this.

- [ ] **Step 1: Bridge-focused subset first (fast signal)**

Run (PowerShell, from repo root):
```powershell
.venv\Scripts\python.exe -m pytest tests -k bridge -q
```
Expected: exit 0, all bridge tests pass. If any fail, STOP — the e2e run cannot be trusted; report failures with output verbatim.

- [ ] **Step 2: Full suite with the Windows deselect list**

```powershell
.venv\Scripts\python.exe -m pytest -q --timeout=120 --deselect tests/unit/test_interfaces.py --deselect tests/unit/test_gumbel_mcts.py --deselect tests/unit/test_enhanced_training_wandb.py --deselect tests/cloud/unit/test_training_muzero_dispatch.py
```
Expected: exit 0. This is a 4800+ test suite — allow up to ~30 minutes; run it in the background and poll, don't kill it early. Record the final `N passed, M skipped/xfailed/deselected in Xs` line.

- [ ] **Step 3: Report the confidence gate**

No commit (nothing changed). State plainly: bridge subset result, full-suite result, and whether the gate to Task 6 is open. Any failure = gate closed = do not run the live game; report instead.

---

### Task 4: Refresh the Civ6 Lua mod install

**Files:**
- No repo files modified. Copies `mod/AlphaZeroAI/` → `C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\Mods\AlphaZeroAI\`.

**Interfaces:**
- Consumes: venv from Task 2; mod source at `mod/AlphaZeroAI` (contains `AlphaZeroAI.modinfo`, `Config/`, `Scripts/`, `panels/`).
- Produces: the installed mod matches this branch's mod source exactly. The mod is ALREADY installed from an earlier session — this refresh matters because recent commits touched the bridge protocol.

- [ ] **Step 1: Run the installer**

```powershell
.venv\Scripts\python.exe scripts\install_civ6_mod.py
```
Expected: exit 0, log lines showing the resolved Mods dir (OneDrive-redirected Documents) and files copied.

- [ ] **Step 2: Verify installed mod matches repo source**

```powershell
$src = "mod\AlphaZeroAI"; $dst = "C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\Mods\AlphaZeroAI"
Compare-Object (Get-ChildItem -Recurse -File $src | Get-FileHash).Hash (Get-ChildItem -Recurse -File $dst | Get-FileHash).Hash
```
Expected: no output (empty diff = identical). Any diff lines = the copy failed; re-run the installer and re-check before proceeding.

---

### Task 5: Pre-game bridge smoke check

**Files:**
- No repo files modified.

**Interfaces:**
- Consumes: venv (Task 2), green tests (Task 3), refreshed mod (Task 4).
- Produces: confirmation that civ6 mode starts, creates/uses the comm dir, and waits for the game — the last automatable checkpoint before a human must launch Civ6.

- [ ] **Step 1: Start civ6 mode without the game and observe the waiting state**

Run in the background (PowerShell, repo root) and stop it after ~60 seconds:
```powershell
.venv\Scripts\python.exe -m alpha_zero.civ6_local --mode civ6 --log-level info
```
Expected log lines (order matters):
```
Connecting to Civ6 bridge at: C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\AlphaZeroAI
Start Civilization 6 with the AlphaZeroAI mod to begin.
Bridge using GreedyNetworkPolicy
```
Then it idles polling for state files. No traceback, no `AttributeError`, no import error. Kill the process after confirming (Ctrl-C / stop the background task; exit 130 on Ctrl-C is expected and fine).

- [ ] **Step 2: Verify the comm dir exists**

```powershell
Test-Path "C:\Users\iansh\OneDrive\Documents\My Games\Sid Meier's Civilization VI\AlphaZeroAI"
```
Expected: `True`.

---

### Task 6: Live E2E gameplay run — USER ACTION REQUIRED

**Files:**
- No repo files modified. Evidence collected to scratchpad.

**Interfaces:**
- Consumes: everything above, all gates green.
- Produces: a completed (or substantially played) live Civ6 session driven by the agent, plus a written verdict on the two bug classes fixed in `0f335e8`.

> **This task cannot start unattended.** A human must launch Civilization VI via Steam, enable the **AlphaZeroAI** mod under *Additional Content*, and start a new game (standard small map is fine). Pause here and hand back to the user with the checklist below; resume when they confirm the game is at the first turn.

- [ ] **Step 1 (USER): Launch the game**

User checklist:
1. Steam → Sid Meier's Civilization VI → Play.
2. Main menu → *Additional Content* → enable **AlphaZeroAI** → back (game may prompt to restart mods context — accept).
3. Single player → Create Game → any small/standard map → Start.
4. Tell the agent the game is loaded and at turn 1.

- [ ] **Step 2: Start the agent loop**

Run in the background with output captured (PowerShell, repo root):
```powershell
.venv\Scripts\python.exe -m alpha_zero.civ6_local --mode civ6 --log-level debug
```
(Greedy policy first. If it plays stably for 10+ turns and the user wants stronger play, restart with `--mcts --num-simulations 16`.)

- [ ] **Step 3: Verify the two historical failure modes stay dead**

Watch the agent log AND `%LOCALAPPDATA%\Firaxis Games\Sid Meier's Civilization VI\Logs\Lua.log` for the specific regressions from commit `0f335e8`:
1. **Zero-observation bug**: agent log must NOT show `missing required field 'width'` or repeated zero-observation fallbacks — state must decode with real map dimensions every turn.
2. **Command-delivery bug**: agent log must NOT show `AttributeError` on `send_and_wait_with_retry`; `Lua.log` must NOT show the 31s poll-stop timeout — commands must be acked each turn.

Expected healthy pattern per turn: state file read → observation built → action chosen → command written → ack received → next turn.

- [ ] **Step 4: Let it play and collect the verdict**

Success bar: agent plays **at least 20 consecutive turns** with no traceback, no circuit-breaker open, no heartbeat loss. Longer (a full game to completion or defeat) is better; 20 clean turns is the minimum to call the e2e PASS. Copy the last ~100 log lines to the scratchpad as evidence.

- [ ] **Step 5: Report**

Report PASS/FAIL against the Step 3 + Step 4 criteria with the actual log excerpts. If FAIL: capture full agent log + `Lua.log`, do NOT attempt fixes in this plan — a failure here is new information that gets its own debugging session (superpowers:systematic-debugging).
