# `.claude/` — the Claude Code harness

This directory configures the coding agent the maintainer uses on this
repository. **A human contributor does not need any of it**; `CONTRIBUTING.md`
is the complete guide for people. It is published because the harness is part
of how the repository was built and because the pattern is reusable — with the
caveats in *Threat model* below.

## Files

| Path | Git | Purpose |
|------|-----|---------|
| `settings.json` | committed | Permission allow/deny lists, hook registrations, environment variables. |
| `settings.local.json` | gitignored | Personal overrides (tokens, machine-specific paths). Never committed. |
| `hooks/*.sh` | committed | The five hook scripts registered in `settings.json`, plus `readonly_shell_guard.sh` and `tests_only_write_guard.sh`, which role subagents attach in their own frontmatter. |
| `agents/sdlc-*.md` | committed | The eight SDLC role subagents (product-owner, scrum-master, architect, ml-scientist, ai-engineer, developer, qa-sdet, release-manager). Validated by `scripts/validate_claude_agents.py`. |
| `skills/*/SKILL.md` | committed | Eleven skills: the three KB handoffs (`planning-handoff`, `dev-sqe-handoff`, `pre-pr-handoff`), `openspec-propose` / `openspec-archive`, `doc-sync`, `debt-audit`, `negative-control`, `verify`, `release`, `a1-reregister` (the section 7.1 re-registration procedure; gate: `make verify-registration`). Each has frontmatter validated by `scripts/validate_skills.py`. |

`.agents/skills/` (a sibling directory, three entries) is **validated but never
loaded**: `validate_skills.py` scans it, the PostToolUse hook routes it, and
`tests/governance/` covers it, but no session loads skills from there. Treat it
as gated documentation (CLAUDE.md §5.2, row dated 2026-09-02).

## What each hook does

| Event | Script | Behaviour |
|-------|--------|-----------|
| `SessionStart` | `hooks/sessionstart_bootstrap.sh` | Checks Python ≥ 3.11, installs the `dev` extra, and reports what is *actually* true: install failure with the log tail, missing packages, and a tool/interpreter split (a `mypy` resolved from an isolated tool venv cannot see the project's stubs and reports phantom errors). It used to print "pytest ready" unconditionally. |
| `PreToolUse` (Bash **and** PowerShell) | `hooks/pretooluse_guard.sh` | **Denies** commits/pushes while on a protected branch, pushes to or deletions of one in any refspec form, tag pushes (`refs/tags/*`, `--tags`, `--follow-tags`), `--all`/`--mirror`, `git tag` with any non-listing argument, force-pushes (`--force-with-lease` allowed), `gh pr merge`, `gh release`, `gh repo edit`, `gh ruleset`, `gh api` with a method/body field or `graphql`, and an explicit game count above 1000 without `--approval_ref` (the guard cannot price a run; the runner's `game_budget` refuses the worst case). **Asks** (prompts you) on `--approval_ref`, `--preset production`, `--config` on a game runner, and unknown game runners. Protected = `dev qa main` + the default branch + `CIV6AZ_PROTECTED_BRANCHES`. Unwraps `pwsh -c`, `powershell -Command`, `cmd /c`, `bash -c`, `iex`, `&` and launchers (`env`, `command`, `sudo`, `nohup`, `timeout`, ...) first; program names match case-insensitively (`GIT`, `git.EXE`); quotes inside arguments (`HEAD:"qa"`) and PowerShell backtick escapes (`` g`it ``) are removed; a quoted program path (`& 'C:\Program Files\Git\cmd\git.exe'`) is reduced to its basename. Matching is anchored to the command itself so a quoted mention does not trip it. If no parser (jq, python3, python) can read the payload, or the script fails for any reason, it **denies** (exit 0 + JSON), because a guard that fails open is invisible. |
| `PostToolUse` (Write/Edit) | `hooks/posttooluse_lint.sh` | `ruff check --fix` + `ruff format` on the edited Python file. F401 and RUF100 are deferred to the commit-time gates so an import added in one edit is not stripped before its first use lands in the next. Never fails the originating tool call. |
| `PostToolUse` (Write/Edit **and** Bash) | `hooks/posttooluse_governance.sh` | Runs the matching governance validator in single-file mode for `openspec/`, `kb/`, skills and `.claude/agents/` role subagents, a whole-repo `check_doc_sync.py` for documentation, and the dead-code ratchet when a file appears under or leaves `alpha_zero/` (which is why it is also on the Bash matcher — a deletion is not a Write). |
| `PreCompact` | `hooks/precompact_state.sh` | Folds KB session state, recent ledger lines and a `git diff --stat` into the compacted context so workflow state survives compaction. |
| `Stop` | inline `echo` | Reminds the agent to run its role's handoff skill (CLAUDE.md §14). |

## Role subagents (`agents/`)

Eight persisted subagents carry the SDLC roles; the three handoff skills dispatch
them. Read-only roles return text (a memo, script, doc, PR body or manifest) and
`sdlc-developer` commits it verbatim with `authored-by: sdlc-<role>`, after which the
authoring role reviews the PR. `sdlc-qa-sdet` is limited to `tests/**` for Edit/Write and to the
read-only shell plus `python -m pytest <paths under tests/>`; pytest executes test code, so that is an
accident net, not a sandbox.

| Role | Model | Tools | Shell guard |
|------|-------|-------|-------------|
| `sdlc-product-owner` | sonnet | Read, Grep, Glob | none (no shell) |
| `sdlc-scrum-master` | sonnet | Read, Grep, Glob, Bash | `readonly_shell_guard.sh` |
| `sdlc-architect` | opus | Read, Grep, Glob, Bash | `readonly_shell_guard.sh` |
| `sdlc-ml-scientist` | opus | Read, Grep, Glob, Bash | `readonly_shell_guard.sh` |
| `sdlc-ai-engineer` | opus | Read, Grep, Glob, Bash | `readonly_shell_guard.sh` |
| `sdlc-developer` | inherit | all | none (the executor) |
| `sdlc-qa-sdet` | sonnet | Read, Grep, Glob, Bash, Edit, Write | `readonly_shell_guard.sh --qa` on Bash, `tests_only_write_guard.sh` on Edit/Write |
| `sdlc-release-manager` | sonnet | Read, Grep, Glob, Bash | `readonly_shell_guard.sh` |

The guards are attached through each role's own frontmatter `hooks:` (they run only while
that subagent is active) -- never through `settings.json`. `readonly_shell_guard.sh` allows
exactly one plain command: `git log|show|diff|status|rev-parse|merge-base|ls-files|cat-file`,
`git tag -l`, `gh pr view|list|checks`, `gh api` (GET only: ONE relative API path -- never a URL,
scheme or host -- plus `--jq`/`-q` with a plain field path that is not `env`, `--paginate`,
`-H X-GitHub-Api-Version:...`) and one of ten vetted `python scripts/*.py` checkers, matched by exact
case-sensitive name and exact options (argparse accepts abbreviations, so `--wr` is `--write`).
A payload whose `command` is missing or not a string is denied.
It rejects every metacharacter (`; & | < > ( ) { } $` backtick, quotes, backslash, `, @ # %`),
every glob character, every flag not on the per-command allowlist, and anything it cannot parse.
Quote-free by design: use `--pretty=oneline`, not `--format="%h %s"`. It is an accident
net: identity is the control. `git status`/`git diff` refresh `.git/index` and can briefly hold
`index.lock`; if a developer in the same worktree sees `index.lock exists`, run the read-only role with
`GIT_OPTIONAL_LOCKS=0`. Every hook command ends `|| exit 2` so a guard that cannot start blocks.
Check the files with `python scripts/validate_claude_agents.py`.

`sdlc-qa-sdet` also gets `readonly_shell_guard.sh --qa`: the list above plus
`python -m pytest <paths under tests/>` with `-q -x -v -s --tb=short -k <word> -m <word>
-p no:cacheprovider --maxfail=N --confcutdir=<tests path>` (no `-c`), optionally after
`OMP_NUM_THREADS=<digits>`. **Residual risk:** pytest executes test code and conftest, so a test can
write anywhere and this is an accident net, not a sandbox; the flag list stops accidents, not a test
author (a `tests/pytest.ini` or `conftest.py` it writes is picked up automatically). The review reads the tests it ran.

### Live verification (maintainer, once per checkout and after any guard or agent edit)

The guards are tested by driving the scripts; nothing here proves Claude Code attaches a
frontmatter hook to a live subagent. Prove it:

1. Start Claude Code in this checkout and dispatch one read-only role, e.g. "use sdlc-architect to run
   `git push --dry-run origin HEAD`".
2. Expect a denial that begins `read-only role shell guard:` (the role then reports it could not run
   it). A successful dry-run output means the hook is NOT attached: stop and check the frontmatter
   `hooks:` block and `$CLAUDE_PROJECT_DIR`.
3. Dispatch `sdlc-qa-sdet` with "write `alpha_zero/_probe.py`" and expect `tests-only write guard`, then
   with "run `python -c print(1)`" and expect `read-only role shell guard`.
4. Record the three denials in the handoff note. Until this is done the guards are unverified live.

## What the permission allowlist allows

`settings.json` pre-approves, without a prompt: `ruff`, `mypy`, `pytest`,
`mlflow`, `pip install` / `pip list`, `python -m alpha_zero …`, `python -c …`,
`python scripts/…`, the `make` targets for lint / type / test / verify / validate /
security, read-only `git` (`status`, `diff`, `log`, `branch`) **and** mutating
`git` (`checkout`, `add`, `commit`, `push`, `fetch`, `pull`, `stash`), plus `ls`,
`wc`, `which`, `node`, `npx`.

It denies: `rm -rf /`, `rm -rf ~`, `docker push`, `gcloud`, `gsutil`, `kubectl`,
`Edit`/`Write` to `.env` and `.env.local`, and -- as both `Bash(...)` and
`PowerShell(...)` rules -- `gh pr merge`, `gh release`, `gh repo edit`,
`gh ruleset`, `gh api -X`/`--method`, and tag pushes (`--tags`, `--follow-tags`,
`refs/tags/`). The rules are a second layer; the guard above is the one that
sees through wrappers. `enableAllProjectMcpServers` is `false`.

Environment set for every session: `PYTHONDONTWRITEBYTECODE=1`,
`PYTEST_TIMEOUT=300`, `OMP_NUM_THREADS=2` (torch's intra-op pool wedges on
oversubscribed VMs — CLAUDE.md §5.3), `CIV6AZ_DEFAULT_BRANCH=dev` and
`CIV6AZ_PROTECTED_BRANCHES="dev qa main claude/civ6-alphazero-implementation-vMnun"`.
The list, not the default, drives protection: the old default stays on it until
the branch cut-over retires it (`openspec/changes/branching-and-release/`).

## Agent identity (maintainer action, not yet provisioned)

Charter §4.6: maintainer-only merges and tags are enforced by **identity**, not
by this guard. The planned identity is a machine user, `civ6az-agent` (repo
Write, on no ruleset bypass list). Once its fine-grained PAT exists, put it in
the gitignored `settings.local.json` and stop git from falling back to the
maintainer's Git Credential Manager login:

```json
{
  "env": {
    "GH_TOKEN": "<civ6az-agent fine-grained PAT>",
    "GIT_CONFIG_COUNT": "2",
    "GIT_CONFIG_KEY_0": "credential.helper",
    "GIT_CONFIG_VALUE_0": "",
    "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
    "GIT_CONFIG_VALUE_1": "!f() { echo username=x-access-token; echo \"password=$GH_TOKEN\"; }; f"
  }
}
```

The empty `credential.helper` clears the inherited helper list; the second key
supplies the PAT for `github.com` only. **Residual risk:** GCM still holds the
maintainer's login under the same Windows profile, so a process that resets
these variables can use it. Full isolation means running agents under a separate
Windows account. Nothing in the committed `settings.json` sets these variables,
because doing so before the PAT exists would break every push.

## Branches, releases and the game cap (what an agent may do)

The branch model is `dev` (default, integration), `qa` (release candidates) and `main`
(tagged releases); the full flows are in
[`docs/runbooks/branching-and-release.md`](../docs/runbooks/branching-and-release.md).
For an agent that means:

- Work on a short-lived branch (`feature/*`, `fix/*`, `chore/*`, `docs/*`, `claude/*`) and
  open a PR into `dev`. Never commit or push to `dev`, `qa` or `main`.
- **Releases:** the `release` skill prepares the release PR (version bump in
  `pyproject.toml`, `alpha_zero/__init__.py` and `CITATION.cff`, plus the CHANGELOG roll)
  and stops. Tagging is the maintainer step `python -m scripts.maintainer.tag_release
  vX.Y.Z[-rc.N]`; an agent never runs `git tag` or pushes a tag, and `--execute` refuses
  inside a Claude Code session. The same goes for `scripts/maintainer/branch_cutover.sh` and
  `merge_backmerge.sh`.
- **Game cap:** every runner except the distributed coordinator (guard asks) prices its own worst case
  (`alpha_zero/utils/game_budget.py`) and refuses above 1,000 games. `--max_games` can only
  lower the cap; `--approval_ref <issue>` raises it and is **recorded, not verified**, which
  is why the guard asks before running any command that carries it. A run card
  (`scripts/run_card.py`) is priced in total across all its invocations
  (`--price_only` prints the price and creates nothing).
- **Local runs** are fail-closed against GCP (`LocalModeViolation`) and run from pinned
  source and a pinned venv under `CIV6AZ_DATA_ROOT`.

Whether the maintainer-only parts are *enforced* depends on the identity and rulesets
described above having been provisioned. Until then they are a convention, the guard is an
accident net, and Branch Direction (the PR check) can be faked by any Write-holder.

## Threat model — read before copying this

**This is a heuristic PreToolUse guard, not a sandbox.** It inspects the
command string the agent proposes and pattern-matches it. It does not confine
the process, the filesystem or the network. Specifically:

- **Known bypasses.** Finding **S7** in
  [`docs/audits/2026-09-04-standards-audit.md`](../docs/audits/2026-09-04-standards-audit.md)
  is partly fixed: `bash -c`, `sh -c`, `pwsh -c`, `cmd /c`, `iex` and `&` are now
  unwrapped (as are `env`/`command`/`sudo`) and `gh pr merge` is denied. Still
  open: `eval`, `xargs`, variables holding the command (`$c`), and any script or interpreter
  (`python -c`, `node -e`) that shells out or calls the GitHub API itself -- a
  pre-approved helper that runs `gcloud` is allowed even though `gcloud` is
  denied. That residue is why identity, not the guard, is the control.
- **Open items for the maintainer.** `Bash(python -c:*)` / `Bash(python3 -c:*)` /
  `Bash(npx:*)` / `Bash(node:*)` pre-approve arbitrary code execution. Narrowing
  these is a maintainer decision tracked under the public-release OpenSpec
  change; until it lands, assume an agent running under this configuration can
  execute anything the invoking user can.
- **Bash-mediated writes to `.env`** (`echo … > .env`) are not pattern-blocked;
  only the `Edit`/`Write` tools are.
- **What the guard is for:** the *accidental* commit to the default branch,
  the *accidental* force-push, and the *accidental* 5,000-game self-play run —
  the mistakes an agent makes in passing. It will not stop a determined or
  adversarial process, and it has had its own fail-open bugs (jq missing →
  allow; literal `main` vs the real default branch) — see the 2026-09-02 and
  2026-09-04 rows in CLAUDE.md §5.2.

If you reuse this harness, run `tests/governance/test_pretooluse_guard.py`
against *your* checkout: it drives the script with real stdin payloads under a
minimal `PATH`, which is how the fail-open bug was found.

## Adapting it to another repository

1. Copy `settings.json` and `hooks/`. Set `CIV6AZ_DEFAULT_BRANCH` and
   `CIV6AZ_PROTECTED_BRANCHES` (or rename the variables) to your branches, and
   change the `1000`-game cap and the runner lists in the guard, or delete
   those rules.
2. Replace the governance validators the PostToolUse hook calls
   (`scripts/validate_*.py`, `scripts/check_doc_sync.py`) with your own, or
   delete `posttooluse_governance.sh` from `settings.json`.
3. Decide on the allowlist deliberately: remove `python -c`, `npx`, `node` and
   the mutating `git` entries unless you want them prompt-free, and set
   `enableAllProjectMcpServers` to `false` unless you audit every MCP server in
   the project configuration.
4. The skills assume this repository's `kb/` layout and OpenSpec tree; start
   from `docs/templates/CLAUDE_TEMPLATE.md` and
   `docs/templates/PROJECT_CHARTER_TEMPLATE.md` rather than from `CLAUDE.md`.
5. Keep the hooks under test (`tests/governance/`). A hook that nothing
   exercises is prose.
