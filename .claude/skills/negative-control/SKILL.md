---
name: negative-control
description: Prove a new or changed test actually catches the bug it targets, by reintroducing the defect and confirming the test fails. Use whenever adding a regression test for a fixed bug, or when a test suite passed while something was known to be broken.
allowed-tools: Bash, Read, Edit
---

# Negative Control Skill

A test that passes proves nothing until you have seen it fail for the right
reason.

## Why this exists

This repository's characteristic failure is a green suite over broken
behaviour:

- Batch self-play emitted no `action_masks`; the trainer silently substituted
  an all-ones mask, so the network trained against illegal actions. The parity
  test that should have caught it compared key sets with `issubset` (which
  cannot detect an omitted field) and wrapped every assertion in
  `if batch_trajs and seq_trajs:` (so it passed on empty output).
- Batch games were cut off at an action-step count compared against a game-turn
  limit, so every value target was 0.0. The tests asserted trajectory keys,
  field alignment and mask density — all of which a uniformly-zero target
  satisfies.
- `compute_microciv_action_space_size()` omitted a block and returned 356, and
  `validate_constants` asserted 356, so the error validated itself green.

In each case the missing step was the same: nobody checked the test could fail.

## Steps

### 1. Identify the discriminating assertion

Name the specific assertion that should fail if the bug returns. If you cannot
name one, the test does not discriminate yet — fix that first. Prefer an
assertion about *content* (a value distribution, a density, a signed outcome)
over one about *shape* (keys, lengths, types), because shape assertions survive
most degenerate data.

### 2. Back up, then reintroduce the defect

First look at the file you are about to change:

```bash
git status --porcelain -- <file>
```

Either the file has no uncommitted changes, or you accept that the backup is of the
file as it is, uncommitted changes included, and that this exact copy is what gets
restored. Both are fine. What is not fine is assuming the committed version is the one
to go back to.

Take the backup in the session's scratchpad or a temporary directory, named by its
full path, never beside the file (a copy beside the file gets linted, collected by
pytest or committed). Name each backup by its repository path, not its base name, so
two files called `__init__.py` cannot overwrite each other: inside the backup
directory, `alpha_zero/core/x.py` is saved as `alpha_zero/core/x.py`.

```bash
BACKUP_DIR="$(mktemp -d)"            # or a new directory in the session scratchpad
echo "backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR/$(dirname <repo/path/to/file>)"
cp <repo/path/to/file> "$BACKUP_DIR/<repo/path/to/file>"
ls -l "$BACKUP_DIR/<repo/path/to/file>"
# edit <repo/path/to/file> to restore the original broken behaviour
```

Print the backup directory and write it down. A shell variable does not survive
between tool calls, so in step 4 `$BACKUP_DIR` is empty: use the literal absolute path
that was printed.

Keep the change minimal and behavioural: reverting the whole commit also reverts the
test.

If the planted change needs a generated file rebuilt before the test can see it (the
Lua bundle `mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`, rebuilt by
`python scripts/build_lua_bundle.py` from the Lua modules beside it), back up the
generated file the same way before rebuilding, and restore it too in step 4.

### 3. Confirm the test fails

```bash
pytest <test target> -q
```

It must FAIL, and the failure message must name the real problem. A test that
errors for an unrelated reason (import error, fixture problem) has not been
validated.

### 4. Restore from the backup and confirm it passes

Copy the backup back, check that the file equals the backup, then run the test. Use
the literal absolute path printed in step 2 in place of `/abs/backup/dir`:

```bash
cp /abs/backup/dir/<repo/path/to/file> <repo/path/to/file> \
  && cmp /abs/backup/dir/<repo/path/to/file> <repo/path/to/file> \
  && echo restored \
  && pytest <test target> -q
```

The commands are one `&&` chain on purpose: if the copy or the comparison fails, the test
does not run. Stop if `cmp` reports a difference (no `restored` line): the file is not restored, and
nothing else should be run or edited until it equals the backup. A checksum of both
files does the same job (`sha256sum`, or `shasum -a 256` on macOS), as does `diff`.
Do the same for every generated file you backed up; restore it from its backup and do
not rebuild it, so that it too equals its backup byte for byte.

**Never use `git checkout`, `git restore` or `git stash` to undo a planted defect.** They put back the committed version, not the file you started from, so they discard every uncommitted change in that file along with the defect.

Then read `git diff -- <file>` and confirm the working tree still holds your intended
fix: a restore that silently drops part of the change is worse than no control at all.

### 5. Record the result

State in the commit message or handoff: which tests failed against the original
code, and that they pass against the fix. For example: "Verified by negative
control: 5 tests fail against the original code and pass against this change."
