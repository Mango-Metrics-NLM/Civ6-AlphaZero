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

### 2. Reintroduce the defect

Take a backup, then revert the fix only:

```bash
cp <file> /tmp/<file>.fixed
# edit <file> to restore the original broken behaviour
```

Keep the change minimal and behavioural — reverting the whole commit also
reverts the test.

### 3. Confirm the test fails

```bash
pytest <test target> -q
```

It must FAIL, and the failure message must name the real problem. A test that
errors for an unrelated reason (import error, fixture problem) has not been
validated.

### 4. Restore and confirm it passes

```bash
cp /tmp/<file>.fixed <file>
pytest <test target> -q
```

Verify with `git diff` that the working tree matches your intended fix — a
restore that silently drops part of the change is worse than no control at all.

### 5. Record the result

State in the commit message or handoff: which tests failed against the original
code, and that they pass against the fix. For example: "Verified by negative
control: 5 tests fail against the original code and pass against this change."
