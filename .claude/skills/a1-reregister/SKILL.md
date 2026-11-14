---
name: a1-reregister
description: Re-register the A1 protocol after any edit inside docs/EVALUATION.md section 7.1 — check the amendment is still allowed, edit the region, add the registration-log row, re-confirm the constants, bring the run cards and their tests back in line. Use whenever the hashed region changes, including the T0b slot fill-in. Gated by make verify-registration.
allowed-tools: Read, Write, Edit, Bash
---

# A1 Re-registration Skill

Section 7.1 of `docs/EVALUATION.md` is a pre-registration: the text between the
`A1-PROTOCOL-BEGIN` and `A1-PROTOCOL-END` markers is hashed, and a run is judged
only under the registration whose digest its artifact records. Any edit inside
the region, deliberate or a formatter's, is a new registration. This skill is the
procedure that keeps the six things that must agree in agreement.

## When to use

- **Use** for any change inside the region: a command, a threshold, a rule's
  wording, the devices block, the "pending T0b" slots.
- **Skip** for edits outside the markers (the registration log itself, other
  sections). They do not change the digest.

## Steps

### 1. Check the amendment is allowed

Read the region's "Amendments" paragraph and decide which case applies.

- **No A1-family run has started** (no run of a registered command at seed 1001,
  1002, 1003, 20261111 or 20261112): any change may be re-registered.
- **An exploratory run has started**: only the "pending T0b" slots and the cards'
  `sha` may change. Anything else needs a **new confirmatory seed pair**, which is
  the maintainer's decision. Stop and ask.

Record who authorized the change (an issue, a PR, a maintainer instruction). It
goes in the log row.

### 2. Edit the region

Make the change. If it touches a registered command, change the same tokens in
`configs/run_cards/a1.yaml` and `a1r.yaml`; if it touches the positive-control
command or the devices block, change `t0b.yaml` too.

### 3. See what is out of step

```bash
python -m alpha_zero.evaluation.a1_protocol digest
```

It prints the digest of the region as it stands, the registration log's last row
and `a1_registration.REGISTRATION_DIGEST`, and says which of them disagree.

### 4. Add the registration-log row

Append a row to the table after the end marker: today's date, the digest printed
as "section 7.1 as it stands", then what changed, why, on whose authority, and
the words "before any A1-family run" when that is the case.

### 5. Re-confirm the constants

`alpha_zero/evaluation/a1_registration.py` holds facts read from the region's
prose (`A1R_SEED`, `DATA_ROOT`) and `a1_protocol.py` holds the verdict thresholds.
Check each against the edited text, then set `REGISTRATION_DIGEST` to the new
digest. Changing the constant without checking is the failure this step exists
to prevent.

### 6. Bring the pinned tests in line

- `tests/scripts/test_a1_registration.py`: `PLAN_TRAINING` and
  `PLAN_ARENA_PREFIX` are hand copies of the plan's commands.
- `kb/specs/20261111-local-first-and-branching-plan.md` section 5.2: add a dated
  note beside the superseded line; do not rewrite history.

### 7. Record it

`CHANGELOG.md` `[Unreleased]`, and a CLAUDE.md section 5.2 row if the change is a
decision rather than a slot fill-in.

## Final gate

```bash
make verify-registration
```

exits 0. It runs the digest check, the registration and card tests, and the
OpenSpec and doc-sync validators.
