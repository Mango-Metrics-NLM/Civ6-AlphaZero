# game-rules -- deltas

## ADDED Requirements

### Requirement: The rules kernel is a leaf outside the play path

Every module under `alpha_zero/rules/` SHALL import only the standard library,
`yaml` and `alpha_zero.rules.*`. No module in the A1 import closure MUST import
`alpha_zero.rules`, and no path matched by `scripts/release_play_path.txt` MUST
change in this change.

#### Scenario: Forbidden import

- **WHEN** a module under `alpha_zero/rules/` imports `torch`, `numpy` or any other `alpha_zero` package
- **THEN** the leaf-import test fails and names the module and the import

#### Scenario: Play path stays empty

- **WHEN** the play-path pathspecs are diffed between the merge base and the change head
- **THEN** the diff lists no file and `tests/scripts/test_release_play_path_closure.py` passes

### Requirement: Rule sets load strictly as integers

The loader SHALL build frozen spec objects from YAML. It MUST reject an unknown
key, a duplicate key, and a float or boolean where an integer is expected, raising
an error that names the offending key.

#### Scenario: Unknown key

- **WHEN** a rule-set file carries a key the schema does not declare
- **THEN** loading raises and the message names that key

#### Scenario: Duplicate key

- **WHEN** a mapping in a rule-set file repeats a key
- **THEN** loading raises instead of keeping the last value

### Requirement: Rule-set identity is a content hash

`content_sha256()` SHALL hash the canonical JSON of the parsed rule set. The hash
MUST be unchanged by CRLF line endings or key reordering. The rule set's own `id` and any field
at its default are left out of the hashed content, so renaming a draft when it is frozen keeps
its hash and a new optional field does not move old pins. Declared order is kept, because
registries index by it. The hash MUST change when any other value, or the order of a list,
changes.

#### Scenario: Line endings and key order

- **WHEN** a rule-set file is rewritten with CRLF line endings and its keys reordered
- **THEN** its hash equals the hash of the original

#### Scenario: A changed cost

- **WHEN** one tech cost in a rule set is changed
- **THEN** its hash differs from the original

### Requirement: The validator enforces structural rules

`validate` SHALL reject a rule set with a duplicate id, an unresolved
prerequisite, a prerequisite cycle, a non-positive cost, an unresolved unlock
reference, or, when a `VictorySpec` names a capstone tech, an unreachable capstone.
Each rule MUST have a fixture that fails only that rule.

#### Scenario: Prerequisite cycle

- **WHEN** a rule set declares tech A requiring B and B requiring A
- **THEN** validation fails and the error names the cycle rule and both ids

#### Scenario: Negative control

- **WHEN** one validator rule is disabled
- **THEN** that rule's fixture test fails

### Requirement: Built-in rule sets are pinned package data

`builtin.py` SHALL map each built-in id to a package-data resource, a pinned
sha256 and a status, `draft` or `frozen`, and loading MUST raise when the
resource's hash differs from the pin. An id SHALL end in `-draft` exactly when
its status is `draft`. A frozen pin MUST never change, and a content change to a
frozen rule set is a new id. A draft pin MAY be re-pinned. This change ships one
built-in, `science-v1-draft` with status `draft`, which SHALL load through
`importlib.resources` so it works from an installed wheel; it adds no frozen
built-in.

#### Scenario: Edited data without a re-pin

- **WHEN** `science_v1_draft.yaml` is edited and `builtin.py` is not
- **THEN** loading `science-v1-draft` raises a hash-mismatch error

#### Scenario: Status and id disagree

- **WHEN** a built-in entry has status `frozen` and an id ending in `-draft`, or status `draft` and an id that does not end in `-draft`
- **THEN** the built-in table test fails and names the entry

### Requirement: The rule-set log lists exactly the frozen pins

The log in `docs/GAME_RULES.md` SHALL list exactly the `(id, sha256)` pairs of
the frozen built-ins and MUST NOT list a draft. The comparison that holds the
log equal to `builtin.py` MUST fail on a missing row, an extra row, a hash
mismatch and a draft row, and MUST be exercised on injected tables so that it is
not vacuous while no frozen built-in exists.

#### Scenario: Log drift

- **WHEN** a frozen pin has no matching row in the log, or a row has no matching frozen pin
- **THEN** the log test fails and names the id

#### Scenario: A draft in the log

- **WHEN** the log lists an id whose built-in status is `draft`
- **THEN** the log test fails

#### Scenario: Empty log, non-vacuous check

- **WHEN** the built-in table holds no frozen entry and the log holds no row
- **THEN** the log test passes, and the same comparison given one injected frozen pin and an empty log fails

### Requirement: Queries and registry are deterministic

`available(spec, researched)` SHALL return exactly the unresearched techs whose
prerequisites are all researched. `research_cost(spec, tech, boosted)` SHALL
return an integer from integer-percent arithmetic. The registry SHALL map names
to indices in declared order, with a fingerprint that MUST be identical across processes.

#### Scenario: Availability at the start

- **WHEN** `available(science_v1_draft, set())` is called
- **THEN** it returns exactly the techs with no prerequisites, in declared order

#### Scenario: Registry fingerprint

- **WHEN** the registry for `science-v1-draft` is built in two separate processes
- **THEN** both index maps and fingerprints are equal

### Requirement: The kernel CLI reports through exit codes

`python -m alpha_zero.rules` SHALL provide `validate`, `hash`, `pace` and
`import-gameinfo`, exiting 0 on success, 1 on a validation failure and 2 on a
usage error or a refused path. `pace science-v1-draft` MUST report a capstone path cost of
580 and a full-tree cost of 780, the first-pass costs of plan section 3.3.

#### Scenario: Validating a broken file

- **WHEN** `validate` is given a rule-set file with an unresolved prerequisite
- **THEN** it exits 1 and prints the failing rule

#### Scenario: Pacing the built-in set

- **WHEN** `pace science-v1-draft` runs
- **THEN** it exits 0 and reports 580 and 780

### Requirement: GameInfo imports never write into the repository

`import-gameinfo` SHALL turn a GameInfo table export into a rule set that passes
`validate`, and it MUST refuse any output path that resolves inside the checkout.

#### Scenario: Output inside the checkout

- **WHEN** `import-gameinfo` is given an output path under the repository root
- **THEN** it exits 2 and writes nothing

#### Scenario: Synthetic export

- **WHEN** `import-gameinfo` reads the synthetic export fixture with an output path outside the checkout
- **THEN** it exits 0 and the written file passes `validate`
