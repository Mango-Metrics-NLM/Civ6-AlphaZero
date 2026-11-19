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
MUST be unchanged by CRLF line endings or key reordering, and MUST change when any value changes.

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

`builtin.py` SHALL map each built-in id to a package-data resource and a pinned
sha256, and loading MUST raise when the resource's hash differs from the pin.
`science-v1` SHALL load through `importlib.resources`, so it works from an
installed wheel. The log in `docs/GAME_RULES.md` MUST list exactly the pinned `(id, sha256)` pairs.

#### Scenario: Edited data without a re-pin

- **WHEN** `science_v1.yaml` is edited and `builtin.py` is not
- **THEN** loading `science-v1` raises a hash-mismatch error

#### Scenario: Log drift

- **WHEN** a pin in `builtin.py` has no matching row in `docs/GAME_RULES.md`
- **THEN** the log test fails

### Requirement: Queries and registry are deterministic

`available(spec, researched)` SHALL return exactly the unresearched techs whose
prerequisites are all researched. `research_cost(spec, tech, boosted)` SHALL
return an integer from integer-percent arithmetic. The registry SHALL map names
to indices in declared order, with a fingerprint that MUST be identical across processes.

#### Scenario: Availability at the start

- **WHEN** `available(science_v1, set())` is called
- **THEN** it returns exactly the techs with no prerequisites, in declared order

#### Scenario: Registry fingerprint

- **WHEN** the registry for `science-v1` is built in two separate processes
- **THEN** both index maps and fingerprints are equal

### Requirement: The kernel CLI reports through exit codes

`python -m alpha_zero.rules` SHALL provide `validate`, `hash`, `pace` and
`import-gameinfo`, exiting 0 on success, 1 on a validation failure and 2 on a
usage error or a refused path. `pace science-v1` MUST report a capstone path cost of
580 and a full-tree cost of 780.

#### Scenario: Validating a broken file

- **WHEN** `validate` is given a rule-set file with an unresolved prerequisite
- **THEN** it exits 1 and prints the failing rule

#### Scenario: Pacing the built-in set

- **WHEN** `pace science-v1` runs
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
