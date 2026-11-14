## ADDED Requirements

### Requirement: A run card gets the devices it names, or does not run

`run_card` MUST refuse to launch any invocation of a card when an invocation's
`--device` names a CUDA device that the card's pinned environment cannot provide,
or when the card pins a torch build and the pinned environment holds a different
one. It MUST NOT fall back to another device. The check MUST read the pinned
child's own preflight report.

#### Scenario: CUDA requested, CPU-only environment
- **WHEN** a card's argv contains `--device cuda:1` and the pinned environment's torch reports CUDA unavailable
- **THEN** the card finishes FAILED with a `devices:` error and no invocation is launched

#### Scenario: Device ordinal out of range
- **WHEN** a card names `cuda:2` and two CUDA devices are visible
- **THEN** the card is refused, naming the invocation and the device

#### Scenario: Rehearsal
- **WHEN** `run_card --probe` runs a card
- **THEN** the pinned environment is built, preflighted and device-checked in a throwaway run directory, nothing is launched, and the card's own run directory is not created

### Requirement: A pinned environment is identified by its torch build

When a card pins a torch build, the pinned environment MUST install exactly that
build from the named index before the project wheel, and MUST NOT reuse an
environment built for the same commit with a different torch pin or none.

#### Scenario: CPU and CUDA environments for one commit
- **WHEN** one commit is prepared first without a torch pin and then with one
- **THEN** two distinct environments exist and the second received the pinned torch install

#### Scenario: Only an exact pin from a plain index
- **WHEN** a card gives a torch range, a non-`torch` requirement, a non-https index, or an index URL carrying credentials, a query or extra arguments
- **THEN** the card is refused at load
