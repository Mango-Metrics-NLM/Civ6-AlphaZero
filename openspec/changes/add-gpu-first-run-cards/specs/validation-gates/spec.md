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

### Requirement: A pin that names its build is compared in full

A torch pin MAY name its build with a local version tag (`torch==2.11.0+cu128`).
When it does, the device check and the release verifier MUST compare the
installed `torch.__version__` with the pinned version in full. A pin with no tag
MUST be compared on the public version alone. The registered pin MUST name its
build.

#### Scenario: Another CUDA variant of the same release
- **WHEN** a card pins `torch==2.11.0+cu128` and the pinned environment holds `2.11.0+cu130`
- **THEN** the card is refused, and a run recorded with that build is VOID

#### Scenario: A pin without a tag
- **WHEN** a card pins `torch==2.11.0` and the pinned environment holds `2.11.0+cu128`
- **THEN** the pin is satisfied

### Requirement: A card that pins a torch build names every device

A run card that pins a torch build MUST give every invocation an explicit device.
`run_card` MUST refuse such a card, before anything is created, when an
invocation's device resolves to `auto` or is left to the runner's default. The
device MUST be read from the runner's own argument parser, so abbreviations and
defaults are resolved exactly as the child will resolve them.

#### Scenario: An abbreviated flag
- **WHEN** an invocation passes `--dev cuda:1`
- **THEN** its device is read as `cuda:1` and is held to the device check

#### Scenario: A device left to chance
- **WHEN** a card with a torch pin has a training invocation with no `--device`
- **THEN** the card is refused and no directory is created

### Requirement: A device is checked for what it can run, and for which GPU it is

The device check MUST refuse a named CUDA device when the pinned torch build has
no kernels for that GPU's architecture, when one real CUDA operation fails there,
or when the card names the GPU it expects at an ordinal and a different GPU is
found there. `torch.cuda.is_available()` being true MUST NOT be taken as proof
that a build can run a GPU.

#### Scenario: A build that predates the GPU
- **WHEN** the pinned build's architecture list has nothing for the GPU's compute capability
- **THEN** the card is refused, naming the GPU, its architecture and the build's list

#### Scenario: GPUs in the other order
- **WHEN** a card expects GPU A at `cuda:0` and the pinned child reports GPU B there
- **THEN** the card is refused before any invocation is launched

### Requirement: The registration names the build and the GPUs, and runs are held to it

Section 7.1 MUST register the torch build, the visible devices and the name of
the GPU at each ordinal in a machine-readable block. The registered run cards
MUST carry exactly those values. The release verifier MUST read `run.json`'s
`device_plan` and treat a run on another torch build, device order or GPU as
VOID. A `run_card --probe` record MUST NOT be accepted as an attempt. A
registration with no such block (one that predates it) is judged without a
device check.

#### Scenario: A card drifts from the registration
- **WHEN** one card's `torch_spec`, `cuda_visible_devices` or expected GPU differs from the registered block
- **THEN** a test fails

#### Scenario: A run on other hardware
- **WHEN** a run's `device_plan` names a different GPU at `cuda:1` than the registration
- **THEN** the verifier reads the attempt as VOID

#### Scenario: A probe offered as an attempt
- **WHEN** a bundle lists a `run.json` that a `--probe` wrote
- **THEN** the verifier reports a problem

### Requirement: A card is rehearsed before it claims its run directory

`run_card` MUST run the pinned environment, the preflight and the device check in
a throwaway directory before it claims the card's single-use run directory, and
MUST claim that directory only when the rehearsal passes. A run directory that
was already used MUST be refused before the rehearsal. `run.json` MUST record the
rehearsal that preceded a run, and the name of each invocation before it starts.

#### Scenario: A refused device
- **WHEN** a card names a device its pinned environment cannot provide
- **THEN** the card finishes FAILED, its own run directory does not exist, and the same card can be run again once the cause is fixed

#### Scenario: An attempt interrupted in its first invocation
- **WHEN** a run is interrupted while its first invocation is running
- **THEN** `run.json` names that invocation as launched although no invocation finished

#### Scenario: A used card
- **WHEN** a card whose run directory already holds a `run.json` is run again
- **THEN** it is refused and no rehearsal directory is created

#### Scenario: A failure after the rehearsal passed
- **WHEN** the rehearsal passes and the run itself is then refused or fails
- **THEN** the card's run directory holds a FAILED `run.json` and the card cannot be run again under that `card_id`

#### Scenario: An interrupted rehearsal
- **WHEN** a card is interrupted while it is being rehearsed
- **THEN** the card's run directory does not exist and the same card can be run again

#### Scenario: Two runs of one card at once
- **WHEN** two runs of one card both find its run directory unused
- **THEN** exactly one of them claims it and the other is refused as a resumed card

### Requirement: A pinned GPU run records the driver and the wheel it ran with

A card that pins a torch build and names a CUDA device MUST be refused when the
NVIDIA driver version cannot be read. The pinned environment MUST record the URL
and sha256 of the torch wheel that was installed, when the installer reports
them; a missing report MUST NOT fail the run, and the record MUST NOT be part of
the torch pin that is compared with the registration.

#### Scenario: The driver cannot be read
- **WHEN** a torch-pinned card names `cuda:0` and the preflight has no driver version
- **THEN** the card is refused before any invocation is launched

#### Scenario: The wheel is recorded
- **WHEN** a torch pin is installed and the installer reports the wheel's hash
- **THEN** `run.json` carries it as `torch_wheel`, and a reused environment reports the same value

#### Scenario: A report that names two torch wheels
- **WHEN** the installer's report lists torch more than once
- **THEN** no wheel is recorded and the run continues with a warning

### Requirement: The release verifier applies the re-run rule

When a release bundle lists an attempt that was interrupted after it launched an
invocation, the bundle MUST carry the reproduction comparison between that
attempt and its re-run, and the verifier MUST recompute that comparison instead
of trusting its stored status. A re-run that differs from the attempt, or that
ran at another commit, device plan or driver, MUST NOT be read as a PASS: its
verdict is INCONCLUSIVE and it is labelled a non-reproducing re-run. An attempt
that launched nothing MUST NOT be listed.

#### Scenario: An interrupted attempt with no comparison
- **WHEN** a bundle lists an interrupted attempt and its re-run but no comparison record
- **THEN** verification fails

#### Scenario: A re-run that differs
- **WHEN** the comparison finds a difference and the re-run's arms would give PASS
- **THEN** the stored and recomputed verdict are INCONCLUSIVE with the label, and a bundle that stores PASS fails verification

#### Scenario: A record that claims more than its digests show
- **WHEN** a record's status is REPRODUCED but its digests differ, or its digests contradict the arena results the bundle holds
- **THEN** verification fails

#### Scenario: A second draw
- **WHEN** an attempt completed and is VOID by its symmetry or positive control
- **THEN** no comparison is required for its re-run

