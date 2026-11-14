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
