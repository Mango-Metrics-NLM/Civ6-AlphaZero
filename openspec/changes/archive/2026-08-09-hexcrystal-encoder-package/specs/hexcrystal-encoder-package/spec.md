# Hex-Crystalline Encoder Package — deltas for hexcrystal-encoder-package

## ADDED Requirements

### Requirement: Standalone encoder subpackage

The hex-crystalline encoder SHALL live in a subpackage whose runtime dependencies
are limited to `torch` and the Python standard library, and it MUST be importable
in an environment where `numpy` and `gymnasium` are absent. Importing the root
`alpha_zero` package MUST NOT eagerly import the training stack.

#### Scenario: The encoder imports without numpy or gymnasium

- **WHEN** a process blocks `numpy` and `gymnasium` from being imported and then
  imports the encoder subpackage and runs a forward pass
- **THEN** the import and the forward pass both succeed, and neither blocked
  module is present in `sys.modules`

#### Scenario: The root package no longer pulls the training stack

- **WHEN** a process imports the root `alpha_zero` package
- **THEN** neither `numpy` nor `gymnasium` is imported as a side effect of that
  import alone

### Requirement: Backwards-compatible encoder import surface

Moving the encoder modules SHALL NOT break any existing import path, checkpoint,
or model weight namespace. The pre-existing `alpha_zero.core` module paths MUST
continue to resolve to the same objects, and network `state_dict` keys MUST be
byte-identical to those produced before the move.

#### Scenario: Old import paths resolve to the moved objects

- **WHEN** a caller imports a class from its historical `alpha_zero.core.*`
  location and the same class from the new subpackage
- **THEN** the two references are the same object

#### Scenario: Checkpoint key namespace is unchanged

- **WHEN** a transformer network is constructed with either the `patch_embed` or
  the `hex_crystalline` encoder and its `state_dict` keys are listed
- **THEN** the key set equals the set recorded before the move, with encoder
  weights still namespaced under `encoder.patch_embed.`

### Requirement: Hub-publishable encoder artifact

The encoder SHALL offer an opt-in Hugging Face Hub interface providing
`save_pretrained` and `from_pretrained`, writing weights as safetensors and
configuration as a flat JSON document. The dependency MUST be optional: the base
subpackage MUST remain importable and functional when the hub dependency is
absent, and the failure mode MUST name the extra that supplies it.

#### Scenario: An encoder round-trips through a local directory

- **WHEN** a hub-enabled encoder is saved to a directory and reloaded from it
  with no network access
- **THEN** the reloaded encoder's weights and forward outputs match the original,
  and the directory contains safetensors weights and a JSON configuration

#### Scenario: The hub dependency is absent

- **WHEN** the hub integration module is imported in an environment without the
  hub dependency installed
- **THEN** an `ImportError` is raised whose message names the extra to install,
  and the base encoder subpackage remains importable

### Requirement: Reproducible encoder A/B evidence

Comparative evidence between the `patch_embed` and `hex_crystalline` encoders
SHALL be produced by a single driver that trains both arms from identical code
under an identical, explicitly recorded budget and seed, and evaluates them
head-to-head. The published evidence MUST record the sample size and an interval
estimate, and MUST NOT assert a difference the interval does not support.

#### Scenario: Both arms are trained under one recorded budget

- **WHEN** the A/B driver trains both encoder arms
- **THEN** each arm's checkpoint is accompanied by a sidecar recording the
  encoder type, seed, resolved configuration, code revision, and device

#### Scenario: Committed A/B evidence

- **WHEN** the evidence run completes
- **THEN** a committed, version-controlled report contains the head-to-head win,
  draw, and loss counts, the win rate with a confidence interval, and the exact
  training budget and seed used

#### Scenario: A checkpoint is loaded against the wrong encoder

- **WHEN** a checkpoint trained with one encoder type is loaded into a network
  built for the other
- **THEN** validation fails with an error naming both the expected and the found
  encoder type, rather than silently loading a partial set of weights

### Requirement: Human-gated model publication

Publishing artifacts to an external model registry SHALL NOT occur as an
automated side effect. The upload tool MUST default to a dry run that performs no
network writes, MUST require an explicit flag to publish, and MUST source
credentials only from the environment.

#### Scenario: The upload tool runs without the publish flag

- **WHEN** the upload tool is invoked without the explicit publish flag
- **THEN** it stages the artifacts locally, prints the file manifest, and
  performs no registry write

#### Scenario: Publishing is attempted without credentials

- **WHEN** the upload tool is invoked with the publish flag but no credential is
  present in the environment
- **THEN** it exits non-zero with an actionable message and performs no registry
  write
