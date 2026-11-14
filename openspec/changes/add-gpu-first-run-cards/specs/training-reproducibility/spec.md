## ADDED Requirements

### Requirement: A seeded run on a GPU enforces deterministic algorithms

A seeded training, benchmark or throughput-calibration run whose device is a CUDA
device MUST enable deterministic algorithms before it builds or loads a network,
so that an operation with no deterministic implementation raises instead of
letting two runs at one seed diverge. A run on any other device MUST be left
unchanged. The reproduction this provides is scoped to the same machine, the
same torch build and the same device.

#### Scenario: Seeded GPU training
- **WHEN** `training_civ6` runs with a seed and `--device cuda:0`
- **THEN** `torch.use_deterministic_algorithms` is on, cuDNN is deterministic with benchmarking off, and `CUBLAS_WORKSPACE_CONFIG` is set, before the network is constructed

#### Scenario: CPU runs are untouched
- **WHEN** a seeded run uses `--device cpu`
- **THEN** no torch backend switch is changed and the CPU determinism gates produce the same results as before

#### Scenario: Two seeded runs on one GPU
- **WHEN** the same seeded training, or the same seeded arena, runs twice on one CUDA device of one machine with one torch build
- **THEN** the saved weights, and the per-game arena records, are identical

### Requirement: A timing names the device it ran on

A throughput calibration MUST record the device the games ran on, and MUST refuse
an explicitly requested CUDA device that the machine cannot run instead of timing
another device under its name.

#### Scenario: A GPU that is not there
- **WHEN** `calibrate_throughput --device cuda:1` runs on a machine with no such device
- **THEN** it exits 2, plays no game and writes no file
