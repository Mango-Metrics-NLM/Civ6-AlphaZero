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

### Requirement: GPU reproduction is checked where the runs execute, before they count

Before any registered run on GPUs is started, a reproduction check MUST have
passed on the machine that will run it: a small run of the registered shape,
trained and judged on the registered devices, executed twice with each repeat in
its own process inside the pinned environment, and compared checkpoint tensor by
checkpoint tensor and arena game by arena game. A difference, an error, or the
check not having run on the registered build and GPUs MUST mean the GPU
registration does not stand.

#### Scenario: Two repeats agree
- **WHEN** the check runs twice on the registered devices and every checkpoint tensor and arena game matches
- **THEN** it reports `REPRODUCED` and exits 0

#### Scenario: A repeat differs
- **WHEN** any checkpoint tensor or arena game differs between the two repeats
- **THEN** it reports `DIFFERS`, names what differs and exits non-zero

#### Scenario: Nothing was compared
- **WHEN** a comparison finds no component to compare
- **THEN** it does not report a pass

### Requirement: A re-run after an interruption is a reproduction check

When an interrupted registered run is re-run at the same seed, every component
the interrupted attempt finished MUST be compared with the re-run's. A re-run
that does not reproduce MUST NOT be read as a PASS.

#### Scenario: The re-run reproduces
- **WHEN** every checkpoint, iteration record and completed arm of the interrupted attempt equals the re-run's
- **THEN** the re-run is the attempt that is judged

#### Scenario: The re-run differs
- **WHEN** a checkpoint tensor, a non-timing field of an iteration record or an arena game differs
- **THEN** the re-run is reported as non-reproducing and a PASS row is read as INCONCLUSIVE

### Requirement: Arena games are recorded move for move

Each arena per-game record MUST carry, beside its pair, seat and result, the
number of moves played and a digest of every action of the game in order. The
reproduction check and the determinism check MUST compare whole records, so two
runs that reach the same outcomes by different moves are reported as different.
The added keys MUST NOT change how a game is played or how a result is scored.

#### Scenario: Same outcome, other moves
- **WHEN** two runs agree on every game's result and one game's action digest differs
- **THEN** the comparison reports `DIFFERS` and names the game

#### Scenario: A seeded arena is repeated
- **WHEN** the same seeded arena is played twice on CPU
- **THEN** every record, digest included, is identical
