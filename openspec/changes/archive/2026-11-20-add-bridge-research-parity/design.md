# Design notes -- add-bridge-research-parity

## One source for the loaded Lua

The mod loads `Scripts/AlphaZeroMain.lua`, a concatenation of the module files.
The generator records its module order and separators explicitly, chosen so that
the first run reproduces the committed bundle byte for byte. Live behaviour
therefore does not move in GM-B0. Line endings are pinned, because the
workstation runs `core.autocrlf`, and an unpinned check would pass on one OS and
fail on the other.

## Lua under test

Civ6 runs Lua 5.1. The harness uses lupa's 5.1-compatible runtime, so `unpack`,
`#` and integer semantics match the game. The repository's import gate requires
`importorskip`. In the CI job that installs the extra, a skip is turned into a
failure, because a skip is not a pass.

## What the bridge sends, and why END_TURN is not additive

New state fields are optional with defaults, and a payload without
`schema_version` reads as legacy. END_TURN reporting a blocked turn changes the
meaning of an existing acknowledgement: today any exception in the loop becomes
END_TURN. The Lua change and the Python handling (log, count, no exception, no
retry storm) ship in one PR. Research execution adds `UI.RequestPlayerOperation`
and keeps the GameCore setter as a fallback. Both are official surfaces (NG-3).

## Registry and snapshot

`tech_registry.py` builds name-to-slot maps through `alpha_zero.rules.registry`,
from a GameInfo-derived rule set under the data root. With none present, slots
stay zero, one WARNING is logged and a counter increments. The GM-B4 snapshot
writes the table format that `python -m alpha_zero.rules import-gameinfo` reads,
and holds only identifiers and integers. Snapshots live under `CIV6AZ_DATA_ROOT`
and never in the repository.

## Search proxy without rule changes

`microciv_proxy.py` changes only how a bridge state seeds `MicroCivEnv`: unit
classes onto MicroCiv's unit types, impassable tiles onto terrain the simulator
already refuses to enter. Anything that needs a new rule belongs to the GM-S
rungs. The proxy does not make `--mcts` usable: the network and action-space
mismatch remains. The mode therefore refuses at start-up with its gap id until an
action translator exists, which is a later GM-B row.
