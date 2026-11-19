# Design notes -- add-game-rules-kernel

## Home and layering

`alpha_zero/rules/` is a leaf: standard library and `yaml`, no `alpha_zero`
import, no torch or numpy, and no per-game state. Rejected homes: `mechanics/`
(coverage-omitted, warns on import), `gameplay/rules/` (on the play path),
`bridge/` (wrong dependency direction for the simulator later), `configs/` (not
installed into the pinned wheel run cards use). Because it is a leaf, the
importer does not use `alpha_zero.utils.data_root`. It reads `CIV6AZ_DATA_ROOT`
from `os.environ` and finds the checkout as the nearest ancestor of the package
holding `pyproject.toml`.

## Liveness from the first PR

A module is dead until something other than a test imports it, and a dead module
still counts as an importer of what it imports. GM-K1 therefore ships the CLI's
`validate` and `hash` subcommands. The entry point reaches `builtin`, which
reaches `loader`, `validate`, `queries`, `registry` and `spec`. GM-K2 adds
`pace` (`pacing.py`) and `import-gameinfo` (`gameinfo.py`). GM-B3's bridge
registry becomes a second importer later.

## Strict, integer-only data

`yaml.safe_load` alone accepts duplicate keys (the last one wins) and YAML 1.1
booleans (`yes`, `on`). The loader uses a `SafeLoader` subclass that rejects
duplicate keys. The spec constructors reject unknown keys, floats and
booleans where integers are expected. Percentages are integer percent.

## Identity

A rule set is `(id, sha256)`. The hash is taken over
`json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`,
so CRLF in a Windows checkout and key order cannot move it. A content change is a
new id (`science-v2`), never a re-pin. `docs/GAME_RULES.md` keeps the append-only
log, and a test holds it equal to `builtin.py`, as the A1 registration digest is held.

## Victory and imported trees

The capstone rule applies only when a `VictorySpec` names a tech. An imported
real tree declares none, so it validates without one.

## GameInfo input

`import-gameinfo IN --out OUT` reads a JSON object of tables (`Technologies`,
`TechnologyPrereqs`, `Civics`, `CivicPrereqs`, `Buildings`, `Units`) holding
identifiers and integers only. It writes a rule-set YAML. The output must lie
outside the checkout, and defaults to `<CIV6AZ_DATA_ROOT>/gameinfo/`. GM-B4's
snapshot writes this format. Real tables are never committed; tests use a
synthetic fixture.
