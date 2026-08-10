# Design — hexcrystal-encoder-package

## Context

The four encoder modules are torch-only by source but unreachable in a torch-only
environment, because importing any of them executes the root package's eager
import chain. Every other constraint in this change follows from three facts
measured on 2026-08-09:

- There is exactly **one** production consumer of the encoder registry:
  `TransformerEncoder.__init__` at `core/transformer_network.py:824`, which
  performs a function-local import and assigns the result to `self.patch_embed`
  regardless of encoder type (`:823` comments that the attribute name is
  preserved for `state_dict` compatibility). Everything else — cloud workers,
  scripts, configs — passes the encoder type as a plain string.
- The registry has a back-edge: `encoder_registry.py:141` lazily imports
  `PatchEmbed` from `transformer_network.py`, which is why the registry cannot
  ship standalone without either moving `PatchEmbed` or inverting registration.
- `HexEncoderConfig` is twelve primitive fields, already JSON-clean, whereas
  `TransformerConfig` carries plain-`Enum` fields and two `field(init=False)`
  derived fields that must be popped on reconstruction.

## Goals / Non-Goals

**Goals.** Make the encoder importable and reusable with torch alone; give it a
hub-standard load path; produce honest comparative evidence; correct the
architecture documentation.

**Non-Goals.** Changing encoder behavior or hyperparameters; altering the default
encoder; touching the cloud training pipeline's semantics; publishing a
competitive game-playing model; making `Civ6TransformerNetwork` itself
hub-loadable (its enum and derived-field handling make it materially harder, and
the encoder is the reusable unit).

## Decisions

### Where the code lives

A new `alpha_zero/hexcrystal/` subpackage, with re-export shims at the four
historical `core/*` paths. Setuptools discovery already includes it
(`include = ["alpha_zero*"]`), so no packaging change is needed. Shims replace
the originals one-for-one, so the documented `core` file count is unchanged; only
a new module row and the submodule total move.

`PatchEmbed` moves into the subpackage rather than staying behind, because
leaving it would keep the registry dependent on `transformer_network` and thus on
the whole core package. It is 42 dependency-clean lines, and moving it is
weight-safe: `state_dict` keys derive from attribute names, not module paths.

### Breaking the eager-import chain

PEP 562 lazy `__getattr__`/`__dir__` on the root package. The `_compat` import
stays eager because it patches `datetime.UTC` before anything else can import it,
and `__version__` stays a literal assignment because the documentation-drift gate
parses it with a regex.

One consequence needs stating because it is a real, if small, surface change.
Today `__all__` is mutated after import based on whether the optional Civ6
environment imports successfully. Deciding that under lazy resolution would
require performing the very import being deferred, and the import-sanity suite
asserts that every name in `__all__` resolves to a non-`None` object. So `__all__`
becomes static and omits the optional pair; both names remain resolvable as
attributes. Direct imports keep working; only a wildcard import stops pulling
them in.

### Hub integration

An opt-in module containing a subclass — the base encoder class is untouched and
keeps its torch-only dependency set. A conditional base class on the base type
would have been the obvious alternative, but it type-checks badly under this
repository's strict settings (inconsistent MRO and "variable not valid as a type"
under a `try`/`except ImportError` guard), and `warn_unused_ignores` makes a
speculative suppression an error once the dependency is present. Making the whole
module opt-in avoids the conditional entirely.

The subclass takes flat primitive keyword arguments and builds the configuration
internally, so the mixin's captured `config.json` is flat JSON. This was verified
end to end against huggingface_hub 0.36.2: safetensors serializes the integer
adjacency buffers without complaint, there are no tied tensors, and an offline
round trip reproduces outputs exactly. Note that `save_pretrained` writes a stub
README, which the upload tool overwrites with the real model card.

The hub dependencies are mirrored into the development extra as well as the new
`hub` extra. Without that, continuous integration would skip the round-trip test
and simultaneously count the module as uncovered, since subprocess-based tests do
not report coverage.

### Evidence

A dedicated driver composes existing pieces rather than adding a training path:
model construction already accepts an encoder type, self-play already runs
without cloud storage or credentials, and the arena already accepts two
independently-configured models. The local training entry point is unsuitable as
a host because it consumes trajectories from cloud storage; the other candidate
hardcodes a different network family entirely.

Seeding deserves care. Seeding the global random state covers weight
initialization and sampling, but the environment holds its own generator and
self-play constructs environments without passing a seed, so game streams would
still vary. The driver therefore threads an optional seed into environment
construction (defaulting to current behavior), and the determinism test is scoped
to what seeding actually guarantees rather than asserting more than is true.

Sample size is a correctness matter, not a tuning knob: at a few hundred games
the confidence interval on a win rate is wide enough to make a small difference
indistinguishable from noise. The evidence therefore reports an interval
alongside the point estimate, and the model card is forbidden from claiming a
difference the interval does not support. Because each arm is trained once,
training-run variance is explicitly declared uncaptured.

### Publication

Publishing is a side-effecting external call, so it is human-confirmed rather
than automated: the tool defaults to a dry run, requires an explicit flag, and
reads its credential only from the environment. The hex arm is published
regardless of which encoder wins, because the artifact's subject is the
hex-crystalline encoder and a negative result is publishable evidence — and
because the extraction path deliberately refuses a mismatched checkpoint.

## Risks / Trade-offs

The riskiest edit is the root-package lazification, since it changes import
semantics for every consumer. It is isolated in its own commit so it can be
reverted without disturbing the extraction, and the surrounding blast radius was
measured as small: no runtime call sites import from the root package, nothing
reads the optional-availability flag, and no documentation build introspects the
package.

The extraction's failure mode — drifted weight keys — is guarded by a key-set
snapshot captured before the move rather than written after it, so the test
cannot be quietly conformed to a regression.
