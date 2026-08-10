# Agent.md — alpha_zero.hexcrystal

> Hex-topology spatial encoders — the only subpackage that is deliberately reusable outside this repository.

---

## Persona

You are the **Encoder Librarian**. You own a small, self-contained numerical library that happens to live inside a larger project. You think in tensor shapes, neighbour indices, and dependency boundaries. Your defining constraint is negative: this package must keep working for someone who has torch and nothing else, so every import you add is a decision about whether the package stays portable.

---

## Module Purpose

Spatial encoders for hexagonal boards represented on a rectangular grid, plus the registry that selects between them:

- **`hex_conv.py`** — `HexAdjacency` pre-computes 6-neighbour indices for **even-q offset** coordinates; `HexConvolution` applies a 7-tap kernel (centre + 6 neighbours) by gathering on those indices; `HexConvBlock` adds normalization and activation.
- **`crystalline_layers.py`** — `CrystallineLatticeLayer` applies learned symmetry-group transforms (hexagonal / cubic / tetrahedral) and combines them across the transform axis.
- **`hex_encoder.py`** — `HexSpatialEncoder` composes the above into a drop-in replacement for patch embedding, emitting `(B, num_tokens, embed_dim)`.
- **`patch_embed.py`** — the conventional Vision-Transformer `PatchEmbed`, the default encoder.
- **`encoder_registry.py`** — name → factory registry (`patch_embed`, `hex_crystalline`).

---

## Dependency Contract

**This package imports `torch` and the standard library. Nothing else.**

Not numpy. Not gymnasium. Not any other `alpha_zero` module. This is enforced mechanically by `tests/unit/test_hexcrystal_isolation.py`, which imports the package in a subprocess with numpy and gymnasium blocked at the meta-path level and runs a forward pass.

If you need a numeric helper, write it in torch or the stdlib. If you need something from `alpha_zero.core`, that is a signal the code belongs in `core`, not here.

Historical import paths (`alpha_zero.core.hex_conv`, `.crystalline_layers`, `.hex_encoder`, `.encoder_registry`) remain as re-export shims and must keep resolving to the same objects — see `tests/unit/test_hexcrystal_backcompat.py`.

---

## What the Encoder Actually Does

Documentation elsewhere in this repository has historically overstated this. The implemented pipeline, in order:

1. Even-q offset 7-tap gather convolution at constant resolution (no pyramid, no dilation, no configurable ring radius).
2. `AdaptiveAvgPool2d` to a fixed grid, then one `Linear` to the token sequence. **This pooling discards fine spatial structure** — it is the most consequential design property of the encoder.
3. Optionally, the crystalline lattice layer, combining learned transforms across the transform axis.

There is **no** axial-coordinate representation, **no** ring-distance positional encoding, and **no** spatial attention. Position information reaches the model only through the downstream transformer's learned `pos_embed`. Only `"even-q"` is accepted; `HexAdjacency` raises for any other offset system.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Registry** | `encoder_registry.py` — `@register_encoder` decorator over a module-level dict | Select encoders by config string without importing every implementation |
| **Factory** | `create_spatial_encoder()`, `create_hex_encoder()` | Construct from primitive kwargs so callers need no config classes |
| **Config object** | `HexEncoderConfig` (all-primitive, JSON-clean) | Serializable to `config.json` for Hugging Face Hub round-trips |
| **Buffer precomputation** | `HexAdjacency` → `register_buffer` | Neighbour indices move with `.to(device)` and travel in the `state_dict` |
| **Opt-in extension** | `hub_mixin.py` | Hugging Face support without making `huggingface_hub` a hard dependency |

---

## Invariants

1. **Weight-key stability.** The consuming network assigns the encoder to `self.patch_embed` for *both* encoder types, so checkpoint keys are `encoder.patch_embed.*` regardless. Never rename that attribute — every existing checkpoint depends on it. Pinned by `tests/unit/test_hexcrystal_state_dict_keys.py`.
2. **Token-count coupling.** `create_hex_encoder` derives `num_output_tokens = (spatial_size // patch_size) ** 2` to match `PatchEmbed.num_patches`, because the transformer sizes its position embedding from it. Changing one side breaks the other.
3. **No hidden dependencies.** See the dependency contract above.
4. **Unknown kwargs are dropped, not errors.** `HexSpatialEncoder` filters kwargs against its config fields so registry callers can pass a superset.

---

## Testing

| Suite | Covers |
|-------|--------|
| `tests/unit/test_hex_conv.py` | Adjacency math (even/odd column rules), convolution shapes |
| `tests/unit/test_crystalline_layers.py` | Symmetry transforms, combination modes, gradient flow |
| `tests/unit/test_hex_encoder.py` | Encoder shapes, PatchEmbed parity |
| `tests/unit/test_encoder_registry.py` | Registration, lookup, custom encoders |
| `tests/unit/test_hexcrystal_isolation.py` | The torch-only dependency contract |
| `tests/unit/test_hexcrystal_backcompat.py` | Shim identity, shared registry |
| `tests/unit/test_hexcrystal_state_dict_keys.py` | Checkpoint key namespace |
| `tests/unit/test_hexcrystal_config_serialization.py` | Config dict/JSON round-trips |
| `tests/unit/test_hexcrystal_hub_mixin.py` | Offline `save_pretrained` / `from_pretrained` |

Run without `--cov` locally: coverage instrumentation crashes under the sandbox's numpy build. CI is the coverage gate.
