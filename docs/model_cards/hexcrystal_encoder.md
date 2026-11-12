---
license: mit
library_name: alpha-zero-hexcrystal
tags:
  - pytorch
  - hexagonal-grid
  - spatial-encoder
  - reinforcement-learning
  - alphazero
---

# Hex-Crystalline Spatial Encoder

A spatial encoder for **hexagonal grids represented on a rectangular array**,
trained inside an AlphaZero-style agent for a simplified 4X strategy game. It
depends on nothing but PyTorch, so it can be reused for any hex-lattice problem —
board games, hex-tiled sensor arrays, or other six-neighbour topologies — without
the surrounding game code.

```python
from alpha_zero.hexcrystal.hub_mixin import HexSpatialEncoderForHub

encoder = HexSpatialEncoderForHub.from_pretrained("ianshank/civ6-hexcrystal-encoder")
tokens = encoder(spatial)  # (B, C, H, W) -> (B, num_tokens, embed_dim)
```

Install with `pip install -e '.[hub]'` from the source repository.

## What this model actually does

Read this section before assuming behaviour from the name. Earlier documentation
in the source repository described features this encoder does **not** have; that
drift is corrected here and in the code.

The forward pass, in order:

1. **Even-q offset hexagonal convolution.** A 7-tap kernel — the centre tile plus
   its six neighbours — applied by gathering pre-computed adjacency indices.
   Off-board neighbours carry a `-1` sentinel and are masked. The stack runs at
   constant resolution: there is no pyramid, no dilation, and no configurable
   ring radius.
2. **Adaptive average pooling to a fixed token grid, then a linear projection.**
   This is the encoder's most consequential property: **pooling discards fine
   spatial structure** before tokenization. If your task needs per-tile fidelity
   at the output, this bottleneck is the first thing to reconsider.
3. **Optional crystalline lattice layer.** A set of learned transforms
   initialised to a symmetry group (hexagonal, cubic, or tetrahedral), combined
   by a softmax-weighted sum or a small attention over the **transform axis**.
   This is not spatial attention and involves no hex-neighbour masking.

What it does **not** contain, despite claims that circulated in earlier docs:

| Claimed | Reality |
|---|---|
| Axial coordinates | Even-q offset only; other systems raise `ValueError` |
| Ring-distance positional encoding | No positional encoding at all — position comes from the downstream transformer's learned `pos_embed` |
| Multi-scale convolutions | Flat, same-resolution stack |
| Attention over hex neighbourhoods | Attention is over transforms, not tiles |

## Intended use

**Intended.** A drop-in replacement for patch embedding when your input has hex
topology; a starting point for hex-aware architectures; a reference
implementation of even-q offset adjacency in PyTorch.

**Not intended.** This is **not** a strong game-playing model. It is one
component of an agent trained at small scale for research purposes. Do not deploy
it as a Civilization VI opponent or treat its weights as competitive.

## Training

Trained as the spatial encoder of a transformer policy/value network, via
AlphaZero-style self-play with Gumbel MCTS on **MicroCiv**, a simplified 4X
environment with a 13-channel state encoding on a square board. Training used
self-play trajectories only; there is no human gameplay data, and no external
dataset is involved.

## Evaluation

The two encoders were trained from identical code, seed, and budget, differing
only in encoder type, then played head to head. Because the same base seed was
used for training and for the arena, 280 of the 800 arena maps were also
training maps; the result below is therefore partly in-sample. Full methodology and the
generating script are in the source repository (`scripts/run_encoder_ab.py`).

### Metrics

| hex_crystalline vs patch_embed | Value |
|---|---|
| Games | 800 (paired-seat design was **not** used; seats alternated) |
| W / D / L | 142 / 588 / 70 |
| Score rate (draw = 0.5) | 0.545 |
| 95% CI, ternary variance | [0.527, 0.562] |
| 95% CI, Wilson on fractional successes (as originally reported; overstated) | [0.510, 0.579] |
| Draw rate | 0.735 |
| Device | CUDA (the earlier text of this card said CPU; the artifact's provenance says `cuda`) |
| Search epoch | pre-2026-09-04 (see `docs/evidence/README.md`) |

**Evaluation status: superseded.** An 800-game head-to-head (2026-08-10, commit `b14534a`, CUDA) scored hex_crystalline 142 W / 588 D / 70 L against patch_embed (score rate 0.545; ternary 95% CI 0.527–0.562, game-sampling noise only). This result is **not evidence that either encoder is better**: both arms were trained once each, with a search implementation later found to back up values with the wrong sign on same-player moves and to train on one-hot policy targets (fixed 2026-09-04), and 280 of the 800 evaluation games reused training seeds. A re-run on disjoint seeds with the corrected search is LF-A9 (milestone v2.3, not v2.1). It waits until `scripts/run_encoder_ab.py` draws its seeds from named streams: re-run as it stands at seed 42, 560 of its 800 arena games would reuse a training map (`kb/specs/20261111-seed-overlap-memo.md` §1.3). v2.1's learning run (A1, `docs/EVALUATION.md` section 7.1) does not compare encoders. The encoder's architecture description is unaffected.


### Evaluation scope and limitations

Read the confidence interval, not just the point estimate.

- **Reduced scale.** MicroCiv with a 13-channel encoder on a small board, a
  modest iteration budget, and a single GPU training run per arm. These are not full-Civilization VI
  results and do not extrapolate to one.
- **A single training run per arm.** Training-run variance is therefore **not
  captured**: the interval reflects game-sampling noise only, and a different
  seed could plausibly reorder the arms.
- **Differences smaller than the interval half-width are not resolved.** Where
  the interval spans 0.5, the honest reading is that this run does not
  distinguish the two encoders — not that they are equal, and not that the
  point estimate is real.
- **Draws are scored 0.5** in the score rate; win/draw/loss counts are reported
  separately so the raw outcome is recoverable.

The value of this evaluation is that it is reproducible, same-code, same-seed,
and bounded — not that it settles which encoder is better.

## Risks, limitations and ethical considerations

- A research artifact trained on self-play in a toy environment; it encodes no
  human data and carries no personal information.
- It is not a game opponent and must not be presented as one; its strength has
  not been demonstrated (see Metrics).
- Misuse risk is low; the main risk is over-reading the evaluation above.

## Architecture details

| Property | Value |
|---|---|
| Coordinate system | Even-q offset (flat-top hexes) |
| Kernel | 7-tap (centre + 6 neighbours) via gather |
| Symmetry groups | `hexagonal` (6), `cubic` (8), `tetrahedral` (12) transforms |
| Tokenization | Adaptive average pool → linear projection |
| Output shape | `(batch, num_output_tokens, embed_dim)` |
| Weights format | safetensors |
| Dependencies | PyTorch only |

Configuration is a flat `config.json` and round-trips through
`HexEncoderConfig.from_dict`. The adjacency indices are registered buffers, so
they travel with `.to(device)` and appear in the `state_dict`.

## Reproducing

```bash
python scripts/run_encoder_ab.py train --iterations N --games-per-iter M
python scripts/run_encoder_ab.py arena --num_games K
python scripts/run_encoder_ab.py report
```

Both arms take the same seed, and the environment seed is threaded into
self-play so game streams are reproducible — seeding the global RNGs alone is
not sufficient, because the environment holds its own generator.

## Citation

If you use this encoder, please cite the repository. The exact commit that
produced these weights is recorded in the evidence report alongside the
training budget.
