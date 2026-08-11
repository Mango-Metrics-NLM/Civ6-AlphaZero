---
title: MicroCiv Explorer
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
tags:
  - reinforcement-learning
  - game-ai
  - civilization-vi
  - alphazero
  - mcts
  - pytorch
---

# 🏛️ MicroCiv Explorer — Civ6 AlphaZero

Interactive explorer for the Civ6 AlphaZero policy/value network.

## What is this?

This Space lets you interact with a trained AlphaZero model that plays
a simplified Civilization VI environment (MicroCiv).

- **Model**: [`Mango-Metrics-NLM/civ6-alphazero-transformer`](https://huggingface.co/Mango-Metrics-NLM/civ6-alphazero-transformer)
- **Algorithm**: AlphaZero (MCTS + Transformer policy/value network)
- **Environment**: MicroCiv — a compact 8×8 hex-grid Civ6 variant

## Features

| Feature | Status |
| --- | --- |
| Board position analysis | 🚧 Scaffolded — requires checkpoint |
| Policy distribution visualisation | 🚧 Scaffolded |
| Value estimation | 🚧 Scaffolded |
| Interactive play vs AI | 🚧 Planned |

## Status

> **Note**: This Space is currently scaffolded. Full interactive gameplay
> will be enabled once a trained model checkpoint is published to the Hub.
> Track progress at the
> [Civ6-AlphaZero GitHub repo](https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero).

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HF_MODEL_REPO` | `Mango-Metrics-NLM/civ6-alphazero-transformer` | Model repo to load from Hub |
| `HF_ORG` | `Mango-Metrics-NLM` | HF organisation override |

## Citation

```bibtex
@software{civ6_alphazero_2025,
  author = {Ian Cruickshank},
  title  = {Civ6 AlphaZero},
  year   = {2026},
  url    = {https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero}
}
```
