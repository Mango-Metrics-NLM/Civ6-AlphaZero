---
title: Civ6 AlphaZero ELO Dashboard
emoji: 📊
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: "4.20.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
tags:
  - reinforcement-learning
  - game-ai
  - civilization-vi
  - alphazero
  - evaluation
  - elo-rating
---

# 📊 Civ6 AlphaZero — ELO Dashboard

Track model improvement through ELO ratings, win-rate matrices, and
head-to-head comparisons across training iterations.

## What is this?

This dashboard visualises how the Civ6 AlphaZero models improve over
successive training runs using ELO tournament data published to the Hub.

- **Dataset**: [`ianshank/civ6-elo-tournament`](https://huggingface.co/datasets/ianshank/civ6-elo-tournament)
- **Algorithm**: Sequential testing + ELO rating system

## Features

| Feature | Status |
| --- | --- |
| ELO rating progression chart | 🚧 Scaffolded — requires dataset |
| Win-rate matrix | 🚧 Scaffolded |
| Head-to-head model comparison | 🚧 Scaffolded |
| Live tournament results | 🚧 Planned |

## Status

> **Note**: This Space is currently scaffolded. Charts will be populated
> once tournament data is published to the Hub dataset.
> Track progress at the
> [Civ6-AlphaZero GitHub repo](https://github.com/ianshank/Civ6-AlphaZero).

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HF_DATASET_REPO` | `ianshank/civ6-elo-tournament` | Dataset repo to load from Hub |
| `HF_ORG` | `ianshank` | HF organisation override |

## Citation

```bibtex
@software{civ6_alphazero_2025,
  author = {ianshank},
  title  = {Civ6 AlphaZero},
  year   = {2025},
  url    = {https://github.com/ianshank/Civ6-AlphaZero}
}
```
