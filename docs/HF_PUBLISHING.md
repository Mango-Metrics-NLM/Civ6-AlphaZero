# Hugging Face Hub Publishing Guide

## Overview

The Civ6-AlphaZero project supports publishing models, datasets, and interactive
demos to the Hugging Face Hub. All publishing tools are **dry-run by default** —
nothing is uploaded unless you explicitly pass `--publish`.

## Prerequisites

```bash
# Install the hub extra
pip install -e '.[hub]'

# For dataset publishing
pip install -e '.[datasets]'

# For Gradio Spaces
pip install -e '.[spaces]'

# Login to Hugging Face (one-time)
huggingface-cli login
```

## Publishing Models

### Supported Model Types

| Type | Class | Description |
|------|-------|-------------|
| `encoder` | `HexSpatialEncoderForHub` | Hex-crystalline spatial encoder |
| `transformer` | `Civ6TransformerNetworkForHub` | Policy/value transformer network |
| `muzero` | `MuZeroNetworkForHub` | MuZero world-model network |

### Upload Script

```bash
# Dry-run (default) — saves locally
python scripts/upload_to_hub.py \
    --model-type transformer \
    --checkpoint-path ./checkpoints/best.pt \
    --output-dir ./output/

# Publish to Hub
python scripts/upload_to_hub.py \
    --model-type transformer \
    --checkpoint-path ./checkpoints/best.pt \
    --repo-id ianshank/civ6-alphazero-transformer \
    --publish
```

### Programmatic Upload

```python
from alpha_zero.core.transformer_hub_mixin import Civ6TransformerNetworkForHub
from alpha_zero.hub.model_card import ModelCardMetadata
from alpha_zero.hub.upload import upload_model_to_hub

# Load or create model
model = Civ6TransformerNetworkForHub(embed_dim=256, num_layers=12)

# Upload
upload_model_to_hub(
    model=model,
    repo_id="ianshank/civ6-transformer",
    metadata=ModelCardMetadata(
        model_name="civ6-transformer",
        description="Civ6 AlphaZero policy/value network",
    ),
    dry_run=False,
)
```

### Loading from Hub

```python
from alpha_zero.core.transformer_hub_mixin import Civ6TransformerNetworkForHub

model = Civ6TransformerNetworkForHub.from_pretrained("ianshank/civ6-alphazero-transformer")
```

## Publishing Datasets

### ELO Tournament Results

```bash
python scripts/export_dataset.py \
    --type elo \
    --source-path ./results/ \
    --repo-id ianshank/civ6-elo-tournament \
    --publish
```

### Replay Buffer

```bash
python scripts/export_dataset.py \
    --type replay \
    --source-path ./replay_buffer/ \
    --max-samples 10000 \
    --publish
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (from `huggingface-cli login`) | HF API token |
| `HF_ORG` | `ianshank` | Hub namespace. The publishing account; a `Mango-Metrics-NLM` organisation does not exist on the Hub yet, so set this once it does |
| `HF_LIBRARY_NAME` | `alpha-zero-civ6` | Library name in Hub metadata |
| `HF_LICENSE` | `MIT` | SPDX licence identifier |
| `HF_CARD_TEMPLATE` | `alpha_zero/hub/templates/model_card.md.template` | Model card template path |

## Model Card Template

The model card is generated from a `string.Template`-based Markdown file at
`alpha_zero/hub/templates/model_card.md.template`. You can customise it by:

1. Editing the template file directly
2. Setting `HF_CARD_TEMPLATE` to point to a custom template
3. Passing a `ModelCardMetadata` with all fields populated

Available template variables: `$model_name`, `$description`, `$architecture`,
`$num_parameters`, `$training_details`, `$metrics`, `$tags_yaml`, `$license`,
`$library_name`, `$import_path`, `$class_name`, `$repo_id`, `$citation`.

## Architecture

```
alpha_zero/hub/              # Shared infrastructure
├── __init__.py              # Public API
├── constants.py             # Env-overridable constants
├── model_card.py            # Template-based card generator
├── upload.py                # Generic upload utility
├── datasets.py              # Dataset exporters
└── templates/
    └── model_card.md.template

alpha_zero/hexcrystal/
└── hub_mixin.py             # Encoder-specific hub integration

alpha_zero/core/
├── transformer_hub_mixin.py # Transformer-specific hub integration
└── muzero_hub_mixin.py      # MuZero-specific hub integration

scripts/
├── upload_to_hub.py         # CLI: upload models
└── export_dataset.py        # CLI: export datasets
```

## Testing

```bash
# All hub-related tests
pytest tests/unit/test_hub_* tests/unit/test_transformer_hub_mixin.py \
       tests/unit/test_muzero_hub_mixin.py -v

# Integration tests
pytest tests/integration/test_hub_publish_pipeline.py -v

# Security tests
pytest tests/security/test_hub_token_safety.py -v
```
