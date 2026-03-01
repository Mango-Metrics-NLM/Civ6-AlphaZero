# Civ6 AlphaZero Inference Service

Production ML inference service for real-time MCTS-based gameplay.

## Quick Start

### 1. Deploy to Cloud Run

```bash
# Build and deploy
./scripts/deploy_inference.sh cloud-run v1.0.0

# Get service URL
SERVICE_URL=$(gcloud run services describe civ6az-inference \
  --region=us-central1 \
  --format="value(status.url)")

echo "Service: $SERVICE_URL"
```

### 2. Test with Python Client

```python
from alpha_zero.inference import InferenceClient, create_random_game_state

# Connect to service
client = InferenceClient("https://civ6az-inference.run.app")

# Check health
health = client.health()
print(f"Status: {health.status}, Model: v{health.model_version}")

# Run inference
game_state = create_random_game_state(board_size=8)
response = client.infer(game_state, mcts_simulations=25)

print(f"Action: {response.action_distribution.action_id}")
print(f"Value: {response.action_distribution.value_estimate:.3f}")
```

### 3. Run Examples

```bash
# Quick start examples
python examples/inference_quickstart.py \
  --url https://civ6az-inference.run.app

# Load testing
python scripts/load_test_inference.py \
  --url https://civ6az-inference.run.app \
  --concurrent-games 50 \
  --duration-seconds 60
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Performance metrics |
| `/v1/infer` | POST | Single inference |
| `/v1/batch_infer` | POST | Batch inference |
| `/v1/models/swap` | POST | Load model version |
| `/docs` | GET | Interactive API docs |

## Performance

| MCTS Sims | Latency (P50) | Use Case |
|-----------|---------------|----------|
| 16 | 450ms | Fast gameplay |
| 25 | 680ms | Standard gameplay |
| 50 | 1200ms | Competitive |
| 100 | 2300ms | Training quality |

## Architecture

```
Client → Load Balancer → Inference Pod (GPU) → GCS (Models)
```

## Files

- `api_schemas.py` - Request/response models
- `api_server.py` - FastAPI application
- `inference_engine.py` - Core inference logic
- `client.py` - Python SDK
- `README.md` - This file

## Documentation

See [docs/INFERENCE_SERVICE.md](../../docs/INFERENCE_SERVICE.md) for full documentation.

## Support

- API Docs: `<service-url>/docs`
- Health: `<service-url>/health`
- Metrics: `<service-url>/metrics`
