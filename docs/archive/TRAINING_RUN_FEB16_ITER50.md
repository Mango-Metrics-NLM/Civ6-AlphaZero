# Training Run Report: Iteration 049-050 (Feb 16, 2026)

## Overview
*   **Run ID**: `run-20260216_iter50` (Vertex AI)
*   **Configuration**:
    *   **Iterations**: 50 (Iter 049 completed at 10:21 AM)
    *   **Workers**: 2 (25 games/worker)
    *   **MCTS**: GumbelMCTS (8 simulations)
    *   **Evaluation**: 25 games, 55% Win Threshold, 10 Epochs
    *   **Learning Rate**: `warmup_cosine` (Value: 0.1x, Policy: 1.0x)
*   **Cost**: ~$7.25 total ($0.145/iter)
*   **Status**: Healthy - Pipeline completed without stalls.

## Model Progression
Significant promotions observed, indicating learning despite noisy evaluation signals:
*   Initial Version: `v079`
*   Current Final Version: `v129`
*   **Key Promotion Steps**:
    *   `v080`
    *   `v088`
    *   `v095`
    *   `v110`
    *   `v115`
    *   `v129`
*   **Win Rate Trend**: Positive (+0.056), oscillating between 0.2 and 0.6. Recent trend: 0.50-0.56.

## Operational Metrics
*   **Total Duration**: ~25 mins per full iteration cycle.
*   **Hardware**: N1-Standard-8 (T4 GPU budget tier).
*   **Throughput**: Consistent, no stalls detected.

## Next Steps
1.  **Download Final Model**: Fetch `model-v129` from GCS.
2.  **Champion Tournament**: Run a round-robin tournament between `v079` (baseline), `v095` (mid-training), and `v129` (final) to robustly verify strength.
3.  **Analysis**: Investigate the noisy win rate (0.2-0.6). Possible causes: High value loss variance, sensitive hyperparameters, or opponent randomization.
