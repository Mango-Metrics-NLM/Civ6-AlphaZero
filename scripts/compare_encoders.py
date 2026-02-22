#!/usr/bin/env python3
"""Compare PatchEmbed vs HexCrystalline encoder performance.

Measures forward/backward latency, memory usage, parameter counts,
and output statistics for both spatial encoder types.

Usage:
    python scripts/compare_encoders.py --batch_size=32 --num_iters=100
    python scripts/compare_encoders.py --batch_size=8 --num_iters=50 --device=cpu
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import torch

from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single encoder benchmark."""

    encoder_type: str
    param_count: int
    encoder_param_count: int
    forward_latency_ms: float
    backward_latency_ms: float
    peak_memory_mb: float
    output_mean: float
    output_std: float


def benchmark_encoder(
    encoder_type: str,
    batch_size: int,
    num_iters: int,
    device: str,
) -> BenchmarkResult:
    """Benchmark a specific encoder type.

    Args:
        encoder_type: Encoder type string ("patch_embed" or "hex_crystalline").
        batch_size: Batch size for benchmarking.
        num_iters: Number of iterations to average over.
        device: Device to run on ("cpu" or "cuda").

    Returns:
        BenchmarkResult with timing and memory metrics.
    """
    config = TransformerConfig(encoder_type=encoder_type)
    network = Civ6TransformerNetwork(config).to(device)
    network.train()

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    encoder_params = sum(p.numel() for p in network.encoder.patch_embed.parameters())

    # Create dummy inputs
    spatial = torch.randn(batch_size, 13, 8, 8, device=device)
    vector = torch.randn(batch_size, 64, device=device)

    # Warmup
    for _ in range(min(5, num_iters)):
        policy, value = network(spatial, vector)
        loss = policy.sum() + value.sum()
        loss.backward()
        network.zero_grad()

    # Reset memory tracking
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Forward pass timing
    forward_times = []
    for _ in range(num_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy, value = network(spatial, vector)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_times.append((t1 - t0) * 1000)  # ms

        # Cleanup for next iteration
        policy.detach()
        value.detach()

    # Backward pass timing
    backward_times = []
    for _ in range(num_iters):
        policy, value = network(spatial, vector)
        loss = policy.sum() + value.sum()

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        backward_times.append((t1 - t0) * 1000)  # ms
        network.zero_grad()

    # Memory tracking
    peak_memory_mb = 0.0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Output statistics
    network.eval()
    with torch.no_grad():
        # Get encoder output (patch embeddings)
        patch_out = network.encoder.patch_embed(spatial)
        output_mean = patch_out.mean().item()
        output_std = patch_out.std().item()

    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)

    return BenchmarkResult(
        encoder_type=encoder_type,
        param_count=total_params,
        encoder_param_count=encoder_params,
        forward_latency_ms=avg_forward,
        backward_latency_ms=avg_backward,
        peak_memory_mb=peak_memory_mb,
        output_mean=output_mean,
        output_std=output_std,
    )


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table.

    Args:
        results: List of benchmark results to compare.
    """
    print("\n" + "=" * 72)
    print("ENCODER COMPARISON")
    print("=" * 72)

    header = f"{'Metric':<30} | "
    header += " | ".join(f"{r.encoder_type:>16}" for r in results)
    print(header)
    print("-" * len(header))

    # Parameter counts
    row = f"{'Total params':<30} | "
    row += " | ".join(f"{r.param_count:>16,}" for r in results)
    print(row)

    row = f"{'Encoder params':<30} | "
    row += " | ".join(f"{r.encoder_param_count:>16,}" for r in results)
    print(row)

    row = f"{'Encoder % of total':<30} | "
    row += " | ".join(f"{r.encoder_param_count / r.param_count * 100:>15.1f}%" for r in results)
    print(row)

    # Timing
    row = f"{'Forward latency (ms)':<30} | "
    row += " | ".join(f"{r.forward_latency_ms:>16.2f}" for r in results)
    print(row)

    row = f"{'Backward latency (ms)':<30} | "
    row += " | ".join(f"{r.backward_latency_ms:>16.2f}" for r in results)
    print(row)

    row = f"{'Total (fwd+bwd) (ms)':<30} | "
    row += " | ".join(f"{r.forward_latency_ms + r.backward_latency_ms:>16.2f}" for r in results)
    print(row)

    # Memory
    if any(r.peak_memory_mb > 0 for r in results):
        row = f"{'Peak memory (MB)':<30} | "
        row += " | ".join(f"{r.peak_memory_mb:>16.1f}" for r in results)
        print(row)

    # Output statistics
    row = f"{'Output mean':<30} | "
    row += " | ".join(f"{r.output_mean:>16.4f}" for r in results)
    print(row)

    row = f"{'Output std':<30} | "
    row += " | ".join(f"{r.output_std:>16.4f}" for r in results)
    print(row)

    # Relative comparison
    if len(results) >= 2:
        base = results[0]
        comp = results[1]
        print("\n" + "-" * 72)
        print(f"Relative ({comp.encoder_type} vs {base.encoder_type}):")
        param_ratio = comp.param_count / base.param_count
        print(f"  Params:          {param_ratio:.2f}x ({comp.param_count - base.param_count:+,})")
        fwd_ratio = comp.forward_latency_ms / base.forward_latency_ms
        print(f"  Forward latency: {fwd_ratio:.2f}x")
        bwd_ratio = comp.backward_latency_ms / base.backward_latency_ms
        print(f"  Backward latency: {bwd_ratio:.2f}x")

    print("=" * 72)


def main() -> None:
    """Run encoder comparison benchmark."""
    parser = argparse.ArgumentParser(description="Compare spatial encoder performance")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto-detect)",
    )
    args = parser.parse_args()

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Benchmark config: batch_size={args.batch_size}, num_iters={args.num_iters}, device={device}"
    )

    encoder_types = ["patch_embed", "hex_crystalline"]
    results = []

    for enc_type in encoder_types:
        print(f"\nBenchmarking {enc_type}...")
        result = benchmark_encoder(enc_type, args.batch_size, args.num_iters, device)
        results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    main()
