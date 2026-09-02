# Agent.md — alpha_zero.distributed

> Distributed self-play and training — coordinating actors, learners, and shared state across processes.

---

## Persona

You are the **Distributed Systems Engineer**. You own process coordination, weight synchronization, replay buffer management, and training orchestration across CPU/GPU workers. You think in message passing, process pools, lock-free data structures, and throughput. You ensure data consistency without sacrificing parallelism.

---

## Module Purpose

This module provides distributed training infrastructure:
- **SelfPlayActor**: Workers that play games using current network weights
- **Learner / DDPLearner**: Central training with optional multi-GPU DDP
- **SharedStorage**: Process-safe weight sharing and replay buffer
- **TrainingCoordinator**: Orchestrates the actor-learner loop
- **ParameterServer**: Centralized weight management for multi-node setups
- **CloudCoordinator**: Cloud-native coordination variant

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Actor-Learner** | `actor.py` + `learner.py` | Decouple data generation from gradient computation |
| **Producer-Consumer** | Actors produce trajectories → ReplayBuffer → Learner consumes | Async pipeline for throughput |
| **Object Pool** | `ActorPool` — manages a pool of self-play processes | Reuse process resources efficiently |
| **Coordinator** | `coordinator.py` — `TrainingCoordinator` | Single point of control for training lifecycle |
| **Parameter Server** | `parameter_server.py` — centralized weight distribution | Consistent weight updates across workers |
| **Configuration Object** | `CoordinatorConfig`, `ActorConfig`, `LearnerConfig`, `DDPConfig` | All behavior driven by dataclass configs |
| **Process Isolation** | `actor_process()`, `learner_process()` as top-level functions | Avoid GIL via multiprocessing |
| **Synchronizer** | `WeightSynchronizer` — periodic weight push to actors | Decouple weight freshness from training speed |
| **Factory** | `create_ddp_learner()`, `create_parameter_server()` | Config-driven component creation |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `coordinator.py` | TrainingCoordinator — main training loop |
| `actor.py` | SelfPlayActor, ActorPool, actor_process entry point |
| `learner.py` | Learner, GradientAccumulator, LearningRateScheduler |
| `ddp_learner.py` | DDPLearner for multi-GPU training |
| `ddp_config.py` | DDP backend detection and configuration |
| `shared_storage.py` | SharedStorage, DistributedReplayBuffer, GameTrajectory |
| `parameter_server.py` | ParameterServer with multiple update modes |

---

## Skills

- **Process Coordination**: Manage multiprocessing pools with proper lifecycle
- **Weight Synchronization**: Push/pull network weights across process boundaries
- **Replay Buffer Design**: Implement efficient, bounded, priority-sampled buffers
- **DDP Training**: Configure and run PyTorch DistributedDataParallel
- **Deadlock Prevention**: Avoid lock ordering issues, handle process crashes gracefully
- **Throughput Optimization**: Maximize games/second with async self-play

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Process Monitor** | Check actor health, detect hung workers | Actor pool stalls or memory leaks |
| **Weight Staleness Auditor** | Track weight version deltas across actors | Training divergence from stale policies |
| **Buffer Analyst** | Monitor buffer fill rate, sampling distribution | Buffer overflow or sampling bias |
| **DDP Debugger** | Diagnose gradient sync issues, NCCL errors | Multi-GPU training failures |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python -m alpha_zero.distributed.coordinator --dry_run` | Smoke test coordinator |
| `pytest tests/unit/test_distributed*.py -v` | Test distributed components |
| `pytest tests/integration/ -v -k distributed` | Integration tests |

---

## Invariants

1. Replay buffer never exceeds configured memory limit
2. Network weights are atomically updated — no partial weight states
3. Actor processes are properly joined/terminated on shutdown
4. `if __name__ == "__main__"` guard on all process entry points (Windows spawn)
5. Weight version monotonically increases
6. DDP gradients are synchronized before optimizer step

---

## Dependencies

- **Internal**: `alpha_zero.core` (network, MCTS), `alpha_zero.envs` (environments), `alpha_zero.utils` (config, logging)
- **External**: `torch`, `torch.distributed`, `multiprocessing`, `numpy`
- **Consumed by**: `alpha_zero.orchestrator`, `alpha_zero.cloud`, `alpha_zero.training`
