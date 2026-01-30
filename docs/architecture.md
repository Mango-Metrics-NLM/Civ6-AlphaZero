# C4 Architecture Documentation

## Context Diagram

```mermaid
C4Context
    title System Context diagram for Civ6 AlphaZero AI

    Person(user, "User/Researcher", "Interacts with the system to train agents or evaluate performance.")
    System(civ6_ai, "Civ6 AlphaZero AI", "An AI agent trained to play a simplified Civilization VI using AlphaZero methodology.")

    Rel(user, civ6_ai, "Trains, Evaluates, Observes")
```

## Container Diagram

```mermaid
C4Container
    title Container diagram for Civ6 AlphaZero AI

    Person(user, "User/Researcher", "Runs training and evaluation scripts")

    System_Boundary(c1, "Civ6 AlphaZero System") {
        Container(env, "MicroCiv Environment", "Python/Gymnasium", "Simulates the Civ6 game rules and state.")
        Container(mcts, "MCTS Engine", "Python", "Monte Carlo Tree Search for planning.")
        Container(network, "Neural Network", "PyTorch", "Predicts policy and value.")
        Container(trainer, "Training Loop", "Python", "Orchestrates self-play and optimization.")
        Container(logger, "Logging", "WandB", "Tracks metrics and performance.")
    }

    Rel(user, trainer, "Starts training")
    Rel(trainer, env, "Interacts with")
    Rel(trainer, mcts, "Uses for data generation")
    Rel(mcts, network, "Queries for priors/values")
    Rel(mcts, env, "Simulates future states")
    Rel(trainer, logger, "Logs metrics")
```
