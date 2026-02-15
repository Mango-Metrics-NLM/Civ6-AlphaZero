# Patent Case 3: Three-Level Hierarchical MCTS with Goal Propagation

## Architecture Diagram

```mermaid
flowchart TB
    subgraph STRATEGIC["Strategic Level (30-100 Turn Horizon)"]
        direction LR
        SG1["Domination\nVictory"]
        SG2["Science\nVictory"]
        SG3["Culture\nVictory"]
        SG4["Expand\nTerritory"]
        SG5["Military\nSupremacy"]
        SG6["Defend\nEmpire"]
        SGN["... 10 more\nStrategic Goals"]
    end

    subgraph TACTICAL["Tactical Level (5-15 Turn Horizon)"]
        direction LR
        TG1["Capture\nCity"]
        TG2["Build\nArmy"]
        TG3["Research\nTech"]
        TG4["Settle\nCity"]
        TG5["Build\nCampus"]
        TG6["Fortify\nPosition"]
        TGN["... 16 more\nTactical Goals"]
    end

    subgraph OPERATIONAL["Operational Level (1-3 Turn Horizon)"]
        direction LR
        OG1["Move\nUnit"]
        OG2["Attack\nTarget"]
        OG3["Build\nBuilding"]
        OG4["Produce\nUnit"]
        OG5["Select\nResearch"]
        OG6["Found\nCity"]
        OGN["... 14 more\nOp Goals"]
    end

    SG1 ==>|"Decompose"| TG1
    SG1 ==>|"Decompose"| TG2
    SG1 ==>|"Decompose"| TG6
    SG2 ==>|"Decompose"| TG3
    SG2 ==>|"Decompose"| TG5
    SG4 ==>|"Decompose"| TG4

    TG1 ==>|"Decompose"| OG1
    TG1 ==>|"Decompose"| OG2
    TG2 ==>|"Decompose"| OG4
    TG3 ==>|"Decompose"| OG5
    TG4 ==>|"Decompose"| OG1
    TG4 ==>|"Decompose"| OG6

    OG1 -.->|"Progress"| TG1
    OG2 -.->|"Progress"| TG1
    OG4 -.->|"Progress"| TG2
    TG1 -.->|"Weighted\nProgress"| SG1
    TG2 -.->|"Weighted\nProgress"| SG1

    style STRATEGIC fill:#1a2744,stroke:#4a9eff,stroke-width:3px,color:#fff
    style TACTICAL fill:#1a3324,stroke:#4aff7f,stroke-width:3px,color:#fff
    style OPERATIONAL fill:#2d1a44,stroke:#9f4aff,stroke-width:3px,color:#fff
```

## Goal Propagation System

```mermaid
flowchart LR
    subgraph PROPAGATION["GoalPropagator - Bidirectional Flow"]
        direction TB

        subgraph DOWN["Downward: Goal Decomposition"]
            direction LR
            D1["Strategic Goal"] --> D2["DecompositionStrategy\n.decompose()"]
            D2 --> D3["Tactical Sub-Goals\n(via DECOMPOSITION_MAP)"]
            D3 --> D4["DecompositionStrategy\n.decompose()"]
            D4 --> D5["Operational Sub-Goals\n(via DECOMPOSITION_MAP)"]
        end

        subgraph UP["Upward: Progress Aggregation"]
            direction RL
            U1["Operational\nProgress"] --> U2["Weighted Sum\n(by priority)"]
            U2 --> U3["Tactical\nProgress"]
            U3 --> U4["Weighted Sum\n(by priority)"]
            U4 --> U5["Strategic\nProgress"]
        end

        subgraph LATERAL["Lateral: Synergy / Conflict Detection"]
            direction LR
            L1["Goal A\n(Build Army)"] <-->|"Synergy\n+10% boost"| L2["Goal B\n(Build Encampment)"]
            L3["Goal C\n(Expand)"] <-->|"Conflict\n-10% priority"| L4["Goal D\n(Defend)"]
        end
    end

    style DOWN fill:#1a2744,stroke:#4aff7f,stroke-width:2px,color:#fff
    style UP fill:#1a2744,stroke:#4a9eff,stroke-width:2px,color:#fff
    style LATERAL fill:#1a2744,stroke:#ffbf4a,stroke-width:2px,color:#fff
```

## Decomposition Maps (Domain Knowledge)

```mermaid
mindmap
    root((Goal Hierarchy\n58+ Types))
        Strategic Goals\n16 Types
            Domination Victory
                Capture City
                Build Army
                Fortify Position
                Eliminate Unit
            Science Victory
                Research Tech
                Build Campus
                Develop City
            Culture Victory
                Build Theater
                Generate Tourism
                Build Wonder
            Expand Territory
                Settle City
                Improve Tile
                Secure Resource
            Military Supremacy
                Build Army
                Build Encampment
                Upgrade Unit
        Tactical Goals\n22 Types
            Capture City
                Move Unit
                Attack Target
                Siege City
            Build Army
                Produce Unit
                Upgrade Unit
            Settle City
                Move Unit
                Found City
            Research Tech
                Select Research
            Build Campus
                Build District
                Build Building
        Operational Goals\n20+ Types
            Move Unit
            Attack Target
            Produce Unit
            Build Building
            Found City
            Select Research
```

## Synchronize Hierarchy Lifecycle

```mermaid
sequenceDiagram
    participant GS as Game State
    participant GP as GoalPropagator
    participant GH as GoalHierarchy
    participant S as Strategic Goals
    participant T as Tactical Goals
    participant O as Operational Goals

    Note over GP: synchronize_hierarchy() called each turn

    GP->>O: Check Operational Goals
    O-->>GP: Progress updates
    Note over GP: Check deadlines,<br/>abandonment

    GP->>T: Check Tactical Goals
    Note over GP: Aggregate child progress<br/>(weighted by priority)
    T-->>GP: Updated progress

    GP->>S: Check Strategic Goals
    Note over GP: Aggregate child progress<br/>(weighted by priority)
    S-->>GP: Updated progress

    alt Goal Completed
        GP->>GH: Mark goal COMPLETED
        GP->>GH: Propagate completion upward
    end

    alt Goal Past Deadline
        GP->>GH: Mark goal FAILED
    end

    alt No Progress (20+ turns)
        GP->>GH: Mark goal ABANDONED
    end

    alt New Strategic Goal
        GP->>S: Create strategic goal
        GP->>GP: propagate_down()
        GP->>T: Auto-create tactical sub-goals
        GP->>GP: propagate_down()
        GP->>O: Auto-create operational sub-goals
    end
```

## Component Class Diagram

```mermaid
classDiagram
    class GoalPropagator {
        +hierarchy: GoalHierarchy
        +strategies: list~DecompositionStrategy~
        +propagation_hooks: list~Callable~
        +propagate_down(goal, game_state, context) PropagationResult
        +propagate_up(goal, game_state) PropagationResult
        +propagate_lateral(goal, game_state) PropagationResult
        +full_propagation(goal, state, ctx, dir) PropagationResult
        +synchronize_hierarchy(state, ctx) PropagationResult
    }

    class DecompositionStrategy {
        <<abstract>>
        +decompose(goal, state, ctx) list~Goal~
        +can_handle(goal) bool
    }

    class StrategicDecomposition {
        +DECOMPOSITION_MAP: dict
        +decompose(goal, state, ctx) list~Goal~
        +can_handle(goal) bool
        -_derive_target(parent, type, state) GoalTarget
    }

    class TacticalDecomposition {
        +DECOMPOSITION_MAP: dict
        +decompose(goal, state, ctx) list~Goal~
        +can_handle(goal) bool
        -_derive_target(parent, type, state) GoalTarget
    }

    class Goal {
        +id: str
        +level: HierarchyLevel
        +goal_type: GoalType
        +target: GoalTarget
        +priority: float
        +progress: float
        +status: GoalStatus
        +parent_goal_id: str?
        +child_goal_ids: list~str~
        +deadline_turn: int?
    }

    class HierarchyLevel {
        <<enum>>
        STRATEGIC
        TACTICAL
        OPERATIONAL
    }

    class GoalStatus {
        <<enum>>
        ACTIVE
        COMPLETED
        FAILED
        ABANDONED
        SUSPENDED
    }

    class PropagationResult {
        +success: bool
        +goals_created: list~Goal~
        +goals_updated: list~Goal~
        +messages: list~str~
        +merge(other) PropagationResult
    }

    class HierarchicalController {
        +config: HierarchicalConfig
        +network: HierarchicalNetwork
        +llm_advisor: Any?
        +decide(game_state, valid_actions) HierarchicalDecision
        +update(state, action, reward, next, done)
    }

    GoalPropagator --> DecompositionStrategy
    StrategicDecomposition --|> DecompositionStrategy
    TacticalDecomposition --|> DecompositionStrategy
    GoalPropagator --> PropagationResult
    GoalPropagator --> Goal
    Goal --> HierarchyLevel
    Goal --> GoalStatus
    HierarchicalController --> GoalPropagator : "uses"
```

## Innovation Comparison

```mermaid
flowchart LR
    subgraph OPTIONS["Options Framework\n(Sutton 1999)"]
        direction TB
        O1["2 Levels\n(option + primitive)"]
        O1 --> O2["Implicit sub-goals"]
        O2 --> O3["No MCTS\nintegration"]
    end

    subgraph FEUDAL["Feudal Networks\n(Vezhnevets 2017)"]
        direction TB
        F1["2 Levels\n(manager + worker)"]
        F1 --> F2["Learned\nembedding goals"]
        F2 --> F3["No MCTS\nintegration"]
    end

    subgraph NOVEL["THIS SYSTEM"]
        direction TB
        N1["3 Levels\n(strat / tact / ops)"]
        N1 --> N2["58+ Explicit\nTyped Goals"]
        N2 --> N3["Full MCTS\nIntegration"]
        N3 --> N4["Bidirectional +\nLateral Propagation"]
    end

    OPTIONS -.->|"ADVANCES"| NOVEL
    FEUDAL -.->|"ADVANCES"| NOVEL

    style OPTIONS fill:#3d1a1a,stroke:#ff4a4a,stroke-width:2px,color:#fff
    style FEUDAL fill:#3d3d1a,stroke:#ffff4a,stroke-width:2px,color:#fff
    style NOVEL fill:#1a3d1a,stroke:#4aff4a,stroke-width:2px,color:#fff
```

## Data Flow Summary

| Step | Component | Input | Output | Novel? |
|------|-----------|-------|--------|--------|
| 1 | HierarchicalController | Game state | Strategic goal selection | Moderate |
| 2 | StrategicDecomposition | Strategic goal | 2-4 tactical sub-goals | **YES** |
| 3 | TacticalDecomposition | Tactical goal | 1-3 operational sub-goals | **YES** |
| 4 | propagate_up | Child progress | Parent progress (weighted) | **YES** |
| 5 | propagate_lateral | Goal update | Synergy/conflict adjustments | **YES** |
| 6 | synchronize_hierarchy | All active goals | Status/deadline management | **YES** |

> **Key Novelty:** The integration of a 3-level goal hierarchy with explicit domain-specific decomposition maps (58+ goal types), bidirectional propagation (down + up + lateral), and MCTS integration. No prior work combines hierarchical RL with this scale of explicit goal management for complex strategy games.
