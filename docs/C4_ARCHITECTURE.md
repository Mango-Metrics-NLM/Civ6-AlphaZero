# C4 Architecture - Civ6 AlphaZero AI Framework

> Architecture documentation using the C4 model for comprehensive system visualization.
> For Mermaid-rendered diagrams, see [architecture.md](architecture.md).

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core AlphaZero (MCTS + Neural Network) | ✅ Complete |
| Phase 2 | Hierarchical Reinforcement Learning | ✅ Complete |
| Phase 3 | Training Improvements (Curriculum, Rewards) | ✅ Complete |
| Phase 4 | Unified Training Orchestrator | ✅ Complete |
| Phase 5a | RAG-LLM Integration | ✅ Complete |
| Phase 5b | GCP/Vertex AI Cloud Infrastructure | ✅ Complete |
| Phase 5c | Cost Optimization & Resource Management | ✅ Complete |
| Phase 6 | ~~Agentic Framework~~ (removed in Phase 11) | ❌ Removed |
| Phase 7 | Production PBT Scaling & Validation | ✅ Complete |
| Phase 8 | Training Safeguards & Automated Intervention | ✅ Complete |
| Phase 9 | MuZero Integration Bridge | ✅ Complete |
| Phase 10 | Tournament & Matchmaking System | ✅ Complete |
| Phase 11 | Code Quality & Tech Debt Reduction | ✅ Complete |
| Phase 12-Prep | Docker & Vertex AI Smoke Test | ✅ Complete |
| Phase 13 | Multi-Iteration Training & Post-Run Evaluation | ✅ Complete |
| Phase 14 | Learning Signal Validation & Training Diagnostics | ✅ Complete |
| Phase 15 | Scaled Training & Value Head Stabilization | ✅ Complete |
| Phase 16 | Batch Inference & Vectorized Self-Play | ✅ Complete |
| Phase 17a | Hex Crystalline Encoder & A/B Comparison | ✅ Complete |
| Phase 17b | Training Readiness — action_mask, W&B groups, gumbel default, eval retry | ✅ Complete |

**Codebase Statistics:** ~62,000 LOC core, 4,880+ tests, Docker v1.2.0 (Artifact Registry), 218+ models in GCS

---

## Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM CONTEXT                                     │
│                                                                              │
│    ┌─────────────┐         ┌─────────────────────────┐       ┌───────────┐  │
│    │   Human     │         │   Civ6-AlphaZero AI     │       │  Civ6     │  │
│    │   Player    │◄───────►│       Framework         │◄─────►│  Game     │  │
│    │             │         │                         │       │  Engine   │  │
│    └─────────────┘         └───────────┬─────────────┘       └───────────┘  │
│                                        │                                     │
│                                        │                                     │
│                            ┌───────────▼───────────┐                        │
│                            │   External Services   │                        │
│                            │  - LLM APIs (OpenAI)  │                        │
│                            │  - W&B (Metrics)      │                        │
│                            │  - Vector DBs         │                        │
│                            └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  ───► Data/Control Flow
  ◄──► Bidirectional Interaction
```

### Context Description

| Actor/System | Description | Interaction |
|--------------|-------------|-------------|
| Human Player | Users who play against or train the AI | Game moves, configuration |
| Civ6-AlphaZero AI | Core AI system using AlphaZero algorithm | Self-play, training, inference |
| Civ6 Game Engine | Civilization 6 or compatible (Unciv/MicroCiv) | Game state, actions |
| LLM APIs | Language models for strategy advice | Strategy queries, insights |
| W&B | Weights & Biases for experiment tracking | Metrics, checkpoints |
| Vector DBs | ChromaDB/FAISS for RAG knowledge | Strategy knowledge retrieval |

---

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CIV6-ALPHAZERO FRAMEWORK                            │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                           TRAINING LAYER                                    │ │
│  │                                                                             │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │ Orchestrator │───►│   Learner    │───►│   Actors     │                 │ │
│  │   │  (Unified)   │    │  (Training)  │    │  (Self-Play) │                 │ │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │ │
│  │          │                   │                   │                          │ │
│  │          ▼                   ▼                   ▼                          │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │  Scheduler   │    │   Shared     │    │   Replay     │                 │ │
│  │   │ (Hyperparams)│    │   Storage    │    │   Buffer     │                 │ │
│  │   └──────────────┘    └──────────────┘    └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                             CORE AI LAYER                                   │ │
│  │                                                                             │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │   Neural     │    │    MCTS      │    │   Action     │                 │ │
│  │   │   Network    │───►│   Search     │───►│   Space      │                 │ │
│  │   │  (Policy/V)  │    │  (Planning)  │    │  (Hierarchy) │                 │ │
│  │   └──────────────┘    └──────────────┘    └──────────────┘                 │ │
│  │          │                                       │                          │ │
│  │          └───────────────────┬───────────────────┘                          │ │
│  │                              ▼                                              │ │
│  │                    ┌──────────────────┐                                     │ │
│  │                    │  Action Pruning  │                                     │ │
│  │                    │  & Macro Actions │                                     │ │
│  │                    └──────────────────┘                                     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       HIERARCHICAL RL LAYER (Phase 2)                       │ │
│  │                                                                             │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │  Strategic   │    │   Tactical   │    │ Operational  │                 │ │
│  │   │   Level      │───►│    Level     │───►│    Level     │                 │ │
│  │   │ (30-100 trn) │    │  (5-15 trn)  │    │  (1-3 trn)   │                 │ │
│  │   └──────────────┘    └──────────────┘    └──────────────┘                 │ │
│  │          │                   │                   │                          │ │
│  │          ▼                   ▼                   ▼                          │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │    Goals     │    │   Encoders   │    │ Controllers  │                 │ │
│  │   │  Management  │    │ (Per-Level)  │    │ (Per-Level)  │                 │ │
│  │   └──────────────┘    └──────────────┘    └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          GAME MECHANICS LAYER                               │ │
│  │                                                                             │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │ │
│  │  │Diplomacy│ │Religion │ │  Trade  │ │Espionage│ │Congress │ │Government│  │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │ │
│  │       │           │           │           │           │           │        │ │
│  │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐  │ │
│  │  │ Unit    │ │  Great  │ │Era/Ages │ │ Climate │ │ Victory │ │ Combat  │  │ │
│  │  │Promotns │ │ People  │ │ System  │ │Disasters│ │ Tracker │ │ System  │  │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          ENVIRONMENT LAYER                                  │ │
│  │                                                                             │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │   MicroCiv   │    │    Unciv     │    │    Civ6      │                 │ │
│  │   │  (Training)  │◄──►│  (Bridge)    │◄──►│   (Full)     │                 │ │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │ │
│  │          │                   │                   │                          │ │
│  │          └───────────────────┼───────────────────┘                          │ │
│  │                              ▼                                              │ │
│  │                    ┌──────────────────┐                                     │ │
│  │                    │  Curriculum      │                                     │ │
│  │                    │  Learning Mgr    │                                     │ │
│  │                    └──────────────────┘                                     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          INTEGRATION LAYER                                  │ │
│  │                                                                             │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │ │
│  │   │     RAG      │    │   Strategy   │    │    Game      │                 │ │
│  │   │  Pipeline    │───►│   Advisor    │───►│  Interface   │                 │ │
│  │   │ (Knowledge)  │    │   (LLM)      │    │   (IPC)      │                 │ │
│  │   └──────────────┘    └──────────────┘    └──────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram - Core AI Layer

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            CORE AI LAYER COMPONENTS                           │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        NEURAL NETWORK MODULE                            │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │ MicroCivNetwork │  │ Civ6NetworkV2   │  │TransformerNet   │         │  │
│  │  │  (13-channel)   │  │ (84-ch + attn)  │  │ (ViT-based)     │         │  │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │  │
│  │           │                    │                    │                   │  │
│  │           └────────────────────┼────────────────────┘                   │  │
│  │                                ▼                                        │  │
│  │                    ┌───────────────────────┐                            │  │
│  │                    │    NetworkFactory     │                            │  │
│  │                    │ - create_from_config()│                            │  │
│  │                    │ - load_checkpoint()   │                            │  │
│  │                    └───────────────────────┘                            │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │             SPATIAL ENCODERS (Phase 17a — Hex Crystalline)        │   │  │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │   │  │
│  │  │  │  PatchEmbed   │ │HexSpatialEnc  │ │EncoderRegistry│          │   │  │
│  │  │  │ (Grid-based)  │ │(Hex-topology) │ │ (Factory)     │          │   │  │
│  │  │  │  Baseline     │ │ Crystalline   │ │ patch_embed / │          │   │  │
│  │  │  │  encoder      │ │ attention     │ │ hex_crystalln │          │   │  │
│  │  │  └───────────────┘ └───────────────┘ └───────────────┘          │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           MCTS MODULE                                   │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │   MCTSConfig    │  │    MCTSNode     │  │   Civ6MCTS      │         │  │
│  │  │ - simulations   │─►│ - prior, visits │─►│ - search()      │         │  │
│  │  │ - c_puct        │  │ - value_sum     │  │ - select_action │         │  │
│  │  │ - dirichlet_α   │  │ - select_child()│  │ - backup()      │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    MCTS OPTIMIZATIONS                            │   │  │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │   │  │
│  │  │  │ Virtual Loss  │ │ Tree Reuse    │ │ Batch Eval    │          │   │  │
│  │  │  │ (Parallelism) │ │ (Between Moves)│ │ (GPU Batching)│          │   │  │
│  │  │  └───────────────┘ └───────────────┘ └───────────────┘          │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       ACTION SPACE MODULE                               │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    HIERARCHICAL STRUCTURE                        │   │  │
│  │  │                                                                   │   │  │
│  │  │   Level 0: Categories (9)                                        │   │  │
│  │  │   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐      │   │  │
│  │  │   │END_  ││UNIT_ ││CITY_ ││TECH_ ││CIVIC_││DIPLO_││ESPION│      │   │  │
│  │  │   │TURN  ││ACTION││ACTION││RSRCH ││RSRCH ││MACY  ││ AGE  │      │   │  │
│  │  │   └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘      │   │  │
│  │  │      │       │       │       │       │       │       │          │   │  │
│  │  │   Level 1: Selectors (Units/Cities/Items)                        │   │  │
│  │  │      │       │       │       │       │       │       │          │   │  │
│  │  │   Level 2: Action Types (Move/Attack/Build/etc.)                 │   │  │
│  │  │      │       │       │       │       │       │       │          │   │  │
│  │  │   Level 3: Targets (Positions/Items)                             │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    ACTION PRUNING ENGINE                         │   │  │
│  │  │                                                                   │   │  │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐          │   │  │
│  │  │  │   Feasibility │ │   Strategic   │ │   Resource    │          │   │  │
│  │  │  │    Pruner     │ │    Pruner     │ │    Pruner     │          │   │  │
│  │  │  │ - Unit moves  │ │ - Threat eval │ │ - Affordability│         │   │  │
│  │  │  │ - Valid tiles │ │ - Opportunity │ │ - Prerequisites│         │   │  │
│  │  │  └───────────────┘ └───────────────┘ └───────────────┘          │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    MACRO ACTION SYSTEM                           │   │  │
│  │  │                                                                   │   │  │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │   │  │
│  │  │  │ MacroAction: Multi-turn action abstraction                  │ │   │  │
│  │  │  │                                                              │ │   │  │
│  │  │  │  Examples:                                                   │ │   │  │
│  │  │  │  - SETTLE_CITY: Move settler + found city                   │ │   │  │
│  │  │  │  - CONQUER_CITY: Move army + siege + capture                │ │   │  │
│  │  │  │  - BUILD_WONDER: Tech prereqs + production queue            │ │   │  │
│  │  │  │  - EXPLORE_REGION: Scout movement pattern                   │ │   │  │
│  │  │  └─────────────────────────────────────────────────────────────┘ │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram - Game Mechanics Layer

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        GAME MECHANICS LAYER COMPONENTS                        │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      DIPLOMACY SYSTEM                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │DiplomacyManager │  │DiplomaticRelation│ │ DiplomaticAction│         │  │
│  │  │- relationships  │  │- score (-100,100)│ │ - DECLARE_WAR   │         │  │
│  │  │- process_turn() │  │- modifiers      │ │ - MAKE_PEACE    │         │  │
│  │  │- calculate_agnd │  │- status (war/etc)│ │ - PROPOSE_DEAL  │         │  │
│  │  └─────────────────┘  └─────────────────┘  │ - FORM_ALLIANCE │         │  │
│  │                                            └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        TRADE SYSTEM                                     │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │  TradeManager   │  │   TradeRoute    │  │   TradeGoods    │         │  │
│  │  │- active_routes  │  │- origin, dest   │  │ - GOLD, FOOD    │         │  │
│  │  │- calculate_ylds │  │- yields         │  │ - PRODUCTION    │         │  │
│  │  │- manage_traders │  │- distance_bonus │  │ - SCIENCE       │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      ESPIONAGE SYSTEM                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │EspionageManager │  │      Spy        │  │ EspionageMission│         │  │
│  │  │- spies          │  │- level (1-3)    │  │ - GATHER_INFO   │         │  │
│  │  │- assign_mission │  │- location       │  │ - STEAL_TECH    │         │  │
│  │  │- process_turn() │  │- status         │  │ - SABOTAGE      │         │  │
│  │  └─────────────────┘  └─────────────────┘  │ - COUP          │         │  │
│  │                                            └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     GOVERNMENT SYSTEM                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │GovernmentManager│  │   Government    │  │   PolicyCard    │         │  │
│  │  │- current_govt   │  │- type (Autocracy│  │ - MILITARY slot │         │  │
│  │  │- policy_slots   │  │   Democracy,..) │  │ - ECONOMIC slot │         │  │
│  │  │- change_policy()│  │- slot_counts    │  │ - DIPLOMATIC    │         │  │
│  │  └─────────────────┘  │- legacy_bonus   │  │ - WILDCARD      │         │  │
│  │                       └─────────────────┘  └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                   UNIT PROMOTION SYSTEM                                 │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │PromotionManager │  │   Promotion     │  │ PromotionTree   │         │  │
│  │  │- unit_xp        │  │- name, effects  │  │ - prerequisites │         │  │
│  │  │- check_eligible │  │- stat_modifiers │  │ - branches      │         │  │
│  │  │- apply_promo()  │  │- abilities      │  │ - exclusions    │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    GREAT PEOPLE SYSTEM                                  │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │GreatPeopleManager│ │   GreatPerson   │  │  GreatPersonType│         │  │
│  │  │- points_per_type│  │- type, era      │  │ - GREAT_GENERAL │         │  │
│  │  │- recruit()      │  │- unique_ability │  │ - GREAT_SCIENTIST│        │  │
│  │  │- activate()     │  │- passive_bonus  │  │ - GREAT_ENGINEER│         │  │
│  │  └─────────────────┘  └─────────────────┘  │ - GREAT_MERCHANT│         │  │
│  │                                            │ - GREAT_PROPHET │         │  │
│  │                                            └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                   ERA & AGES SYSTEM                                     │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │  EraManager     │  │  EraScore       │  │     Age         │         │  │
│  │  │- current_era    │  │- total_score    │  │ - DARK_AGE      │         │  │
│  │  │- era_score      │  │- dedications    │  │ - NORMAL_AGE    │         │  │
│  │  │- historic_moments│ │- threshold      │  │ - GOLDEN_AGE    │         │  │
│  │  │- check_age()    │  └─────────────────┘  │ - HEROIC_AGE    │         │  │
│  │  └─────────────────┘                       └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                  CLIMATE & DISASTERS SYSTEM                             │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │ ClimateManager  │  │ NaturalDisaster │  │  ClimatePhase   │         │  │
│  │  │- co2_level      │  │ - FLOOD         │  │ - NORMAL        │         │  │
│  │  │- sea_level      │  │ - VOLCANO       │  │ - WARMING_1     │         │  │
│  │  │- process_turn() │  │ - HURRICANE     │  │ - WARMING_2     │         │  │
│  │  │- trigger_event()│  │ - DROUGHT       │  │ - CATASTROPHIC  │         │  │
│  │  └─────────────────┘  │ - BLIZZARD      │  └─────────────────┘         │  │
│  │                       └─────────────────┘                              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                   WORLD CONGRESS SYSTEM                                 │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │CongressManager  │  │   Resolution    │  │  CongressPhase  │         │  │
│  │  │- active_resols  │  │- type, target   │  │ - PROPOSAL      │         │  │
│  │  │- diplomatic_fav │  │- votes_for/agnst│  │ - DISCUSSION    │         │  │
│  │  │- emergency_sess │  │- effects        │  │ - VOTING        │         │  │
│  │  │- vote()         │  └─────────────────┘  │ - RESOLUTION    │         │  │
│  │  └─────────────────┘                       └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram - Curriculum Learning Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      CURRICULUM LEARNING PIPELINE                             │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     CURRICULUM STAGES                                   │  │
│  │                                                                         │  │
│  │   Stage 1: MICROCIV      Stage 2: UNCIV        Stage 3: CIV6           │  │
│  │   ┌─────────────────┐    ┌─────────────────┐   ┌─────────────────┐     │  │
│  │   │ 16x16 board     │    │ JSON interface  │   │ Full game       │     │  │
│  │   │ 3 unit types    │───►│ Core mechanics  │──►│ All mechanics   │     │  │
│  │   │ Basic combat    │    │ Tech trees      │   │ 50k actions     │     │  │
│  │   │ 381 actions     │    │ ~5000 actions   │   │ Lua mod/API     │     │  │
│  │   └─────────────────┘    └─────────────────┘   └─────────────────┘     │  │
│  │                                                                         │  │
│  │   Graduation Criteria:                                                  │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │   │ - Win rate vs baseline > threshold (configurable)               │  │  │
│  │   │ - Policy entropy stabilized                                     │  │  │
│  │   │ - Value prediction accuracy > threshold                         │  │  │
│  │   │ - Minimum training iterations completed                         │  │  │
│  │   └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                   CURRICULUM MANAGER                                    │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    CurriculumConfig                              │   │  │
│  │  │  @dataclass                                                      │   │  │
│  │  │  stages: List[StageConfig]                                       │   │  │
│  │  │  graduation_criteria: GraduationCriteria                         │   │  │
│  │  │  transfer_learning: TransferConfig                               │   │  │
│  │  │  difficulty_schedule: DifficultySchedule                         │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    CurriculumManager                             │   │  │
│  │  │  - current_stage: Stage                                          │   │  │
│  │  │  - stage_metrics: Dict[Stage, MetricsTracker]                    │   │  │
│  │  │  - check_graduation() -> bool                                    │   │  │
│  │  │  - advance_stage() -> None                                       │   │  │
│  │  │  - get_environment() -> Environment                              │   │  │
│  │  │  - get_network_config() -> NetworkConfig                         │   │  │
│  │  │  - transfer_weights(old_net, new_net) -> None                    │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                   TRANSFER LEARNING                                     │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │   Encoder       │  │  Policy Head    │  │   Value Head    │         │  │
│  │  │   Transfer      │  │   Expansion     │  │   Fine-tuning   │         │  │
│  │  │ - Feature maps  │  │ - Action space  │  │ - Domain adapt  │         │  │
│  │  │ - Interpolation │  │   projection    │  │ - LR schedule   │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Code Diagram - Action Pruning System

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ACTION PRUNING SYSTEM                                 │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  class ActionPruner:                                                    │  │
│  │      """Intelligent action space pruning for computational efficiency"""│  │
│  │                                                                         │  │
│  │      def __init__(self, config: PrunerConfig):                          │  │
│  │          self.feasibility_pruner = FeasibilityPruner(config)           │  │
│  │          self.strategic_pruner = StrategicPruner(config)               │  │
│  │          self.resource_pruner = ResourcePruner(config)                 │  │
│  │          self.cache = LRUCache(config.cache_size)                      │  │
│  │                                                                         │  │
│  │      def prune(self, state: GameState, actions: ActionMask)            │  │
│  │              -> PrunedActions:                                          │  │
│  │          # Stage 1: Feasibility (hard constraints)                      │  │
│  │          feasible = self.feasibility_pruner.filter(state, actions)     │  │
│  │                                                                         │  │
│  │          # Stage 2: Resource requirements                               │  │
│  │          affordable = self.resource_pruner.filter(state, feasible)     │  │
│  │                                                                         │  │
│  │          # Stage 3: Strategic relevance (soft pruning)                  │  │
│  │          relevant = self.strategic_pruner.filter(state, affordable)    │  │
│  │                                                                         │  │
│  │          return PrunedActions(                                          │  │
│  │              actions=relevant,                                          │  │
│  │              pruning_stats=self._compute_stats()                        │  │
│  │          )                                                              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  class FeasibilityPruner:                                               │  │
│  │      """Remove physically impossible actions"""                         │  │
│  │                                                                         │  │
│  │      def filter(self, state, actions):                                  │  │
│  │          pruned = actions.copy()                                        │  │
│  │                                                                         │  │
│  │          # Units can only move to valid tiles                           │  │
│  │          for unit in state.units:                                       │  │
│  │              valid_moves = self._get_reachable_tiles(unit, state)      │  │
│  │              pruned.mask_invalid_moves(unit.id, valid_moves)           │  │
│  │                                                                         │  │
│  │          # Cities can only build unlocked items                         │  │
│  │          for city in state.cities:                                      │  │
│  │              buildable = self._get_buildable_items(city, state)        │  │
│  │              pruned.mask_invalid_production(city.id, buildable)        │  │
│  │                                                                         │  │
│  │          return pruned                                                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  class StrategicPruner:                                                 │  │
│  │      """Remove strategically irrelevant actions"""                      │  │
│  │                                                                         │  │
│  │      def filter(self, state, actions):                                  │  │
│  │          # Don't consider building Wonders we can't realistically get   │  │
│  │          if self._wonder_unlikely(state, wonder):                       │  │
│  │              actions.mask_action(wonder_action)                         │  │
│  │                                                                         │  │
│  │          # Prioritize threatened units/cities                           │  │
│  │          threats = self._evaluate_threats(state)                        │  │
│  │          if threats.level > THRESHOLD:                                  │  │
│  │              actions.boost_defensive_actions()                          │  │
│  │                                                                         │  │
│  │          # Prune distant exploration when at war                        │  │
│  │          if state.is_at_war:                                            │  │
│  │              actions.deprioritize_exploration()                         │  │
│  │                                                                         │  │
│  │          return actions                                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW                                          │
│                                                                               │
│   ┌─────────────────┐                                                        │
│   │   Game Engine   │                                                        │
│   │  (MicroCiv/     │                                                        │
│   │   Unciv/Civ6)   │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                  │
│            ▼ (game state)                                                    │
│   ┌─────────────────┐                                                        │
│   │  State Encoder  │ ─────────────────────────────────────┐                 │
│   │  - 13/84 ch     │                                      │                 │
│   │  - 512 vector   │                                      │                 │
│   └────────┬────────┘                                      │                 │
│            │                                               │                 │
│            ▼ (tensors)                                     │                 │
│   ┌─────────────────┐                                      │                 │
│   │ Neural Network  │                                      │                 │
│   │  - ResNet/ViT   │                                      │                 │
│   │  - Policy+Value │                                      │                 │
│   └────────┬────────┘                                      │                 │
│            │                                               │                 │
│        ┌───┴───┐                                           │                 │
│        ▼       ▼                                           │                 │
│   ┌────────┐┌────────┐                                     │                 │
│   │Policy  ││ Value  │                                     │                 │
│   │Logits  ││Estimate│                                     │                 │
│   └───┬────┘└───┬────┘                                     │                 │
│       │         │                                          │                 │
│       └────┬────┘                                          │                 │
│            ▼                                               │                 │
│   ┌─────────────────┐                                      │                 │
│   │  Action Pruner  │◄─────────────────────────────────────┘                 │
│   │  - Feasibility  │ (state for pruning decisions)                          │
│   │  - Strategic    │                                                        │
│   └────────┬────────┘                                                        │
│            │                                                                  │
│            ▼ (pruned policy)                                                 │
│   ┌─────────────────┐                                                        │
│   │      MCTS       │◄──┐                                                    │
│   │  - Selection    │   │                                                    │
│   │  - Expansion    │   │ (policy/value priors)                             │
│   │  - Backup       │───┘                                                    │
│   └────────┬────────┘                                                        │
│            │                                                                  │
│            ▼ (selected action)                                               │
│   ┌─────────────────┐         ┌─────────────────┐                           │
│   │ Action Executor │────────►│  Replay Buffer  │                           │
│   │  - Apply action │         │  - State, π, v  │                           │
│   │  - Get next obs │         │  - Sampling     │                           │
│   └─────────────────┘         └────────┬────────┘                           │
│                                        │                                     │
│                                        ▼ (training batch)                    │
│                               ┌─────────────────┐                           │
│                               │    Learner      │                           │
│                               │  - Loss calc    │                           │
│                               │  - Backprop     │                           │
│                               │  - Update       │                           │
│                               └─────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT ARCHITECTURE                              │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      TRAINING CLUSTER                                   │  │
│  │                                                                         │  │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │  │
│  │   │  Coordinator    │    │    Learner      │    │  Actor Pool     │    │  │
│  │   │  (1x CPU)       │    │ (1x GPU/TPU)    │    │ (Nx CPU)        │    │  │
│  │   │  - Orchestrate  │◄──►│  - Training     │◄──►│  - Self-play    │    │  │
│  │   │  - Scheduling   │    │  - Gradients    │    │  - Game sims    │    │  │
│  │   │  - Checkpoints  │    │                 │    │                 │    │  │
│  │   └─────────────────┘    └─────────────────┘    └─────────────────┘    │  │
│  │            │                      │                      │              │  │
│  │            └──────────────────────┼──────────────────────┘              │  │
│  │                                   ▼                                     │  │
│  │                    ┌─────────────────────────┐                          │  │
│  │                    │     Shared Storage      │                          │  │
│  │                    │  - Redis (weights)      │                          │  │
│  │                    │  - Replay Buffer        │                          │  │
│  │                    └─────────────────────────┘                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      INFERENCE SERVER                                   │  │
│  │                                                                         │  │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │  │
│  │   │  API Gateway    │    │  Model Server   │    │   Game Bridge   │    │  │
│  │   │  (FastAPI)      │───►│  (TorchServe)   │───►│  (Lua/JSON)     │    │  │
│  │   │  - REST/WS      │    │  - Batching     │    │  - State sync   │    │  │
│  │   │  - Auth         │    │  - GPU accel    │    │  - Actions      │    │  │
│  │   └─────────────────┘    └─────────────────┘    └─────────────────┘    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     MONITORING & LOGGING                                │  │
│  │                                                                         │  │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │  │
│  │   │   Prometheus    │    │    Grafana      │    │   W&B Cloud     │    │  │
│  │   │   (Metrics)     │───►│   (Dashboard)   │    │   (Experiments) │    │  │
│  │   └─────────────────┘    └─────────────────┘    └─────────────────┘    │  │
│  │                                                                         │  │
│  │   ┌─────────────────┐    ┌─────────────────┐                           │  │
│  │   │   ELK Stack     │    │   TensorBoard   │                           │  │
│  │   │   (Logs)        │    │   (Training)    │                           │  │
│  │   └─────────────────┘    └─────────────────┘                           │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| Core AI | Neural Network | PyTorch 2.0+ | Model training/inference |
| Core AI | MCTS | NumPy | Tree search algorithms |
| Core AI | Transformer | PyTorch | ViT with RoPE, MoE |
| Environment | MicroCiv | Gymnasium | Training simulation |
| Environment | Civ6 Bridge | Lua/Python IPC | Real game integration |
| Training | Orchestrator | Python | Training coordination |
| Training | Distributed | multiprocessing | Parallel self-play |
| Training | PBT | Python | Hyperparameter evolution |
| Cloud | Compute | Vertex AI | Distributed training |
| Cloud | Storage | GCS | Artifacts, checkpoints |
| Cloud | Container | Docker + Artifact Registry | Containerized workers (v1.2.0) |
| Cloud | Optimization | Python | Cost/quota management |
| Knowledge | RAG | ChromaDB/FAISS | Vector search |
| Knowledge | LLM | OpenAI/Anthropic | Strategy advisor |
| Logging | Metrics | W&B, TensorBoard | Experiment tracking |
| Testing | Framework | pytest | 4,880+ tests, >80% coverage |
| Linting | Quality | ruff, mypy, bandit | Static analysis, security |

---

## Key Architectural Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Hierarchical action space | Reduces branching factor from 10^6 to 10^3 | Complexity in encoding/decoding |
| Curriculum learning | Faster convergence, better generalization | Requires environment abstraction |
| Process-based distribution | Avoid GIL, maximize CPU for MCTS | IPC overhead |
| ResNet + Attention | Proven spatial reasoning + global context | Computational cost |
| GCP Spot VMs | 60-91% cost reduction for self-play | Preemption handling complexity |
| Protocol-driven cloud | Backend-agnostic design for testability | Abstraction layers |
| LZMA trajectory compression | Minimizes GCS egress costs | CPU overhead |
| NetworkArchitectureConfig | Single source of truth for model dimensions | Must stay in sync with TransformerConfig |
| Post-run tournament local | Avoids extra Vertex AI job cost for evaluation | Limited by local CPU speed |
| model_config persistence | Enables legacy model loading with correct architecture | Enum→string conversion needed |
| RAG integration | Knowledge-enhanced decisions | Latency for queries |

---

## Level 3: Component Diagram - Hierarchical RL Layer

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      HIERARCHICAL RL LAYER COMPONENTS                         │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       DECISION HIERARCHY                                │  │
│  │                                                                         │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    STRATEGIC LEVEL (30-100 turns)               │  │  │
│  │   │   Goals: Victory condition, Long-term expansion, Tech path      │  │  │
│  │   │   Encoder: Global state features, Player relations              │  │  │
│  │   │   Output: High-level objectives, Resource allocation            │  │  │
│  │   └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    TACTICAL LEVEL (5-15 turns)                  │  │  │
│  │   │   Goals: Regional control, Army composition, City development   │  │  │
│  │   │   Encoder: Regional features, Unit positions, Threats           │  │  │
│  │   │   Output: Unit group commands, Production priorities            │  │  │
│  │   └─────────────────────────────────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │   │                   OPERATIONAL LEVEL (1-3 turns)                 │  │  │
│  │   │   Goals: Individual unit actions, Immediate threats             │  │  │
│  │   │   Encoder: Local tile features, Combat modifiers                │  │  │
│  │   │   Output: Specific move/attack/build actions                    │  │  │
│  │   └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       HIERARCHICAL CONTROLLER                           │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │ GoalManager     │  │ LevelScheduler  │  │ RewardPropagator│         │  │
│  │  │ - goal_stack    │  │ - active_level  │  │ - intrinsic_rwd │         │  │
│  │  │ - propagate()   │  │ - switch_level()│  │ - extrinsic_rwd │         │  │
│  │  │ - evaluate()    │  │ - time_horizon  │  │ - backprop()    │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    LLM GUIDANCE INTEGRATION                      │   │  │
│  │  │  - Strategy advice from RAG knowledge base                       │   │  │
│  │  │  - Goal suggestions based on game state                          │   │  │
│  │  │  - Explanation generation for decisions                          │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 2: Agentic Framework (Phase 6) — REMOVED

> The agentic framework (~14,600 LOC) was removed in Phase 11 as dead code.
> See CHANGELOG.md for details.

---

## Level 3: Component Diagram — Training Safeguards (Phase 8)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING SAFEGUARDS (Phase 8)                         │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       HEALTH MONITORING                                │  │
│  │                                                                         │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │  │
│  │   │  NaN         │    │  Loss        │    │  Gradient    │             │  │
│  │   │  Detection   │    │  Explosion   │    │  Health      │             │  │
│  │   │ - consecutive│    │ - rolling    │    │ - NORMAL     │             │  │
│  │   │   threshold  │    │   baseline   │    │ - HIGH       │             │  │
│  │   │              │    │   comparison │    │ - EXPLODING  │             │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      AUTOMATED INTERVENTIONS                           │  │
│  │                                                                         │  │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │   │  Reduce LR  │ │  Rollback   │ │  Reduce     │ │  Early      │     │  │
│  │   │  (loss      │ │  Checkpoint │ │  Batch Size │ │  Stop       │     │  │
│  │   │   explode)  │ │  (NaN)      │ │  (memory)   │ │  (stalled)  │     │  │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          REPORTING                                     │  │
│  │                                                                         │  │
│  │   ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │   │ TrainingReport: periodic summaries, SafeguardReport: per-step    │ │  │
│  │   └──────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 3: Component Diagram — Tournament System (Phase 10)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TOURNAMENT SYSTEM (Phase 10)                          │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       ORCHESTRATION                                    │  │
│  │                                                                         │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │  │
│  │   │  Tournament  │    │   Bracket    │    │  Standings   │             │  │
│  │   │ Orchestrator │───►│ Round-Robin  │───►│  ELO + Win%  │             │  │
│  │   │              │    │ Elimination  │    │  Rankings    │             │  │
│  │   │              │    │ Swiss        │    │              │             │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        ANALYSIS                                        │  │
│  │                                                                         │  │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │   │ Head-to-Head│ │  Player     │ │   Meta      │ │   Upset     │     │  │
│  │   │   Matrix    │ │  Trends     │ │  Analysis   │ │  Detection  │     │  │
│  │   │  N×N wins   │ │  ELO/time   │ │  top-k HPs  │ │  ELO diff   │     │  │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                      W&B VISUALIZATION                                 │  │
│  │                                                                         │  │
│  │   ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │   │ TournamentWandbLogger: Leaderboard, Heatmap, ELO History chart  │ │  │
│  │   └──────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Cloud Training Pipeline (Phases 12-13)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    VERTEX AI TRAINING PIPELINE (Phase 13)                     │
│                                                                               │
│  User / CLI                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────┐                                                        │
│  │ Vertex           │─── W&B init ──► W&B Cloud                             │
│  │ Orchestrator     │                                                        │
│  │ (local machine)  │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           │  For each iteration:                                             │
│           │                                                                  │
│           ├──► Self-Play Workers (N) ──► GCS (trajectories)                 │
│           │        (Vertex AI T4)                                            │
│           │                                                                  │
│           ├──► Training Worker ──► GCS (model v_new + model_config)          │
│           │        (Vertex AI T4)                                            │
│           │                                                                  │
│           ├──► Evaluation Worker ──► GCS (results JSON)                     │
│           │        (Vertex AI T4)                                            │
│           │                                                                  │
│           ├──► Accept/Reject model (win_rate > threshold?)                   │
│           └──► Log iteration metrics + cost ──► W&B                         │
│                                                                               │
│  Post-Run Tournament (optional, runs LOCALLY on orchestrator):               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Load all model versions from GCS                                   │     │
│  │  Round-robin arena games (CPU) ──► ELO ratings                     │     │
│  │  Save results ──► GCS (tournament/post_run_results.json)           │     │
│  │  Log ELO rankings ──► W&B                                          │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
│  Key Components:                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐               │
│  │ NetworkArch  │  │      Arena       │  │  GCS Storage     │               │
│  │   Config     │  │ (shared eval +   │  │ - ModelStorage   │               │
│  │ (defaults)   │  │  tournament)     │  │ - TrajectoryStore│               │
│  └──────────────┘  └──────────────────┘  │ - CheckpointStore│               │
│                                          └──────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 4: Neural Network Architecture Diagrams

### Network Family Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     NEURAL NETWORK ARCHITECTURE FAMILY                        │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       NETWORK HIERARCHY                                  │ │
│  │                                                                          │ │
│  │   ┌──────────────────┐                                                   │ │
│  │   │  MicroCivNetwork │ ◄─── LEGACY (Phase 12 and earlier)               │ │
│  │   │   (13 channels)  │      Simple, fast iteration                      │ │
│  │   │   376 actions    │                                                   │ │
│  │   └────────┬─────────┘                                                   │ │
│  │            │ Transfer Learning                                           │ │
│  │            ▼                                                             │ │
│  │   ┌──────────────────┐                                                   │ │
│  │   │  Civ6NetworkV2   │ ◄─── Production Network                          │ │
│  │   │   (84 channels)  │      Full feature set + attention                │ │
│  │   │   50k+ actions   │                                                   │ │
│  │   └────────┬─────────┘                                                   │ │
│  │            │ Optional Upgrade                                            │ │
│  │            ▼                                                             │ │
│  │   ┌──────────────────┐                                                   │ │
│  │   │TransformerNetwork│ ◄─── Advanced Architecture                       │ │
│  │   │  (ViT + RoPE)    │      Global attention, MoE optional              │ │
│  │   │   Configurable   │                                                   │ │
│  │   └──────────────────┘                                                   │ │
│  │                                                                          │ │
│  │   ┌──────────────────┐                                                   │ │
│  │   │  MuZeroNetwork   │ ◄─── Learned World Model                         │ │
│  │   │ (dynamics model) │      Model-based RL extension                    │ │
│  │   │   Latent space   │                                                   │ │
│  │   └──────────────────┘                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### MicroCivNetwork Architecture (Training)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MICROCIV NETWORK ARCHITECTURE                          │
│                      (Lightweight Training Network)                           │
│                                                                               │
│  INPUT                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    DUAL-INPUT ENCODING                                   │ │
│  │                                                                          │ │
│  │   Spatial Input                    Vector Input                          │ │
│  │   ┌──────────────────┐            ┌──────────────────┐                  │ │
│  │   │ 13 Channels      │            │ 64 Dimensions    │                  │ │
│  │   │ 16×16 Grid       │            │ Global Features  │                  │ │
│  │   │                  │            │                  │                  │ │
│  │   │ Ch 0: Terrain    │            │ - Turn number    │                  │ │
│  │   │ Ch 1: Units      │            │ - Resources      │                  │ │
│  │   │ Ch 2: Cities     │            │ - Score          │                  │ │
│  │   │ Ch 3: Ownership  │            │ - Population     │                  │ │
│  │   │ Ch 4: Visibility │            │ - Tech progress  │                  │ │
│  │   │ Ch 5-7: Unit HP  │            │ - City count     │                  │ │
│  │   │ Ch 8-12: Types   │            │ - Unit count     │                  │ │
│  │   └────────┬─────────┘            └────────┬─────────┘                  │ │
│  └────────────┼────────────────────────────────┼────────────────────────────┘ │
│               │                                │                              │
│               ▼                                ▼                              │
│  ENCODER                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │   ┌──────────────────┐            ┌──────────────────┐                  │ │
│  │   │ Initial Conv     │            │ Vector MLP       │                  │ │
│  │   │ 13 → 64 channels │            │ 64 → 128 → 64    │                  │ │
│  │   │ 3×3, stride 1    │            │ ReLU activation  │                  │ │
│  │   │ BatchNorm + ReLU │            │                  │                  │ │
│  │   └────────┬─────────┘            └────────┬─────────┘                  │ │
│  │            │                                │                            │ │
│  │            ▼                                │                            │ │
│  │   ┌──────────────────┐                      │                            │ │
│  │   │ ResNet Tower     │                      │                            │ │
│  │   │ 6 Residual Blocks│                      │                            │ │
│  │   │ 64 channels each │                      │                            │ │
│  │   │                  │                      │                            │ │
│  │   │ Block Structure: │                      │                            │ │
│  │   │ ┌──────────────┐ │                      │                            │ │
│  │   │ │Conv 3×3 → BN │ │                      │                            │ │
│  │   │ │    → ReLU    │ │                      │                            │ │
│  │   │ │Conv 3×3 → BN │ │                      │                            │ │
│  │   │ │  + Skip Conn │ │                      │                            │ │
│  │   │ │    → ReLU    │ │                      │                            │ │
│  │   │ └──────────────┘ │                      │                            │ │
│  │   └────────┬─────────┘                      │                            │ │
│  │            │                                │                            │ │
│  │            └─────────────┬──────────────────┘                            │ │
│  │                          ▼                                               │ │
│  │                 ┌──────────────────┐                                     │ │
│  │                 │ Feature Fusion   │                                     │ │
│  │                 │ Spatial flatten  │                                     │ │
│  │                 │ + Vector concat  │                                     │ │
│  │                 │ 64×16×16 + 64    │                                     │ │
│  │                 │ = 16,448 total   │                                     │ │
│  │                 └────────┬─────────┘                                     │ │
│  └──────────────────────────┼───────────────────────────────────────────────┘ │
│                             │                                                 │
│                             ▼                                                 │
│  OUTPUT HEADS                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   ┌──────────────────────────┐    ┌──────────────────────────┐          │ │
│  │   │      POLICY HEAD         │    │       VALUE HEAD          │          │ │
│  │   │                          │    │                           │          │ │
│  │   │  Conv 1×1 → 2 channels   │    │  Conv 1×1 → 1 channel     │          │ │
│  │   │         ↓                │    │         ↓                 │          │ │
│  │   │  Flatten (2×16×16=512)   │    │  Flatten (1×16×16=256)    │          │ │
│  │   │         ↓                │    │         ↓                 │          │ │
│  │   │  Linear: 512 → 381       │    │  Linear: 256 → 128        │          │ │
│  │   │         ↓                │    │         ↓ (ReLU)          │          │ │
│  │   │  Log-Softmax             │    │  Linear: 128 → 1          │          │ │
│  │   │         ↓                │    │         ↓                 │          │ │
│  │   │  π(a|s) distribution     │    │  Tanh → v(s) ∈ [-1, 1]    │          │ │
│  │   │                          │    │                           │          │ │
│  │   │  Action Space (381):     │    │  Value: Win probability   │          │ │
│  │   │  - 1 end_turn            │    │   -1 = certain loss       │          │ │
│  │   │  - 160 unit_move         │    │    0 = draw               │          │ │
│  │   │  - 160 unit_attack       │    │   +1 = certain win        │          │ │
│  │   │  - 20 unit_skip          │    │                           │          │ │
│  │   │  - 20 found_city         │    │                           │          │ │
│  │   │  - 15 city_produce       │    │                           │          │ │
│  │   │  - 5 tech_research       │    │                           │          │ │
│  │   └──────────────────────────┘    └──────────────────────────┘          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  PARAMETERS: ~500K total (lightweight for rapid iteration)                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Civ6NetworkV2 Architecture (Production)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     CIV6 NETWORK V2 ARCHITECTURE                              │
│                   (Production Network with Attention)                         │
│                                                                               │
│  INPUT                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    RICH STATE ENCODING                                   │ │
│  │                                                                          │ │
│  │   Spatial Input (84 Channels)         Vector Input (512 Dimensions)     │ │
│  │   ┌──────────────────────────┐       ┌──────────────────────────┐       │ │
│  │   │ Terrain (20 channels)    │       │ Player State (128 dim)   │       │ │
│  │   │ - Base terrain type      │       │ - Gold, faith, culture   │       │ │
│  │   │ - Features (forest,etc)  │       │ - Science per turn       │       │ │
│  │   │ - Resources              │       │ - Era, government        │       │ │
│  │   │ - Improvements           │       │ - Victory progress       │       │ │
│  │   │                          │       │                          │       │ │
│  │   │ Units (24 channels)      │       │ Diplomacy (64 dim)       │       │ │
│  │   │ - Unit type one-hot      │       │ - Relations per civ      │       │ │
│  │   │ - Health/movement        │       │ - War/peace status       │       │ │
│  │   │ - Promotions             │       │ - Trade agreements       │       │ │
│  │   │ - Owner                  │       │                          │       │ │
│  │   │                          │       │ Tech/Civic (128 dim)     │       │ │
│  │   │ Cities (20 channels)     │       │ - Completed techs        │       │ │
│  │   │ - Population             │       │ - Current research       │       │ │
│  │   │ - Districts              │       │ - Unlocked buildings     │       │ │
│  │   │ - Buildings              │       │                          │       │ │
│  │   │ - Production queue       │       │ Military (64 dim)        │       │ │
│  │   │                          │       │ - Army strength          │       │ │
│  │   │ Visibility (10 channels) │       │ - Threat assessment      │       │ │
│  │   │ - Fog of war             │       │ - Strategic resources    │       │ │
│  │   │ - Previously seen        │       │                          │       │ │
│  │   │ - Ownership layers       │       │ Time (128 dim)           │       │ │
│  │   │                          │       │ - Turn number            │       │ │
│  │   │ Combat (10 channels)     │       │ - Era multipliers        │       │ │
│  │   │ - Threat zones           │       │ - Game phase             │       │ │
│  │   │ - Defense modifiers      │       │                          │       │ │
│  │   └──────────────────────────┘       └──────────────────────────┘       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  SPATIAL ENCODER (with Attention)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │   │              ATTENTION-AUGMENTED SPATIAL ENCODER                  │  │ │
│  │   │                                                                   │  │ │
│  │   │  Input: 84 × H × W                                                │  │ │
│  │   │         ↓                                                         │  │ │
│  │   │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │   │  │ Positional Encoding 2D (Learned)                            │ │  │ │
│  │   │  │ - Adds spatial position information                         │ │  │ │
│  │   │  │ - sin/cos encoding + learned embeddings                     │ │  │ │
│  │   │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │   │         ↓                                                         │  │ │
│  │   │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │   │  │ Initial Conv Block: 84 → 128 channels                       │ │  │ │
│  │   │  │ 3×3 conv, BatchNorm, ReLU                                   │ │  │ │
│  │   │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │   │         ↓                                                         │  │ │
│  │   │  ┌─────────────────────────────────────────────────────────────┐ │  │ │
│  │   │  │ ResNet Tower: 10 Residual Blocks (128 channels)             │ │  │ │
│  │   │  │                                                              │ │  │ │
│  │   │  │   Block 1-3: Standard ResBlocks                              │ │  │ │
│  │   │  │        ↓                                                     │ │  │ │
│  │   │  │   Block 4: ResBlock + Spatial Self-Attention                │ │  │ │
│  │   │  │   ┌─────────────────────────────────────────┐               │ │  │ │
│  │   │  │   │ Multi-Head Attention (4 heads)          │               │ │  │ │
│  │   │  │   │ Q = W_q × features                      │               │ │  │ │
│  │   │  │   │ K = W_k × features                      │               │ │  │ │
│  │   │  │   │ V = W_v × features                      │               │ │  │ │
│  │   │  │   │ Attn = softmax(QK^T/√d) × V             │               │ │  │ │
│  │   │  │   └─────────────────────────────────────────┘               │ │  │ │
│  │   │  │        ↓                                                     │ │  │ │
│  │   │  │   Block 5-7: Standard ResBlocks                              │ │  │ │
│  │   │  │        ↓                                                     │ │  │ │
│  │   │  │   Block 8: ResBlock + Spatial Self-Attention                │ │  │ │
│  │   │  │        ↓                                                     │ │  │ │
│  │   │  │   Block 9-10: Standard ResBlocks                             │ │  │ │
│  │   │  └─────────────────────────────────────────────────────────────┘ │  │ │
│  │   │         ↓                                                         │  │ │
│  │   │  Output: 128 × H × W (spatial features with global context)      │  │ │
│  │   └──────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  CROSS-MODAL FUSION                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │   ┌────────────────────────────────────────────────────────────────┐    │ │
│  │   │              CROSS-MODAL ATTENTION                              │    │ │
│  │   │                                                                 │    │ │
│  │   │   Spatial Features (128×H×W)    Vector Features (512)          │    │ │
│  │   │         │                              │                        │    │ │
│  │   │         │    ┌─────────────────────────┘                        │    │ │
│  │   │         │    │                                                  │    │ │
│  │   │         ▼    ▼                                                  │    │ │
│  │   │   ┌────────────────────────┐                                    │    │ │
│  │   │   │ Cross-Attention Block  │                                    │    │ │
│  │   │   │                        │                                    │    │ │
│  │   │   │ Q = Spatial features   │                                    │    │ │
│  │   │   │ K,V = Vector features  │                                    │    │ │
│  │   │   │                        │                                    │    │ │
│  │   │   │ Allows spatial to      │                                    │    │ │
│  │   │   │ attend to global state │                                    │    │ │
│  │   │   └────────────────────────┘                                    │    │ │
│  │   │              │                                                  │    │ │
│  │   │              ▼                                                  │    │ │
│  │   │   ┌────────────────────────┐                                    │    │ │
│  │   │   │ Fused Representation   │                                    │    │ │
│  │   │   │ 128×H×W + 512 = embed  │                                    │    │ │
│  │   │   └────────────────────────┘                                    │    │ │
│  │   └────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  HIERARCHICAL POLICY HEAD                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │   ┌────────────────────────────────────────────────────────────────┐    │ │
│  │   │            HIERARCHICAL ACTION DECOMPOSITION                    │    │ │
│  │   │                                                                 │    │ │
│  │   │   Fused Features                                                │    │ │
│  │   │         │                                                       │    │ │
│  │   │         ▼                                                       │    │ │
│  │   │   ┌──────────────┐                                              │    │ │
│  │   │   │ Category Head│ → P(category) [9 categories]                │    │ │
│  │   │   │ Linear → SM  │   END_TURN, UNIT_ACTION, CITY_ACTION,       │    │ │
│  │   │   └──────┬───────┘   TECH, CIVIC, DIPLOMACY, ESPIONAGE...      │    │ │
│  │   │          │                                                      │    │ │
│  │   │          ▼                                                      │    │ │
│  │   │   ┌──────────────┐                                              │    │ │
│  │   │   │Selector Head │ → P(selector|category) [varies]             │    │ │
│  │   │   │ Per-category │   Which unit? Which city?                   │    │ │
│  │   │   └──────┬───────┘                                              │    │ │
│  │   │          │                                                      │    │ │
│  │   │          ▼                                                      │    │ │
│  │   │   ┌──────────────┐                                              │    │ │
│  │   │   │Action Type Hd│ → P(action_type|selector) [varies]          │    │ │
│  │   │   │ Per-selector │   MOVE, ATTACK, FORTIFY, BUILD...           │    │ │
│  │   │   └──────┬───────┘                                              │    │ │
│  │   │          │                                                      │    │ │
│  │   │          ▼                                                      │    │ │
│  │   │   ┌──────────────┐                                              │    │ │
│  │   │   │ Target Head  │ → P(target|action_type) [varies]            │    │ │
│  │   │   │Spatial/Item  │   Position, building, tech, etc.            │    │ │
│  │   │   └──────────────┘                                              │    │ │
│  │   │                                                                 │    │ │
│  │   │   Final: π(a|s) = P(cat) × P(sel|cat) × P(act|sel) × P(tgt|act)│    │ │
│  │   └────────────────────────────────────────────────────────────────┘    │ │
│  │                                                                          │ │
│  │   ┌────────────────────────────────────────────────────────────────┐    │ │
│  │   │                      VALUE HEAD                                 │    │ │
│  │   │                                                                 │    │ │
│  │   │   Fused Features → Conv 1×1 → Flatten → MLP(256→128→1) → Tanh  │    │ │
│  │   │                                                                 │    │ │
│  │   │   Output: v(s) ∈ [-1, 1] (expected game outcome)               │    │ │
│  │   └────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  PARAMETERS: ~8M total (production-grade)                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### MuZero Network Architecture (World Model)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MUZERO NETWORK ARCHITECTURE                            │
│                    (Learned World Model for Planning)                         │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      THREE-FUNCTION ARCHITECTURE                         │ │
│  │                                                                          │ │
│  │                           Observation (o_t)                              │ │
│  │                                 │                                        │ │
│  │                                 ▼                                        │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │              h = REPRESENTATION(o_t)                             │   │ │
│  │   │                                                                  │   │ │
│  │   │   ┌────────────────────────────────────────────────────────┐    │   │ │
│  │   │   │ Observation Encoder (same as Civ6NetworkV2 encoder)    │    │   │ │
│  │   │   │                                                         │    │   │ │
│  │   │   │ Spatial: 84ch × H × W ──┐                               │    │   │ │
│  │   │   │                         ├──► ResNet + Attention         │    │   │ │
│  │   │   │ Vector: 512 dim ────────┘       │                       │    │   │ │
│  │   │   │                                 ▼                       │    │   │ │
│  │   │   │                    Hidden State h (256 dim latent)      │    │   │ │
│  │   │   └────────────────────────────────────────────────────────┘    │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                 │                                        │ │
│  │                                 ▼                                        │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │         (r, h') = DYNAMICS(h, a)                                 │   │ │
│  │   │                                                                  │   │ │
│  │   │   ┌────────────────────────────────────────────────────────┐    │   │ │
│  │   │   │ Dynamics Function (Learned State Transition)           │    │   │ │
│  │   │   │                                                         │    │   │ │
│  │   │   │  Hidden State h ────┐                                   │    │   │ │
│  │   │   │  (256 dim)          │                                   │    │   │ │
│  │   │   │                     ├──► Concat ──► MLP (512→256→256)  │    │   │ │
│  │   │   │  Action a ──────────┘              │                    │    │   │ │
│  │   │   │  (one-hot encoded)                 ▼                    │    │   │ │
│  │   │   │                          ┌─────────┴─────────┐          │    │   │ │
│  │   │   │                          │                   │          │    │   │ │
│  │   │   │                          ▼                   ▼          │    │   │ │
│  │   │   │                    Next Hidden h'      Reward r         │    │   │ │
│  │   │   │                     (256 dim)       (scalar or dist)    │    │   │ │
│  │   │   │                                                         │    │   │ │
│  │   │   │   Key: Predicts NEXT latent state without actual env    │    │   │ │
│  │   │   │        Enables planning in imagination (MCTS in latent) │    │   │ │
│  │   │   └────────────────────────────────────────────────────────┘    │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                 │                                        │ │
│  │                                 ▼                                        │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │         (π, v) = PREDICTION(h)                                   │   │ │
│  │   │                                                                  │   │ │
│  │   │   ┌────────────────────────────────────────────────────────┐    │   │ │
│  │   │   │ Prediction Function (Policy & Value from Latent)       │    │   │ │
│  │   │   │                                                         │    │   │ │
│  │   │   │   Hidden State h                                        │    │   │ │
│  │   │   │        │                                                │    │   │ │
│  │   │   │        ├──────────────────────────┐                     │    │   │ │
│  │   │   │        │                          │                     │    │   │ │
│  │   │   │        ▼                          ▼                     │    │   │ │
│  │   │   │  ┌───────────────┐         ┌───────────────┐            │    │   │ │
│  │   │   │  │  Policy Head  │         │  Value Head   │            │    │   │ │
│  │   │   │  │ MLP(256→512   │         │ MLP(256→128   │            │    │   │ │
│  │   │   │  │   →action_sz) │         │   →1) + Tanh  │            │    │   │ │
│  │   │   │  │ + Log-Softmax │         │               │            │    │   │ │
│  │   │   │  └───────────────┘         └───────────────┘            │    │   │ │
│  │   │   │        │                          │                     │    │   │ │
│  │   │   │        ▼                          ▼                     │    │   │ │
│  │   │   │   π(a|h) policy             v(h) value                  │    │   │ │
│  │   │   └────────────────────────────────────────────────────────┘    │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  MCTS IN LATENT SPACE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   Real Observation o_t                                                   │ │
│  │          │                                                               │ │
│  │          ▼                                                               │ │
│  │   h_0 = representation(o_t)     ←── Root node                           │ │
│  │          │                                                               │ │
│  │   For each simulation:                                                   │ │
│  │          │                                                               │ │
│  │   ┌──────▼──────────────────────────────────────────────────────────┐   │ │
│  │   │  SELECT: UCB-based selection using π(h), v(h)                   │   │ │
│  │   │     │                                                            │   │ │
│  │   │     ▼                                                            │   │ │
│  │   │  EXPAND: h' = dynamics(h, a).next_state                         │   │ │
│  │   │     │    r = dynamics(h, a).reward                               │   │ │
│  │   │     │                                                            │   │ │
│  │   │     ▼                                                            │   │ │
│  │   │  EVALUATE: π', v' = prediction(h')                               │   │ │
│  │   │     │                                                            │   │ │
│  │   │     ▼                                                            │   │ │
│  │   │  BACKUP: Propagate v' up the tree                                │   │ │
│  │   └──────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                          │ │
│  │   Key Advantage: No need for environment simulator during planning!     │ │
│  │                  All rollouts happen in learned latent space            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  TRAINING LOSSES                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   L_total = L_policy + L_value + L_reward + L_consistency               │ │
│  │                                                                          │ │
│  │   L_policy = CrossEntropy(π_pred, π_mcts)     ← Match MCTS policy       │ │
│  │   L_value  = MSE(v_pred, v_target)            ← Match game outcome      │ │
│  │   L_reward = MSE(r_pred, r_actual)            ← Predict rewards         │ │
│  │   L_consistency = ||h_repr - h_dyn||²         ← Latent alignment        │ │
│  │                                                (EfficientZero addition)  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  PARAMETERS: ~12M total (includes dynamics model)                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Transformer Network Architecture (Advanced)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER NETWORK ARCHITECTURE                           │
│                  (Vision Transformer for Game State)                          │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     PATCH EMBEDDING INPUT                                │ │
│  │                                                                          │ │
│  │   Spatial Input (84 × H × W)                                             │ │
│  │          │                                                               │ │
│  │          ▼                                                               │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │   │ Patch Embedding (patch_size = 4)                                  │  │ │
│  │   │                                                                   │  │ │
│  │   │   Split into non-overlapping patches:                             │  │ │
│  │   │   84 × H × W → (H/4 × W/4) patches of 84 × 4 × 4                  │  │ │
│  │   │                                                                   │  │ │
│  │   │   Linear projection: 84 × 4 × 4 = 1344 → 256 (embed_dim)         │  │ │
│  │   │                                                                   │  │ │
│  │   │   Result: Sequence of (H/4 × W/4) tokens, each 256-dim           │  │ │
│  │   └──────────────────────────────────────────────────────────────────┘  │ │
│  │          │                                                               │ │
│  │          ▼                                                               │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │   │ Positional Encoding (RoPE - Rotary Position Embedding)            │  │ │
│  │   │                                                                   │  │ │
│  │   │   - 2D extension of RoPE for grid positions                       │  │ │
│  │   │   - Encodes (row, col) position into rotation matrices            │  │ │
│  │   │   - Applied directly to Q and K in attention                      │  │ │
│  │   │   - Better extrapolation to larger boards than learned PE         │  │ │
│  │   └──────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  TRANSFORMER ENCODER                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │   │ Transformer Block (× num_layers, default 6)                       │  │ │
│  │   │                                                                   │  │ │
│  │   │   Input tokens                                                    │  │ │
│  │   │        │                                                          │  │ │
│  │   │        ▼                                                          │  │ │
│  │   │   ┌────────────────────────────────────────────────────────────┐ │  │ │
│  │   │   │ Pre-Norm (LayerNorm or RMSNorm)                            │ │  │ │
│  │   │   │        │                                                   │ │  │ │
│  │   │   │        ▼                                                   │ │  │ │
│  │   │   │ Multi-Head Self-Attention (8 heads)                        │ │  │ │
│  │   │   │                                                            │ │  │ │
│  │   │   │   ┌─────────────────────────────────────────────────────┐ │ │  │ │
│  │   │   │   │ Attention Types (configurable):                     │ │ │  │ │
│  │   │   │   │                                                      │ │ │  │ │
│  │   │   │   │ 1. STANDARD: Full O(n²) attention                   │ │ │  │ │
│  │   │   │   │    Attn(Q,K,V) = softmax(QK^T/√d)V                  │ │ │  │ │
│  │   │   │   │                                                      │ │ │  │ │
│  │   │   │   │ 2. FLASH: Memory-efficient attention                │ │ │  │ │
│  │   │   │   │    Fused kernel, reduced memory O(n) → O(√n)        │ │ │  │ │
│  │   │   │   │                                                      │ │ │  │ │
│  │   │   │   │ 3. LINEAR: O(n) approximation                       │ │ │  │ │
│  │   │   │   │    φ(Q)φ(K)^T V (kernel feature maps)               │ │ │  │ │
│  │   │   │   └─────────────────────────────────────────────────────┘ │ │  │ │
│  │   │   │                                                            │ │  │ │
│  │   │   │        + Residual Connection                               │ │  │ │
│  │   │   └────────────────────────────────────────────────────────────┘ │  │ │
│  │   │        │                                                          │  │ │
│  │   │        ▼                                                          │  │ │
│  │   │   ┌────────────────────────────────────────────────────────────┐ │  │ │
│  │   │   │ Pre-Norm                                                   │ │  │ │
│  │   │   │        │                                                   │ │  │ │
│  │   │   │        ▼                                                   │ │  │ │
│  │   │   │ Feed-Forward Network                                       │ │  │ │
│  │   │   │                                                            │ │  │ │
│  │   │   │   Standard FFN:                                            │ │  │ │
│  │   │   │   Linear(256 → 1024) → GELU → Linear(1024 → 256)          │ │  │ │
│  │   │   │                                                            │ │  │ │
│  │   │   │   OR Mixture of Experts (optional):                        │ │  │ │
│  │   │   │   ┌─────────────────────────────────────────────────────┐ │ │  │ │
│  │   │   │   │ Router: softmax(W_router × input) → expert weights  │ │ │  │ │
│  │   │   │   │                                                      │ │ │  │ │
│  │   │   │   │   Expert 1 ──┐                                       │ │ │  │ │
│  │   │   │   │   Expert 2 ──┼──► Weighted sum (top-k routing)      │ │ │  │ │
│  │   │   │   │   Expert 3 ──┤   k=2 experts active per token       │ │ │  │ │
│  │   │   │   │   Expert 4 ──┘                                       │ │ │  │ │
│  │   │   │   │                                                      │ │ │  │ │
│  │   │   │   │   + Load balancing loss for expert utilization       │ │ │  │ │
│  │   │   │   └─────────────────────────────────────────────────────┘ │ │  │ │
│  │   │   │                                                            │ │  │ │
│  │   │   │        + Residual Connection                               │ │  │ │
│  │   │   └────────────────────────────────────────────────────────────┘ │  │ │
│  │   │        │                                                          │  │ │
│  │   │        ▼                                                          │  │ │
│  │   │   Output tokens (same shape as input)                             │  │ │
│  │   └──────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  OUTPUT HEADS                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   Encoded Tokens                                                         │ │
│  │        │                                                                 │ │
│  │        ├───────────────────────────────────────────┐                     │ │
│  │        │                                           │                     │ │
│  │        ▼                                           ▼                     │ │
│  │   ┌────────────────────────┐              ┌────────────────────────┐    │ │
│  │   │     POLICY HEAD        │              │      VALUE HEAD         │    │ │
│  │   │                        │              │                         │    │ │
│  │   │ Option 1: Flat         │              │ Global Average Pool     │    │ │
│  │   │ - Pool → Linear(→N)    │              │        │                │    │ │
│  │   │                        │              │        ▼                │    │ │
│  │   │ Option 2: Hierarchical │              │ MLP(256→128→1)         │    │ │
│  │   │ - Same as Civ6NetV2    │              │        │                │    │ │
│  │   │ - Category→Selector→   │              │        ▼                │    │ │
│  │   │   Action→Target        │              │ Tanh → v(s)            │    │ │
│  │   └────────────────────────┘              └────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  CONFIGURATION OPTIONS                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │   TransformerConfig:                                                     │ │
│  │   ├── embed_dim: 256 (token dimension)                                   │ │
│  │   ├── num_heads: 8 (attention heads)                                     │ │
│  │   ├── num_layers: 6 (transformer blocks)                                 │ │
│  │   ├── mlp_ratio: 4.0 (FFN expansion)                                     │ │
│  │   ├── dropout: 0.1                                                       │ │
│  │   ├── attention_type: STANDARD | FLASH | LINEAR                          │ │
│  │   ├── norm_type: LAYER_NORM | RMS_NORM                                   │ │
│  │   ├── use_moe: false (enable Mixture of Experts)                         │ │
│  │   ├── num_experts: 4 (if MoE enabled)                                    │ │
│  │   └── experts_per_token: 2 (top-k routing)                               │ │
│  │                                                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  PARAMETERS: ~6-15M depending on configuration                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Network Comparison Summary

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      NETWORK COMPARISON MATRIX                                │
│                                                                               │
│  ┌─────────────────┬───────────┬─────────────┬───────────┬─────────────────┐ │
│  │ Attribute       │ MicroCiv  │ Civ6NetV2   │ MuZero    │ Transformer     │ │
│  ├─────────────────┼───────────┼─────────────┼───────────┼─────────────────┤ │
│  │ Parameters      │ ~500K     │ ~8M         │ ~12M      │ ~6-15M          │ │
│  │ Spatial Input   │ 13 ch     │ 84 ch       │ 84 ch     │ 84 ch           │ │
│  │ Vector Input    │ 64 dim    │ 512 dim     │ 512 dim   │ 512 dim         │ │
│  │ Action Space    │ 381       │ 50K+ (hier) │ 50K+      │ Configurable    │ │
│  │ Architecture    │ ResNet    │ ResNet+Attn │ 3-func    │ ViT             │ │
│  │ Attention       │ None      │ Spatial SA  │ Optional  │ Full Self-Attn  │ │
│  │ World Model     │ No        │ No          │ Yes       │ No              │ │
│  │ Use Case        │ Training  │ Production  │ Planning  │ Research        │ │
│  │ Board Size      │ 16×16     │ Variable    │ Variable  │ Variable        │ │
│  │ Training Speed  │ Fastest   │ Medium      │ Slowest   │ Fast (GPU)      │ │
│  └─────────────────┴───────────┴─────────────┴───────────┴─────────────────┘ │
│                                                                               │
│  TRAINING PROGRESSION:                                                       │
│                                                                               │
│   MicroCivNetwork ──► Civ6NetworkV2 ──► MuZero/Transformer                  │
│   (fast iteration)    (production)      (advanced planning)                  │
│                                                                               │
│   Curriculum Learning: Transfer encoder weights between stages               │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Level 2: Agent Data Mesh Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT DATA MESH                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Memory     │  │   SQLite     │  │   ChromaDB   │              │
│  │  (Graph KB)  │  │ (Structured) │  │  (Vectors)   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │  Filesystem │                                  │
│                    │   (Shared)  │                                  │
│                    └─────────────┘                                  │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                       │
│         │                 │                 │                       │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐               │
│  │ Civ6 Agents │  │ AlphaGalerkin│  │  Global     │               │
│  │             │  │    Agents    │  │  Agents     │               │
│  └─────────────┘  └──────────────┘  └─────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Stores

| Store | Purpose | Access Method | Use Cases |
|-------|---------|---------------|-----------|
| Memory Server | Entity relationships, cross-session context | MCP `memory` tools | "Remember that network X worked well" |
| SQLite | Structured, queryable experiment data | MCP `sqlite` tools | Experiments, agent logs, decisions, metrics |
| ChromaDB | Vector embeddings for semantic search | MCP `chroma` tools | Code patterns, error solutions, strategy docs |
| Filesystem | Shared files across projects | MCP `filesystem` tools | Knowledge bases, project data |

### Agent Data Access Matrix

| Agent | Primary Store | Secondary | Use Case |
|-------|--------------|-----------|----------|
| ml-training-orchestrator | SQLite | Memory | Track experiments |
| experiment-reviewer | W&B + SQLite | Memory | Analyze runs |
| neural-network-architect | Memory | ChromaDB | Design patterns |
| mcts-optimizer | Memory | SQLite | Tuning history |
| code-quality | ChromaDB | - | Error patterns |
| test-runner | SQLite | - | Test history |
| rag-specialist | ChromaDB + Pinecone | - | Embeddings |

---

## Level 2: GCP Cloud Infrastructure (Phase 2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GCP DISTRIBUTED TRAINING INFRASTRUCTURE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         CLOUD CONFIGURATION                          │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│   │   │  CloudConfig │  │ Environment │  │    YAML     │                │   │
│   │   │  (Dataclass) │◄─┤  Variables  │◄─┤   Config    │                │   │
│   │   │             │  │  (GCP_/CIV6)|  │   Files     │                │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      GCP SERVICE INTEGRATIONS                        │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐    ┌──────────────────┐                       │   │
│   │  │  GCS Checkpoint  │    │  Vertex AI       │                       │   │
│   │  │     Manager      │    │    Trainer       │                       │   │
│   │  ├──────────────────┤    ├──────────────────┤                       │   │
│   │  │ • Upload/Download│    │ • Job submission │                       │   │
│   │  │ • Metadata track │    │ • GPU/TPU config │                       │   │
│   │  │ • Version control│    │ • Spot instances │                       │   │
│   │  │ • Mock support   │    │ • Monitoring     │                       │   │
│   │  └──────────────────┘    └──────────────────┘                       │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐    ┌──────────────────┐                       │   │
│   │  │   Preemption     │    │  Cloud Logger/   │                       │   │
│   │  │    Handler       │    │  Metrics Export  │                       │   │
│   │  ├──────────────────┤    ├──────────────────┤                       │   │
│   │  │ • 30s warning    │    │ • Structured logs│                       │   │
│   │  │ • Checkpoint CB  │    │ • Custom metrics │                       │   │
│   │  │ • Signal handlers│    │ • Correlation ID │                       │   │
│   │  │ • Graceful stop  │    │ • Severity levels│                       │   │
│   │  └──────────────────┘    └──────────────────┘                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     DISTRIBUTED TRAINING LAYER                       │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐    ┌──────────────────┐                       │   │
│   │  │    DDP Learner   │◄──►│ Cloud Coordinator│                       │   │
│   │  ├──────────────────┤    ├──────────────────┤                       │   │
│   │  │ • Multi-GPU sync │    │ • Resource mgmt  │                       │   │
│   │  │ • Gradient accum │    │ • Auto-scaling   │                       │   │
│   │  │ • Checkpoint sync│    │ • Job lifecycle  │                       │   │
│   │  └──────────────────┘    └──────────────────┘                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GCP Components

| Component | Location | Purpose |
|-----------|----------|---------|
| CloudConfig | `alpha_zero/cloud/config.py` | Unified configuration with env/yaml loading |
| GCSCheckpointManager | `alpha_zero/cloud/gcp/storage.py` | Checkpoint storage in GCS |
| VertexAITrainer | `alpha_zero/cloud/gcp/vertex.py` | Managed training job submission |
| PreemptionHandler | `alpha_zero/cloud/gcp/preemption.py` | Graceful spot instance handling |
| CloudLogger | `alpha_zero/cloud/gcp/monitoring.py` | Cloud Logging integration |
| CloudMetricsExporter | `alpha_zero/cloud/gcp/monitoring.py` | Cloud Monitoring metrics |
| DDPLearner | `alpha_zero/distributed/ddp_learner.py` | PyTorch DDP multi-GPU training |
| CloudCoordinator | `alpha_zero/distributed/cloud_coordinator.py` | GCP resource orchestration |

### Mock Support Pattern

All GCP components support `force_mock=True` for testing without SDK:

```python
# Testing without GCP
manager = GCSCheckpointManager(config, force_mock=True)
trainer = VertexAITrainer(config, force_mock=True)
handler = PreemptionHandler(force_mock=True)
```

---

## Level 3: Component Diagram - PBT Production Pipeline (Phase 7)

Production-hardened Population-Based Training with W&B dashboards, GCS checkpoint rotation, GPU quota guarding, and baseline benchmarking.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PBT PRODUCTION PIPELINE (Phase 7)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        PBT ORCHESTRATOR                              │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │  Generation   │  │   Quota      │  │  Strategy    │             │   │
│   │   │    Loop       │──│   Guard      │──│   Switch     │             │   │
│   │   │              │  │  (GPU check) │  │(round_robin→ │             │   │
│   │   │  selfplay    │  │  exp backoff │  │ top_k_vs_best│             │   │
│   │   │  → train     │  └──────────────┘  │ at gen N)    │             │   │
│   │   │  → evaluate  │                    └──────────────┘             │   │
│   │   │  → evolve    │                                                  │   │
│   │   │  → rotate    │  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │  → dashboard │  │  Checkpoint  │  │   W&B        │             │   │
│   │   │              │──│  Rotation    │──│  Dashboard   │             │   │
│   │   └──────────────┘  │  (manifest)  │  │  (pbt/*)     │             │   │
│   │                      └──────────────┘  └──────────────┘             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     EVALUATION LAYER                                 │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐    ┌──────────────────┐                       │   │
│   │  │  PBTEvaluator    │    │  Baseline         │                       │   │
│   │  ├──────────────────┤    │  Benchmarks       │                       │   │
│   │  │ • Round-robin /  │    ├──────────────────┤                       │   │
│   │  │   top_k_vs_best  │    │ • Random agent   │                       │   │
│   │  │ • ELO ratings    │    │ • Heuristic agent│                       │   │
│   │  │ • Fitness scores │    │ • Interval-based │                       │   │
│   │  │ • Arena runner   │    │ • No ELO impact  │                       │   │
│   │  └──────────────────┘    └──────────────────┘                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     OBSERVABILITY                                    │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐    ┌──────────────────┐                       │   │
│   │  │  Population      │    │  Mutation         │                       │   │
│   │  │  Stats           │    │  Events           │                       │   │
│   │  ├──────────────────┤    ├──────────────────┤                       │   │
│   │  │ pbt/elo_max      │    │ pbt/mutations_cnt│                       │   │
│   │  │ pbt/elo_mean     │    │ W&B table:       │                       │   │
│   │  │ pbt/elo_min      │    │  replaced_id     │                       │   │
│   │  │ pbt/fitness_*    │    │  source_id       │                       │   │
│   │  │ pbt/agent/*/elo  │    │  new_lr, new_c   │                       │   │
│   │  └──────────────────┘    └──────────────────┘                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 7 Components

| Component | Location | Purpose |
|-----------|----------|---------|
| PBTOrchestrator | `alpha_zero/cloud/gcp/pbt_orchestrator.py` | Production PBT loop with all Phase 7 features |
| PBTEvaluator | `alpha_zero/pbt/evaluator.py` | Cross-agent ELO evaluation + baseline benchmarks |
| PBTConfig | `alpha_zero/pbt/config.py` | Configuration with 4 new Phase 7 fields |
| WandbLogger | `alpha_zero/utils/wandb_logger.py` | Population dashboards + PBT chart definitions |
| PopulationManager | `alpha_zero/pbt/population.py` | Agent lifecycle, exploit & explore |
| MutationOperator | `alpha_zero/pbt/mutations.py` | 6 mutation strategies (Gaussian, Adaptive, etc.) |
| PBTScheduler | `alpha_zero/pbt/scheduler.py` | Generation timing control |
| pbt_prod.yaml | `configs/pbt_prod.yaml` | Production config with GumbelMCTS-tuned ranges |

### Phase 7 Data Flow

```text
Generation N:
  1. _maybe_switch_strategy(N)     → evaluation_strategy updated if N >= switch_gen
  2. _check_quota(N)               → GPU availability verified (optimistic fallback)
  3. _wait_for_quota(N)            → Exponential backoff if over quota
  4. _run_selfplay_phase(N)        → GumbelMCTS self-play on Vertex AI
  5. _run_training_phase(N)        → Neural network training on Vertex AI
  6. _run_evaluation_phase(N)      → Cross-agent ELO + baseline benchmarks
  7. _run_evolution_phase(N)       → Exploit (truncation) + Explore (mutation)
  8. _save_state(N)                → GCS state persistence
  9. _rotate_checkpoints(N)        → Prune old generations from GCS
 10. _log_population_dashboard(N)  → W&B pbt/* metrics
 11. _log_mutation_events(N)       → W&B mutation table
```

---

## Level 3: Component Diagram - Batch Inference & Vectorized Self-Play (Phase 16)

Batch inference optimization and vectorized self-play for throughput scaling.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              BATCH INFERENCE & VECTORIZED SELF-PLAY (Phase 16)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        BATCH MCTS ENGINE                               │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │  │
│  │  │ BatchGumbelMCTS   │   │  BatchMCTSConfig │   │  MCTSVariant     │  │  │
│  │  │ - batch_search()  │◄──│  - batch_size    │   │  .BATCH_GUMBEL   │  │  │
│  │  │ - batch_evaluate()│   │  - simulations   │   │  (mcts_factory)  │  │  │
│  │  │ - select_action() │   │  - search_mode   │   └──────────────────┘  │  │
│  │  └────────┬─────────┘   └──────────────────┘                          │  │
│  │           │                                                            │  │
│  │           ├───── batch_simple_search()      ← Collect all pending     │  │
│  │           │                                     leaf evaluations,      │  │
│  │           │                                     single _batch_evaluate │  │
│  │           │                                                            │  │
│  │           └───── batch_sequential_halving() ← Gumbel sampling with    │  │
│  │                                                batched evaluation      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    VECTORIZED ENVIRONMENTS                              │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │  │
│  │  │VectorizedMicroCiv│   │VectorizedEnvConf │   │  Pre-allocated   │  │  │
│  │  │ - reset_all()    │◄──│ - num_envs       │   │  Numpy Buffers   │  │  │
│  │  │ - step_all()     │   │ - board_size     │   │ - spatial obs    │  │  │
│  │  │ - clone_env(i)   │   │ - max_turns      │   │ - vector obs     │  │  │
│  │  │ - get_env(i)     │   └──────────────────┘   │ - rewards        │  │  │
│  │  └──────────────────┘                           └──────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      GPU OPTIMIZATIONS                                  │  │
│  │                                                                        │  │
│  │  ┌──────────────────┐   ┌──────────────────┐                          │  │
│  │  │ CUDAGraphCache   │   │InferenceBenchmark│                          │  │
│  │  │ - capture graph  │   │ - single vs batch│                          │  │
│  │  │ - replay graph   │   │ - MCTS search    │                          │  │
│  │  │ - CPU fallback   │   │ - vectorized     │                          │  │
│  │  └──────────────────┘   │   self-play      │                          │  │
│  │                          └──────────────────┘                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      CLOUD INTEGRATION                                  │  │
│  │                                                                        │  │
│  │  entrypoint.py ──→ --mode=orchestrator ──→ vertex_orchestrator.main()  │  │
│  │                                                                        │  │
│  │  selfplay_worker.py ──→ --num_parallel_games=N --max_batch_size=M     │  │
│  │                    ──→ run_batch_selfplay()                             │  │
│  │                    ──→ BatchGumbelMCTS + VectorizedMicroCivEnv          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Roadmap

### Completed Phases

1. **Phase 1**: Core AlphaZero — MCTS + Neural Network ✅
2. **Phase 2**: Hierarchical RL — Multi-level decision making ✅
3. **Phase 3**: Training Improvements — Curriculum, rewards, tuning ✅
4. **Phase 4**: Unified Orchestrator — Single training entry point ✅
5. **Phase 5**: RAG Integration — Knowledge-enhanced decisions ✅
6. **Phase 6**: ~~Agentic Framework~~ — Removed in Phase 11 (14,600 LOC dead code) ❌
7. **Phase 7**: GCP Vertex AI Pipeline — Cloud training with GumbelMCTS, 3-phase pipeline ✅
8. **Phase 7+**: Production PBT Scaling — W&B dashboards, checkpoint rotation, quota guard ✅
9. **Phase 8**: Training Safeguards — NaN detection, loss explosion, gradient health, auto-intervention ✅
10. **Phase 9**: MuZero Integration — MuZeroTrainingBridge, MuZeroMCTS, PBT hyperparameters ✅
11. **Phase 10**: Tournament System — Round-robin, elimination, Swiss brackets, ELO, W&B dashboards ✅
12. **Phase 11**: Code Quality — Deleted 14.6k LOC, fixed xfails, CI hardened, mypy strict ✅
13. **Phase 12-Prep**: Docker v0.9.0 → GCR, container smoke tests, Vertex AI smoke test ($0.04) ✅
14. **Phase 13**: Multi-Iteration Training — NetworkArchitectureConfig, arena module, post-run tournament, W&B orchestrator logging, 6 bug fixes, Docker v0.10.2 ✅
15. **Phase 14**: Learning Signal Validation — 20-iter run ($2.44), 2 promotions (v6→v19→v26), training diagnostics, warmup-cosine LR, stall detection ✅
16. **Phase 15**: Value Head Stabilization — Per-head LR multipliers (value=0.1x), optimizer param groups, base_lr_ratio preservation ✅
17. **Phase 16**: Batch Inference & Vectorized Self-Play — BatchGumbelMCTS, VectorizedMicroCivEnv, CUDAGraphCache, InferenceBenchmark, cloud orchestrator mode, 101 new tests, Docker v0.13.0 ✅
18. **Phase 17a**: Hex Crystalline Encoder & A/B Comparison — HexSpatialEncoder, CrystallineAttention, HexConv2d, EncoderRegistry, cloud pipeline encoder propagation, Python 3.10 compat shim, gameplay systems (terrain, fog, tech, combat), A/B training runs on VM, Docker v0.14.1 ✅
19. **Phase 17b**: Training Readiness (v1.2.0) — action_mask in 5-tensor TensorDataset + masked cross-entropy, W&B Group coordination (WANDB_GROUP/WANDB_RUN_NAME per phase), eval worker W&B logging, evaluation retry with backoff (max_retries=2), LR forwarding, GumbelMCTS as default, MockStorageBackend centralized, GCS trajectory cleanup script, pytest-timeout=120, Cloud Build c72f98e9 (8 steps), Docker v1.2.0 on Artifact Registry, 50-iteration production run active ✅

### Next Steps

See [NEXT_STEPS.md](../NEXT_STEPS.md) for the full post-Phase 17 roadmap.

| Priority | Feature | Description |
|----------|---------|-------------|
| P0 | Monitor Production Run (`dashing-firebrand-1393`) | 50 iters, 4×50 games, GumbelMCTS 16 sims — active on `civ6az-orchestrator` VM |
| P1 | Batch Self-Play Cloud Integration (Phase 18) | Enable `--num_parallel_games` in cloud for 5-10x throughput |
| P2 | MuZero Training Run (Phase 19) | Validate learned dynamics model on MicroCiv |
| P3 | PBT on Vertex AI (Phase 20) | Population-based training with auto hyperparameter tuning |
| P4 | DDP Test Rewrites (Phase 21) | Fix 12 remaining DDP xfails, full mypy strict |
| P5 | Human vs AI Interface (Phase 22) | Web-based UI for playing against trained agent |
| P6 | Full Civ6 Integration (Phase 23) | Direct game interface via Lua mod bridge |

---

*Last updated: 2026-02-21*
*Architecture version: 6.0*
