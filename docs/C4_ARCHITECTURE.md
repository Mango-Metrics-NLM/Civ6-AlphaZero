# C4 Architecture - Civ6 AlphaZero AI Framework

> Architecture documentation using the C4 model for comprehensive system visualization.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core AlphaZero (MCTS + Neural Network) | ✅ Complete |
| Phase 2 | Hierarchical Reinforcement Learning | ✅ Complete |
| Phase 3 | Training Improvements (Curriculum, Rewards) | ✅ Complete |
| Phase 4 | Unified Training Orchestrator | ✅ Complete |
| Phase 5 | RAG-LLM Integration | ✅ Complete |
| Phase 6 | Agentic Framework for Claude Code | ✅ Complete |

**Codebase Statistics:** ~51,800 LOC core + ~14,600 LOC agentic framework, 350+ tests

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
| Core AI | Neural Network | PyTorch | Model training/inference |
| Core AI | MCTS | NumPy | Tree search |
| Environment | MicroCiv | Gymnasium | Training environment |
| Environment | Unciv | JSON/Python | Bridge environment |
| Environment | Civ6 | Lua mod | Production game |
| Training | Orchestrator | Python | Training coordination |
| Training | Distributed | multiprocessing | Parallel self-play |
| Training | Checkpointing | PyTorch | Model persistence |
| Knowledge | RAG | ChromaDB/FAISS | Strategy knowledge |
| Knowledge | LLM | OpenAI/Anthropic | Strategy advisor |
| Logging | Metrics | TensorBoard/W&B | Experiment tracking |
| Logging | Structured | JSON/Python logging | Debug/audit |
| Config | Settings | YAML/Pydantic | Configuration management |
| Testing | Framework | pytest | Test execution |
| Linting | Code Quality | ruff/mypy | Static analysis |

---

## Key Architectural Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Hierarchical action space | Reduces branching factor from 10^6 to 10^3 | Complexity in encoding/decoding |
| Curriculum learning | Faster convergence, better generalization | Requires environment abstraction |
| Process-based distribution | Avoid GIL, maximize CPU for MCTS | IPC overhead |
| ResNet + Attention | Proven spatial reasoning + global context | Computational cost |
| JSON state serialization | Human-readable, compatible with Unciv | Parsing overhead |
| YAML configuration | Readable, environment overrides | Schema validation needed |
| Action pruning | Reduce search space intelligently | May prune good actions |
| Macro actions | Multi-turn planning abstraction | Learning complexity |
| Hierarchical RL | Multi-level decision making | Coordination complexity |
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

## Level 2: Agentic Framework (Phase 6)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC FRAMEWORK (~14,600 LOC)                       │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           CORE COMPONENTS                               │  │
│  │                                                                         │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │  │
│  │   │  Constraint  │    │  Permission  │    │   Session    │             │  │
│  │   │  Manager     │    │  Manager     │    │   Memory     │             │  │
│  │   │ - hard/soft  │    │ - scope      │    │ - CLAUDE.md  │             │  │
│  │   │ - validate() │    │ - autonomy   │    │ - persist()  │             │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          SPECIALIZED AGENTS                             │  │
│  │                                                                         │  │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │  │
│  │   │  Planning   │ │Implementation│ │  Testing   │ │Verification │     │  │
│  │   │   Agent     │ │    Agent    │ │   Agent    │ │   Agent     │     │  │
│  │   │ - analyze   │ │ - write code│ │ - pytest   │ │ - lint/type │     │  │
│  │   │ - plan      │ │ - rollback  │ │ - coverage │ │ - verify    │     │  │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │  │
│  │                                                                         │  │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    DOCUMENTATION AGENT                          │  │  │
│  │   │  - Update CLAUDE.md with new patterns                           │  │  │
│  │   │  - Generate docstrings and README updates                       │  │  │
│  │   │  - Track architectural decisions                                │  │  │
│  │   └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           FEEDBACK LOOP                                 │  │
│  │                                                                         │  │
│  │   ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │   │ 1. Lint → 2. Type Check → 3. Unit Test → 4. Integration Test    │ │  │
│  │   │         ↓ (on failure)                                           │ │  │
│  │   │ 5. Auto-fix → 6. Retry (max 3x) → 7. Report with hypothesis     │ │  │
│  │   └──────────────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Roadmap

### Completed Phases

1. **Phase 1**: Core AlphaZero - MCTS + Neural Network ✅
2. **Phase 2**: Hierarchical RL - Multi-level decision making ✅
3. **Phase 3**: Training Improvements - Curriculum, rewards, tuning ✅
4. **Phase 4**: Unified Orchestrator - Single training entry point ✅
5. **Phase 5**: RAG Integration - Knowledge-enhanced decisions ✅
6. **Phase 6**: Agentic Framework - Claude Code automation ✅

### Planned Features

| Priority | Feature | Description |
|----------|---------|-------------|
| P1 | Training Validation | Achieve >50% win rate vs random |
| P2 | Multi-GPU Training | DistributedDataParallel support |
| P3 | Tournament System | ELO-based agent ranking |
| P4 | Population-Based Training | Adaptive hyperparameters |
| P5 | Human vs AI Interface | Web-based gameplay UI |
| P6 | Full Civ 6 Integration | Direct game modding API |

---

*Last updated: 2026-01-24*
*Architecture version: 3.0*
