#!/usr/bin/env python3
"""Generate a multi-tab Draw.io C4 architecture diagram for Civ6-AlphaZero."""

import xml.etree.ElementTree as ET
from xml.dom import minidom

# ─── Style Constants ───────────────────────────────────────────────────────────
STYLES = {
    "person": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#08427B;fontColor=#ffffff;"
        "strokeColor=#052E56;arcSize=50;fontSize=13;fontStyle=1;verticalAlign=middle;"
    ),
    "system_internal": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#1168BD;fontColor=#ffffff;"
        "strokeColor=#0B4884;arcSize=10;fontSize=13;fontStyle=1;"
    ),
    "system_external": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#999999;fontColor=#ffffff;"
        "strokeColor=#6B6B6B;arcSize=10;fontSize=13;fontStyle=1;"
    ),
    "container": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#438DD5;fontColor=#ffffff;"
        "strokeColor=#3C7FC0;arcSize=10;fontSize=12;"
    ),
    "container_db": (
        "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;"
        "size=15;fillColor=#438DD5;fontColor=#ffffff;strokeColor=#3C7FC0;fontSize=12;"
    ),
    "container_queue": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#438DD5;fontColor=#ffffff;"
        "strokeColor=#3C7FC0;arcSize=10;fontSize=12;dashed=1;dashPattern=5 5;"
    ),
    "component": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#85BBF0;fontColor=#000000;"
        "strokeColor=#78A8D8;arcSize=10;fontSize=11;"
    ),
    "component_key": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#5BA0E0;fontColor=#ffffff;"
        "strokeColor=#4A8AC8;arcSize=10;fontSize=11;fontStyle=1;"
    ),
    "boundary": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#1168BD;"
        "strokeWidth=2;dashed=1;arcSize=5;fontSize=16;fontStyle=1;"
        "fontColor=#1168BD;verticalAlign=top;align=left;spacingLeft=10;spacingTop=5;"
    ),
    "boundary_sub": (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#438DD5;"
        "strokeWidth=1;dashed=1;arcSize=5;fontSize=13;fontStyle=1;"
        "fontColor=#438DD5;verticalAlign=top;align=left;spacingLeft=10;spacingTop=5;"
    ),
    "arrow": (
        "endArrow=blockThin;endFill=1;html=1;strokeColor=#666666;"
        "fontColor=#333333;fontSize=10;exitX=0.5;exitY=1;exitDx=0;exitDy=0;"
    ),
    "arrow_plain": (
        "endArrow=blockThin;endFill=1;html=1;strokeColor=#666666;"
        "fontColor=#333333;fontSize=10;"
    ),
    "arrow_dashed": (
        "endArrow=blockThin;endFill=1;html=1;strokeColor=#666666;"
        "fontColor=#333333;fontSize=10;dashed=1;dashPattern=5 5;"
    ),
    "note": (
        "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;size=15;"
        "fillColor=#FFF2CC;strokeColor=#D6B656;fontSize=11;align=left;"
        "spacingLeft=5;spacingRight=5;spacingTop=5;"
    ),
    "legend_title": (
        "text;html=1;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=14;fontStyle=5;"
    ),
    "legend_item": (
        "text;html=1;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;"
    ),
}


def label(name: str, tech: str = "", desc: str = "") -> str:
    """Build a C4-style HTML label."""
    parts = [f"<b>{name}</b>"]
    if tech:
        parts.append(f"<br/><i>[{tech}]</i>")
    if desc:
        parts.append(f'<br/><br/><font style="font-size:10px">{desc}</font>')
    return "".join(parts)


def add_cell(
    root, cell_id, value, style, x, y, w, h, parent="1", vertex=True
):
    """Add a vertex cell."""
    cell = ET.SubElement(root, "mxCell")
    cell.set("id", cell_id)
    cell.set("value", value)
    cell.set("style", style)
    cell.set("parent", parent)
    if vertex:
        cell.set("vertex", "1")
    geo = ET.SubElement(cell, "mxGeometry")
    geo.set("x", str(x))
    geo.set("y", str(y))
    geo.set("width", str(w))
    geo.set("height", str(h))
    geo.set("as", "geometry")
    return cell


def add_edge(root, edge_id, source, target, value="", style=None, parent="1"):
    """Add an edge cell."""
    if style is None:
        style = STYLES["arrow_plain"]
    cell = ET.SubElement(root, "mxCell")
    cell.set("id", edge_id)
    cell.set("value", value)
    cell.set("style", style)
    cell.set("parent", parent)
    cell.set("edge", "1")
    cell.set("source", source)
    cell.set("target", target)
    geo = ET.SubElement(cell, "mxGeometry")
    geo.set("relative", "1")
    geo.set("as", "geometry")
    return cell


def make_diagram(diagram_id: str, name: str):
    """Create a diagram element and return (diagram, root)."""
    diagram = ET.Element("diagram")
    diagram.set("id", diagram_id)
    diagram.set("name", name)
    model = ET.SubElement(diagram, "mxGraphModel")
    model.set("dx", "1422")
    model.set("dy", "794")
    model.set("grid", "1")
    model.set("gridSize", "10")
    model.set("guides", "1")
    model.set("tooltips", "1")
    model.set("connect", "1")
    model.set("arrows", "1")
    model.set("fold", "1")
    model.set("page", "1")
    model.set("pageScale", "1")
    model.set("pageWidth", "1600")
    model.set("pageHeight", "1200")
    model.set("math", "0")
    model.set("shadow", "0")
    root = ET.SubElement(model, "root")
    c0 = ET.SubElement(root, "mxCell")
    c0.set("id", "0")
    c1 = ET.SubElement(root, "mxCell")
    c1.set("id", "1")
    c1.set("parent", "0")
    return diagram, root


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: C4 Level 1 — System Context
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab1():
    d, r = make_diagram("tab1", "C4 Level 1 - System Context")

    # Title
    add_cell(r, "t1_title", (
        '<font style="font-size:20px"><b>System Context Diagram</b></font>'
        '<br/><font style="font-size:12px">Civ6-AlphaZero — AlphaZero-style RL for Civilization VI</font>'
    ), STYLES["legend_title"], 20, 10, 500, 50)

    # ── Person: AI Researcher ──
    add_cell(r, "t1_person", label(
        "AI Researcher / Developer",
        "Person",
        "Trains, evaluates, and monitors<br/>AlphaZero agents for Civ6"
    ), STYLES["person"], 580, 40, 240, 160)

    # ── Central System ──
    add_cell(r, "t1_system", label(
        "Civ6-AlphaZero System",
        "Python / PyTorch",
        "Self-play reinforcement learning system<br/>"
        "using MCTS and deep neural networks<br/>"
        "to learn Civilization VI strategy"
    ), STYLES["system_internal"], 500, 340, 400, 200)

    # ── External Systems ──
    # Left column - Game engines
    add_cell(r, "t1_microciv", label(
        "MicroCiv",
        "Python Environment",
        "Simplified 4X game for<br/>fast Phase 1 training"
    ), STYLES["system_external"], 50, 280, 240, 140)

    add_cell(r, "t1_unciv", label(
        "Unciv / CivRealm",
        "External Game Engine",
        "Open-source Civ V clone<br/>with JSON save interface"
    ), STYLES["system_external"], 50, 460, 240, 140)

    # Right column - Cloud/AI services
    add_cell(r, "t1_openai", label(
        "OpenAI API",
        "External Service",
        "LLM for strategy analysis,<br/>RAG queries, embeddings"
    ), STYLES["system_external"], 1100, 180, 240, 140)

    add_cell(r, "t1_anthropic", label(
        "Anthropic API",
        "External Service",
        "Alternative LLM provider<br/>with automatic failover"
    ), STYLES["system_external"], 1100, 360, 240, 140)

    add_cell(r, "t1_wandb", label(
        "Weights &amp; Biases",
        "External Service",
        "Experiment tracking,<br/>training metrics, dashboards"
    ), STYLES["system_external"], 1100, 540, 240, 140)

    # Bottom row - Infrastructure
    add_cell(r, "t1_vertex", label(
        "Google Vertex AI",
        "Cloud Platform",
        "Distributed training jobs,<br/>Spot VM orchestration"
    ), STYLES["system_external"], 330, 680, 240, 140)

    add_cell(r, "t1_gcs", label(
        "Google Cloud Storage",
        "Cloud Storage",
        "Model checkpoints, game<br/>trajectories, metadata"
    ), STYLES["system_external"], 640, 680, 240, 140)

    add_cell(r, "t1_chromadb", label(
        "ChromaDB / FAISS",
        "Vector Database",
        "RAG knowledge base for<br/>strategy documents"
    ), STYLES["system_external"], 950, 720, 240, 120)

    # ── Relationships ──
    add_edge(r, "t1_e1", "t1_person", "t1_system",
             "Configures training,<br/>monitors progress,<br/>plays against AI",
             STYLES["arrow_plain"])
    add_edge(r, "t1_e2", "t1_system", "t1_microciv",
             "Runs self-play<br/>training games",
             STYLES["arrow_plain"])
    add_edge(r, "t1_e3", "t1_system", "t1_unciv",
             "Interfaces via<br/>JSON state files",
             STYLES["arrow_plain"])
    add_edge(r, "t1_e4", "t1_system", "t1_openai",
             "Strategy analysis,<br/>embeddings [HTTPS]",
             STYLES["arrow_dashed"])
    add_edge(r, "t1_e5", "t1_system", "t1_anthropic",
             "Fallback LLM<br/>provider [HTTPS]",
             STYLES["arrow_dashed"])
    add_edge(r, "t1_e6", "t1_system", "t1_wandb",
             "Logs metrics,<br/>ELO ratings [HTTPS]",
             STYLES["arrow_dashed"])
    add_edge(r, "t1_e7", "t1_system", "t1_vertex",
             "Submits training<br/>jobs [gRPC]",
             STYLES["arrow_plain"])
    add_edge(r, "t1_e8", "t1_system", "t1_gcs",
             "Reads/writes checkpoints<br/>&amp; trajectories",
             STYLES["arrow_plain"])
    add_edge(r, "t1_e9", "t1_system", "t1_chromadb",
             "Stores &amp; queries<br/>strategy knowledge",
             STYLES["arrow_dashed"])

    # Legend
    add_cell(r, "t1_leg1",
             '<font style="font-size:12px"><b>Legend</b></font>',
             STYLES["legend_title"], 20, 850, 100, 20)
    add_cell(r, "t1_leg2", "", STYLES["person"], 20, 880, 30, 20)
    add_cell(r, "t1_leg2t", "Person",
             STYLES["legend_item"], 60, 880, 100, 20)
    add_cell(r, "t1_leg3", "", STYLES["system_internal"], 20, 910, 30, 20)
    add_cell(r, "t1_leg3t", "Internal System",
             STYLES["legend_item"], 60, 910, 120, 20)
    add_cell(r, "t1_leg4", "", STYLES["system_external"], 20, 940, 30, 20)
    add_cell(r, "t1_leg4t", "External System",
             STYLES["legend_item"], 60, 940, 120, 20)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: C4 Level 2 — Container Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab2():
    d, r = make_diagram("tab2", "C4 Level 2 - Containers")

    add_cell(r, "t2_title", (
        '<font style="font-size:20px"><b>Container Diagram</b></font>'
        '<br/><font style="font-size:12px">Civ6-AlphaZero — Internal containers / subsystems</font>'
    ), STYLES["legend_title"], 20, 10, 500, 50)

    # ── System Boundary ──
    add_cell(r, "t2_boundary", "Civ6-AlphaZero System",
             STYLES["boundary"], 30, 70, 1150, 1080)

    # ── Row 1: Entry Points ──
    add_cell(r, "t2_web", label(
        "Web Interface",
        "FastAPI / WebSocket",
        "Human vs AI gameplay,<br/>game replay viewer"
    ), STYLES["container"], 60, 120, 220, 130)

    add_cell(r, "t2_orch", label(
        "Training Orchestrator",
        "Python CLI",
        "Unified entry point for all<br/>training modes with dynamic<br/>hyperparameter scheduling"
    ), STYLES["container"], 320, 120, 260, 130)

    add_cell(r, "t2_cli", label(
        "Cloud Orchestrator",
        "Vertex AI Client",
        "Submits &amp; monitors<br/>distributed cloud jobs"
    ), STYLES["container"], 620, 120, 240, 130)

    # ── Row 2: Core Logic ──
    add_cell(r, "t2_core", label(
        "Core Algorithms",
        "PyTorch",
        "Neural networks (ResNet,<br/>Transformer, MuZero),<br/>MCTS, action space"
    ), STYLES["container"], 60, 310, 220, 150)

    add_cell(r, "t2_envs", label(
        "Game Environments",
        "Gymnasium",
        "MicroCiv (simplified),<br/>Unciv bridge, CivRealm<br/>Docker, vectorized envs"
    ), STYLES["container"], 320, 310, 220, 150)

    add_cell(r, "t2_eval", label(
        "Evaluation Pipeline",
        "Python",
        "Agent matchups, baseline<br/>comparison, sequential<br/>significance testing"
    ), STYLES["container"], 580, 310, 220, 150)

    add_cell(r, "t2_hier", label(
        "Hierarchical Decision",
        "Python",
        "Strategic / Tactical /<br/>Operational multi-level<br/>controller with goals"
    ), STYLES["container"], 840, 310, 220, 150)

    # ── Row 3: Training Infrastructure ──
    add_cell(r, "t2_dist", label(
        "Distributed Training",
        "Python Multiprocessing",
        "Self-play actors, central<br/>learner, DDP multi-GPU,<br/>parameter server"
    ), STYLES["container"], 60, 520, 240, 150)

    add_cell(r, "t2_replay", label(
        "Replay Buffer",
        "Shared Memory / GCS",
        "Game trajectories with<br/>prioritized sampling"
    ), STYLES["container_db"], 340, 530, 200, 130)

    add_cell(r, "t2_storage", label(
        "Storage Backend",
        "Local FS / GCS",
        "Abstract storage for<br/>checkpoints, trajectories,<br/>evaluation results"
    ), STYLES["container_db"], 580, 530, 200, 130)

    add_cell(r, "t2_cloud", label(
        "Cloud Workers",
        "Docker / Vertex AI",
        "Containerized self-play,<br/>training, evaluation<br/>with preemption handling"
    ), STYLES["container"], 820, 520, 240, 150)

    # ── Row 4: Intelligence & Tuning ──
    add_cell(r, "t2_rag", label(
        "RAG System",
        "ChromaDB / FAISS",
        "Knowledge base ingestion,<br/>vector search, context<br/>building for strategy"
    ), STYLES["container"], 60, 730, 220, 140)

    add_cell(r, "t2_tourney", label(
        "Tournament System",
        "Python",
        "Round-robin / Swiss brackets,<br/>ELO ratings, matchmaking,<br/>league seasons"
    ), STYLES["container"], 320, 730, 220, 140)

    add_cell(r, "t2_pbt", label(
        "Population-Based Training",
        "Python",
        "Population of agents,<br/>exploit/explore mutations,<br/>adaptive hyperparams"
    ), STYLES["container"], 580, 730, 240, 140)

    add_cell(r, "t2_utils", label(
        "Utilities",
        "Python",
        "Config management, LLM<br/>client, logging, factories,<br/>test runner, agent tools"
    ), STYLES["container"], 860, 730, 200, 140)

    # ── Row 5: Config ──
    add_cell(r, "t2_config", label(
        "Configuration",
        "YAML + Env Overrides",
        "Presets (fast/balanced/production),<br/>schedule configs, machine profiles"
    ), STYLES["container"], 60, 940, 300, 100)

    add_cell(r, "t2_kb", label(
        "Knowledge Base",
        "Markdown / JSONL",
        "Session state, ledger,<br/>handoffs, summaries"
    ), STYLES["container_db"], 400, 945, 200, 90)

    # ── External Systems (right side) ──
    ext_x = 1250
    add_cell(r, "t2_ext_llm", label(
        "LLM APIs",
        "OpenAI / Anthropic"
    ), STYLES["system_external"], ext_x, 120, 200, 80)

    add_cell(r, "t2_ext_vertex", label(
        "Google Vertex AI",
        "Cloud Platform"
    ), STYLES["system_external"], ext_x, 240, 200, 80)

    add_cell(r, "t2_ext_gcs", label(
        "Google Cloud Storage",
        "Object Storage"
    ), STYLES["system_external"], ext_x, 360, 200, 80)

    add_cell(r, "t2_ext_wandb", label(
        "Weights &amp; Biases",
        "Experiment Tracking"
    ), STYLES["system_external"], ext_x, 480, 200, 80)

    add_cell(r, "t2_ext_games", label(
        "Game Engines",
        "MicroCiv / Unciv"
    ), STYLES["system_external"], ext_x, 600, 200, 80)

    add_cell(r, "t2_ext_vectordb", label(
        "Vector Database",
        "ChromaDB / FAISS"
    ), STYLES["system_external"], ext_x, 720, 200, 80)

    # ── Internal Edges ──
    add_edge(r, "t2_e01", "t2_orch", "t2_core", "Uses networks &amp; MCTS")
    add_edge(r, "t2_e02", "t2_orch", "t2_dist", "Launches actors &amp; learner")
    add_edge(r, "t2_e03", "t2_orch", "t2_eval", "Triggers evaluation")
    add_edge(r, "t2_e04", "t2_orch", "t2_rag", "Queries strategy context")
    add_edge(r, "t2_e05", "t2_core", "t2_envs", "Interacts with game env")
    add_edge(r, "t2_e06", "t2_dist", "t2_replay", "Writes trajectories")
    add_edge(r, "t2_e07", "t2_dist", "t2_core", "Uses network for self-play")
    add_edge(r, "t2_e08", "t2_replay", "t2_storage", "Persists to storage")
    add_edge(r, "t2_e09", "t2_cloud", "t2_storage", "Reads/writes GCS")
    add_edge(r, "t2_e10", "t2_cli", "t2_cloud", "Deploys worker jobs")
    add_edge(r, "t2_e11", "t2_eval", "t2_tourney", "Feeds match results")
    add_edge(r, "t2_e12", "t2_pbt", "t2_dist", "Manages agent population")
    add_edge(r, "t2_e13", "t2_web", "t2_core", "Loads trained models")
    add_edge(r, "t2_e14", "t2_web", "t2_envs", "Runs game sessions")
    add_edge(r, "t2_e15", "t2_hier", "t2_core", "Multi-level network heads")

    # ── External Edges ──
    add_edge(r, "t2_x01", "t2_rag", "t2_ext_llm", "LLM queries",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x02", "t2_cli", "t2_ext_vertex", "Submit jobs",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x03", "t2_storage", "t2_ext_gcs", "Object I/O",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x04", "t2_orch", "t2_ext_wandb", "Log metrics",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x05", "t2_envs", "t2_ext_games", "Game simulation",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x06", "t2_rag", "t2_ext_vectordb", "Vector search",
             STYLES["arrow_dashed"])
    add_edge(r, "t2_x07", "t2_utils", "t2_ext_llm", "LLM client calls",
             STYLES["arrow_dashed"])

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: C4 Level 3 — Core Algorithms Components
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab3():
    d, r = make_diagram("tab3", "C4 Level 3 - Core Algorithms")

    add_cell(r, "t3_title", (
        '<font style="font-size:20px"><b>Component Diagram: Core Algorithms</b></font>'
        '<br/><font style="font-size:12px">alpha_zero.core — Neural networks, MCTS, and action space</font>'
    ), STYLES["legend_title"], 20, 10, 600, 50)

    # ── Boundary ──
    add_cell(r, "t3_bound", "alpha_zero.core",
             STYLES["boundary"], 30, 70, 1100, 850)

    # ── Neural Network Group ──
    add_cell(r, "t3_nn_bound", "Neural Networks (civ6_network.py)",
             STYLES["boundary_sub"], 60, 110, 520, 370)

    add_cell(r, "t3_microciv_net", label(
        "MicroCivNetwork",
        "nn.Module",
        "13-ch spatial input, 6 ResBlocks,<br/>64 filters, policy + value heads"
    ), STYLES["component_key"], 80, 160, 230, 110)

    add_cell(r, "t3_civ6_net", label(
        "Civ6Network",
        "nn.Module",
        "84-ch spatial, multi-head attention,<br/>12 ResBlocks, 128 filters"
    ), STYLES["component_key"], 330, 160, 230, 110)

    add_cell(r, "t3_transformer", label(
        "TransformerNetwork",
        "nn.Module",
        "Transformer-based variant<br/>with positional encoding"
    ), STYLES["component"], 80, 300, 220, 90)

    add_cell(r, "t3_muzero", label(
        "MuZeroNetwork",
        "nn.Module",
        "Dynamics + representation<br/>+ prediction networks"
    ), STYLES["component"], 330, 300, 220, 90)

    add_cell(r, "t3_heads", label(
        "Network Heads",
        "PolicyHead / ValueHead",
        "Separate output heads<br/>with auxiliary losses"
    ), STYLES["component"], 80, 410, 220, 50)

    add_cell(r, "t3_losses", label(
        "Loss Functions",
        "losses.py",
        "Policy CE, value MSE,<br/>consistency, reward loss"
    ), STYLES["component"], 330, 410, 220, 50)

    # ── MCTS Group ──
    add_cell(r, "t3_mcts_bound", "MCTS Search (civ6_mcts.py)",
             STYLES["boundary_sub"], 60, 510, 520, 270)

    add_cell(r, "t3_civ6mcts", label(
        "Civ6MCTS",
        "PUCT Search",
        "Neural network-guided MCTS<br/>with Dirichlet noise, temperature<br/>scheduling, action masking"
    ), STYLES["component_key"], 80, 560, 230, 110)

    add_cell(r, "t3_gumbel", label(
        "GumbelMCTS",
        "Gumbel-Max Variant",
        "Policy improvement via<br/>sequential halving (MuZero)"
    ), STYLES["component"], 330, 560, 230, 90)

    add_cell(r, "t3_simple", label(
        "SimpleMCTS",
        "Random Rollout",
        "Baseline MCTS with<br/>random rollout policy"
    ), STYLES["component"], 80, 690, 220, 70)

    add_cell(r, "t3_batch", label(
        "BatchMCTS",
        "Parallel Search",
        "Batched neural network<br/>evaluation for speed"
    ), STYLES["component"], 330, 690, 220, 70)

    # ── Action Space Group ──
    add_cell(r, "t3_act_bound", "Action Space (civ6_actions.py)",
             STYLES["boundary_sub"], 620, 110, 480, 370)

    add_cell(r, "t3_action_space", label(
        "Civ6ActionSpace",
        "Hierarchical Actions",
        "9 categories: EndTurn, UnitAction,<br/>CityAction, Research, Diplomacy,<br/>Espionage, Government, GreatPerson"
    ), STYLES["component_key"], 640, 160, 250, 130)

    add_cell(r, "t3_action_enc", label(
        "Action Encoder/Decoder",
        "Int ↔ Structured Action",
        "Flat index to hierarchical<br/>action conversion (376 total)"
    ), STYLES["component"], 640, 310, 230, 90)

    add_cell(r, "t3_pruning", label(
        "Action Pruning",
        "action_pruning.py",
        "Invalid action masking<br/>based on game state"
    ), STYLES["component"], 640, 420, 220, 50)

    add_cell(r, "t3_action_cat", (
        '<font style="font-size:9px">'
        '<b>Action Categories:</b><br/>'
        '0: END_TURN<br/>'
        '1: UNIT_ACTION (select → type → target)<br/>'
        '2: CITY_ACTION (select → type → target)<br/>'
        '3: RESEARCH_TECH<br/>'
        '4: RESEARCH_CIVIC<br/>'
        '5: DIPLOMACY<br/>'
        '6: ESPIONAGE<br/>'
        '7: GOVERNMENT<br/>'
        '8: GREAT_PERSON'
        '</font>'
    ), STYLES["note"], 910, 160, 170, 200)

    # ── Reflection ──
    add_cell(r, "t3_reflect_bound", "Self-Reflection",
             STYLES["boundary_sub"], 620, 510, 480, 270)

    add_cell(r, "t3_reflect_gen", label(
        "ReflectionGenerator",
        "reflection_generator.py",
        "Generates training insights<br/>from game analysis"
    ), STYLES["component"], 640, 560, 220, 90)

    add_cell(r, "t3_reflect_mem", label(
        "ReflectionMemory",
        "reflection_memory.py",
        "Long-term storage for<br/>training reflections"
    ), STYLES["component"], 640, 680, 220, 70)

    # ── Internal edges ──
    add_edge(r, "t3_e01", "t3_civ6mcts", "t3_microciv_net",
             "Evaluates positions")
    add_edge(r, "t3_e02", "t3_civ6mcts", "t3_action_space",
             "Gets valid actions")
    add_edge(r, "t3_e03", "t3_gumbel", "t3_muzero",
             "Uses dynamics model")
    add_edge(r, "t3_e04", "t3_batch", "t3_civ6mcts",
             "Parallelizes search")
    add_edge(r, "t3_e05", "t3_pruning", "t3_action_space",
             "Masks invalid actions")
    add_edge(r, "t3_e06", "t3_losses", "t3_heads", "Computes gradients")
    add_edge(r, "t3_e07", "t3_reflect_gen", "t3_reflect_mem",
             "Stores insights")

    # ── Data flow note ──
    add_cell(r, "t3_note", (
        '<font style="font-size:10px">'
        '<b>Data Flow:</b><br/>'
        'State (13/84 ch) → Network → (policy, value)<br/>'
        'Policy → MCTS → action distribution<br/>'
        'MCTS visits → training target policy<br/>'
        'Game outcome → training target value'
        '</font>'
    ), STYLES["note"], 910, 520, 190, 110)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: C4 Level 3 — Training Pipeline Components
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab4():
    d, r = make_diagram("tab4", "C4 Level 3 - Training Pipeline")

    add_cell(r, "t4_title", (
        '<font style="font-size:20px"><b>Component Diagram: Training Pipeline</b></font>'
        '<br/><font style="font-size:12px">'
        'alpha_zero.distributed + alpha_zero.orchestrator</font>'
    ), STYLES["legend_title"], 20, 10, 600, 50)

    # ── Distributed Training Boundary ──
    add_cell(r, "t4_dist_bound", "alpha_zero.distributed",
             STYLES["boundary"], 30, 70, 700, 530)

    add_cell(r, "t4_actor", label(
        "Actor / ActorPool",
        "actor.py",
        "Self-play workers running<br/>MCTS games in parallel<br/>(multiprocessing)"
    ), STYLES["component_key"], 60, 130, 240, 110)

    add_cell(r, "t4_learner", label(
        "Learner",
        "learner.py",
        "Central gradient descent,<br/>samples replay buffer,<br/>publishes updated weights"
    ), STYLES["component_key"], 340, 130, 240, 110)

    add_cell(r, "t4_ddp", label(
        "DDP Learner",
        "ddp_learner.py",
        "Multi-GPU distributed<br/>data-parallel training"
    ), STYLES["component"], 340, 260, 220, 80)

    add_cell(r, "t4_shared", label(
        "SharedStorage",
        "shared_storage.py",
        "Thread-safe model weight<br/>container (latest weights)"
    ), STYLES["component_key"], 60, 270, 220, 90)

    add_cell(r, "t4_replay_dist", label(
        "DistributedReplayBuffer",
        "shared_storage.py",
        "Multiprocessing queue-based,<br/>stores GameTrajectory objects"
    ), STYLES["component_key"], 60, 390, 240, 90)

    add_cell(r, "t4_coord", label(
        "Coordinator",
        "coordinator.py",
        "Orchestrates actors/learner/<br/>checkpointing training loop"
    ), STYLES["component_key"], 340, 380, 240, 100)

    add_cell(r, "t4_param_server", label(
        "ParameterServer",
        "parameter_server.py",
        "Alternative weight distribution<br/>for large models"
    ), STYLES["component"], 340, 510, 220, 70)

    add_cell(r, "t4_cloud_coord", label(
        "CloudCoordinator",
        "cloud_coordinator.py",
        "Cloud-specific coordination<br/>(GCS, Vertex AI)"
    ), STYLES["component"], 60, 510, 220, 70)

    # ── Orchestrator Boundary ──
    add_cell(r, "t4_orch_bound", "alpha_zero.orchestrator",
             STYLES["boundary"], 30, 640, 700, 480)

    add_cell(r, "t4_training_orch", label(
        "TrainingOrchestrator",
        "orchestrator.py",
        "Unified training loop: self-play →<br/>training → evaluation cycle<br/>with dynamic configuration"
    ), STYLES["component_key"], 60, 700, 280, 110)

    add_cell(r, "t4_orch_config", label(
        "OrchestratorConfig",
        "config.py",
        "Hierarchical config with presets<br/>(fast/balanced/production),<br/>env overrides, schedules"
    ), STYLES["component"], 380, 700, 250, 110)

    add_cell(r, "t4_scheduler", label(
        "HyperparameterScheduler",
        "scheduler.py",
        "LR warmup_cosine, step decay,<br/>exponential, batch size scaling"
    ), STYLES["component"], 60, 840, 250, 90)

    add_cell(r, "t4_checkpoint", label(
        "CheckpointManager",
        "checkpointer.py",
        "Save/load model versions,<br/>best-model tracking"
    ), STYLES["component"], 350, 840, 230, 90)

    add_cell(r, "t4_rag_int", label(
        "RAGIntegration",
        "rag_integration.py",
        "Queries strategy knowledge<br/>base during training"
    ), STYLES["component"], 60, 960, 230, 80)

    add_cell(r, "t4_strategy", label(
        "StrategyAdvisor",
        "strategy_advisor.py",
        "LLM-based game analysis<br/>&amp; improvement suggestions"
    ), STYLES["component"], 330, 960, 230, 80)

    add_cell(r, "t4_main", label(
        "__main__.py",
        "CLI Entry Point",
        "argparse CLI: --preset, --config,<br/>--dry_run, --iterations, etc."
    ), STYLES["component"], 60, 1060, 220, 70)

    # ── Data Structures (right side) ──
    add_cell(r, "t4_ds_bound", "Key Data Structures",
             STYLES["boundary_sub"], 780, 70, 380, 540)

    add_cell(r, "t4_ds_example", (
        '<font style="font-size:10px">'
        '<b>TrainingExample</b><br/>'
        '─────────────────<br/>'
        'spatial: ndarray (C, H, W)<br/>'
        'vector: ndarray (D,)<br/>'
        'action_mask: ndarray (A,)<br/>'
        'policy: ndarray (A,) [MCTS visits]<br/>'
        'value: float [-1, 1] [game outcome]'
        '</font>'
    ), STYLES["note"], 800, 120, 220, 140)

    add_cell(r, "t4_ds_traj", (
        '<font style="font-size:10px">'
        '<b>GameTrajectory</b><br/>'
        '─────────────────<br/>'
        'examples: list[TrainingExample]<br/>'
        'game_id: int<br/>'
        'actor_id: int<br/>'
        'game_length: int<br/>'
        'winner: int | None'
        '</font>'
    ), STYLES["note"], 800, 280, 220, 130)

    add_cell(r, "t4_ds_config", (
        '<font style="font-size:10px">'
        '<b>OrchestratorConfig</b><br/>'
        '─────────────────<br/>'
        'EnvironmentConfig<br/>'
        'NetworkConfig<br/>'
        'MCTSConfig<br/>'
        'TrainingConfig (+ schedules)<br/>'
        'EvaluationConfig<br/>'
        'RAGConfig<br/>'
        'LoggingConfig'
        '</font>'
    ), STYLES["note"], 800, 430, 220, 150)

    # ── Training Flow (right side bottom) ──
    add_cell(r, "t4_flow_bound", "Training Loop Flow",
             STYLES["boundary_sub"], 780, 640, 380, 480)

    add_cell(r, "t4_flow1", label(
        "1. Self-Play",
        "",
        "Actors run MCTS games<br/>→ produce trajectories"
    ), STYLES["component"], 810, 690, 180, 80)

    add_cell(r, "t4_flow2", label(
        "2. Buffer",
        "",
        "Trajectories stored in<br/>replay buffer"
    ), STYLES["component"], 810, 790, 180, 70)

    add_cell(r, "t4_flow3", label(
        "3. Training",
        "",
        "Learner samples batches,<br/>computes loss, updates weights"
    ), STYLES["component"], 810, 880, 180, 80)

    add_cell(r, "t4_flow4", label(
        "4. Evaluation",
        "",
        "New model vs champion,<br/>promote if win_rate &gt; 55%"
    ), STYLES["component"], 810, 980, 180, 80)

    add_edge(r, "t4_fe1", "t4_flow1", "t4_flow2", "")
    add_edge(r, "t4_fe2", "t4_flow2", "t4_flow3", "")
    add_edge(r, "t4_fe3", "t4_flow3", "t4_flow4", "")
    add_edge(r, "t4_fe4", "t4_flow4", "t4_flow1", "Next iteration",
             STYLES["arrow_dashed"])

    # ── Internal Edges ──
    add_edge(r, "t4_e01", "t4_actor", "t4_shared", "Gets latest weights")
    add_edge(r, "t4_e02", "t4_actor", "t4_replay_dist",
             "Writes trajectories")
    add_edge(r, "t4_e03", "t4_learner", "t4_replay_dist",
             "Samples batches")
    add_edge(r, "t4_e04", "t4_learner", "t4_shared",
             "Publishes new weights")
    add_edge(r, "t4_e05", "t4_coord", "t4_actor", "Manages lifecycle")
    add_edge(r, "t4_e06", "t4_coord", "t4_learner", "Controls training")
    add_edge(r, "t4_e07", "t4_training_orch", "t4_coord",
             "Delegates distributed work")
    add_edge(r, "t4_e08", "t4_training_orch", "t4_scheduler",
             "Gets LR / batch size")
    add_edge(r, "t4_e09", "t4_training_orch", "t4_checkpoint",
             "Save/load models")
    add_edge(r, "t4_e10", "t4_training_orch", "t4_rag_int",
             "Queries strategies")
    add_edge(r, "t4_e11", "t4_rag_int", "t4_strategy",
             "Gets advisor suggestions")
    add_edge(r, "t4_e12", "t4_main", "t4_training_orch",
             "Creates &amp; runs")
    add_edge(r, "t4_e13", "t4_ddp", "t4_learner", "Extends")

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: C4 Level 3 — Cloud & Storage Components
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab5():
    d, r = make_diagram("tab5", "C4 Level 3 - Cloud &amp; Storage")

    add_cell(r, "t5_title", (
        '<font style="font-size:20px"><b>Component Diagram: Cloud &amp; Storage</b></font>'
        '<br/><font style="font-size:12px">'
        'alpha_zero.cloud + alpha_zero.storage — Vertex AI deployment and persistence</font>'
    ), STYLES["legend_title"], 20, 10, 700, 50)

    # ── Storage Boundary ──
    add_cell(r, "t5_store_bound", "alpha_zero.storage",
             STYLES["boundary"], 30, 70, 520, 520)

    add_cell(r, "t5_backend_abc", label(
        "StorageBackend",
        "ABC / Interface",
        "save_model, load_model,<br/>save_trajectory, load_trajectory,<br/>save_evaluation, load_evaluation"
    ), STYLES["component_key"], 60, 130, 240, 110)

    add_cell(r, "t5_local", label(
        "LocalBackend",
        "local_backend.py",
        "Filesystem storage for<br/>local development &amp; testing"
    ), STYLES["component"], 60, 280, 220, 90)

    add_cell(r, "t5_gcs", label(
        "GCSBackend",
        "gcs_backend.py",
        "Google Cloud Storage with<br/>streaming IterableDataset<br/>for memory-efficient loading"
    ), STYLES["component"], 300, 280, 220, 90)

    add_cell(r, "t5_factory", label(
        "create_storage_backend",
        "factory.py",
        "Instantiates Local or GCS<br/>based on StorageConfig"
    ), STYLES["component"], 60, 400, 220, 80)

    add_cell(r, "t5_store_config", label(
        "StorageConfig",
        "config.py",
        "backend_type, bucket, prefix,<br/>for_local(), for_gcs()"
    ), STYLES["component"], 300, 400, 220, 80)

    add_cell(r, "t5_models", label(
        "Metadata Models",
        "models.py",
        "ModelMetadata, CheckpointMetadata,<br/>TrajectoryMetadata, EvaluationResult"
    ), STYLES["component"], 60, 500, 260, 70)

    add_cell(r, "t5_stream_replay", label(
        "Streaming ReplayBuffer",
        "replay_buffer.py",
        "GCS-backed IterableDataset<br/>for cloud training"
    ), STYLES["component"], 340, 500, 190, 70)

    # ── Cloud Boundary ──
    add_cell(r, "t5_cloud_bound", "alpha_zero.cloud",
             STYLES["boundary"], 30, 630, 1100, 530)

    # Cloud Config sub-group
    add_cell(r, "t5_cloud_config", label(
        "CloudConfig / GCPConfig",
        "config.py",
        "Project, region, bucket,<br/>machine profiles"
    ), STYLES["component"], 60, 700, 230, 90)

    add_cell(r, "t5_machine", (
        '<font style="font-size:9px">'
        '<b>Machine Profiles</b><br/>'
        '─────────────<br/>'
        'selfplay_t4_spot: $0.11/hr<br/>'
        'selfplay_l4_spot: $0.24/hr<br/>'
        'train_l4_single: $0.80/hr<br/>'
        'train_l4_ondemand: $2.68/hr<br/>'
        'train_a100_burst: $3.67/hr<br/>'
        'eval_l4_spot: $0.24/hr'
        '</font>'
    ), STYLES["note"], 60, 810, 180, 150)

    # Workers sub-group
    add_cell(r, "t5_workers_bound", "Cloud Workers (Containerized)",
             STYLES["boundary_sub"], 310, 680, 500, 200)

    add_cell(r, "t5_selfplay_w", label(
        "SelfPlayWorker",
        "selfplay_worker.py",
        "Runs N games with MCTS,<br/>saves trajectories to GCS"
    ), STYLES["component_key"], 330, 730, 220, 90)

    add_cell(r, "t5_trainer_w", label(
        "TrainingWorker",
        "training_worker.py",
        "Loads trajectories, trains<br/>for N steps, saves model"
    ), STYLES["component_key"], 570, 730, 220, 90)

    add_cell(r, "t5_eval_w", label(
        "EvaluationWorker",
        "evaluation_worker.py",
        "Model vs champion,<br/>reports win rate"
    ), STYLES["component"], 440, 840, 220, 30)

    # Vertex AI sub-group
    add_cell(r, "t5_vertex_bound", "Vertex AI Integration (cloud.gcp)",
             STYLES["boundary_sub"], 310, 910, 500, 230)

    add_cell(r, "t5_vertex_client", label(
        "VertexAIClient",
        "vertex.py",
        "Job submission &amp; monitoring<br/>via Vertex AI SDK"
    ), STYLES["component_key"], 330, 960, 220, 90)

    add_cell(r, "t5_vertex_mgr", label(
        "VertexAIManager",
        "vertex.py",
        "Multi-job orchestration,<br/>wait-for-completion"
    ), STYLES["component"], 570, 960, 220, 80)

    add_cell(r, "t5_vertex_jobs", label(
        "VertexJobs",
        "vertex_jobs.py",
        "SelfPlay / Training / Eval<br/>job spec wrappers"
    ), STYLES["component"], 330, 1070, 220, 60)

    add_cell(r, "t5_vertex_orch", label(
        "VertexOrchestrator",
        "vertex_orchestrator.py",
        "Full iteration loop:<br/>selfplay → train → eval"
    ), STYLES["component_key"], 570, 1060, 220, 70)

    # Resilience sub-group
    add_cell(r, "t5_resil_bound", "Resilience",
             STYLES["boundary_sub"], 840, 680, 270, 270)

    add_cell(r, "t5_preempt", label(
        "PreemptionHandler",
        "preemption.py",
        "SIGTERM handler for Spot VMs,<br/>flushes state in 30s grace"
    ), STYLES["component"], 860, 730, 230, 90)

    add_cell(r, "t5_health", label(
        "HealthCheck",
        "container_healthcheck.py",
        "Liveness &amp; readiness probes"
    ), STYLES["component"], 860, 840, 220, 60)

    add_cell(r, "t5_idle", label(
        "IdleManager",
        "idle_manager.py",
        "Idle resource cleanup"
    ), STYLES["component"], 860, 910, 200, 30)

    # Monitoring sub-group
    add_cell(r, "t5_monitor_bound", "Monitoring &amp; Cost",
             STYLES["boundary_sub"], 840, 970, 270, 180)

    add_cell(r, "t5_monitor", label(
        "CloudMonitoring",
        "monitoring.py",
        "Structured logging, metrics"
    ), STYLES["component"], 860, 1010, 220, 60)

    add_cell(r, "t5_cost", label(
        "CostMonitor",
        "cost_monitor.py",
        "Budget tracking, alerts"
    ), STYLES["component"], 860, 1090, 220, 50)

    # ── External Systems ──
    ext_x = 600
    add_cell(r, "t5_ext_gcs", label(
        "Google Cloud Storage",
        "External"
    ), STYLES["system_external"], ext_x, 70, 200, 70)

    add_cell(r, "t5_ext_vertex", label(
        "Vertex AI Platform",
        "External"
    ), STYLES["system_external"], ext_x + 240, 70, 200, 70)

    # Docker images note
    add_cell(r, "t5_docker", (
        '<font style="font-size:10px">'
        '<b>Docker Images</b><br/>'
        '─────────────<br/>'
        'Dockerfile.base (PyTorch)<br/>'
        'Dockerfile.selfplay<br/>'
        'Dockerfile.trainer<br/>'
        'Dockerfile.evaluator<br/><br/>'
        'Pushed to GCP Artifact Registry'
        '</font>'
    ), STYLES["note"], 1140, 700, 170, 150)

    # ── Edges ──
    add_edge(r, "t5_e01", "t5_factory", "t5_local", "Creates")
    add_edge(r, "t5_e02", "t5_factory", "t5_gcs", "Creates")
    add_edge(r, "t5_e03", "t5_local", "t5_backend_abc", "Implements")
    add_edge(r, "t5_e04", "t5_gcs", "t5_backend_abc", "Implements")
    add_edge(r, "t5_e05", "t5_gcs", "t5_ext_gcs", "Object I/O",
             STYLES["arrow_dashed"])
    add_edge(r, "t5_e06", "t5_selfplay_w", "t5_backend_abc",
             "Writes trajectories")
    add_edge(r, "t5_e07", "t5_trainer_w", "t5_backend_abc",
             "Reads trajectories,<br/>writes models")
    add_edge(r, "t5_e08", "t5_vertex_client", "t5_ext_vertex",
             "Submits jobs", STYLES["arrow_dashed"])
    add_edge(r, "t5_e09", "t5_vertex_orch", "t5_vertex_client",
             "Uses for job control")
    add_edge(r, "t5_e10", "t5_vertex_orch", "t5_vertex_jobs",
             "Creates job specs")
    add_edge(r, "t5_e11", "t5_preempt", "t5_selfplay_w",
             "Graceful shutdown")
    add_edge(r, "t5_e12", "t5_stream_replay", "t5_gcs",
             "Streams from GCS")

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: C4 Level 3 — RAG, Tournament, PBT, Hierarchical
# ═══════════════════════════════════════════════════════════════════════════════
def build_tab6():
    d, r = make_diagram("tab6", "C4 Level 3 - Intelligence &amp; Tuning")

    add_cell(r, "t6_title", (
        '<font style="font-size:20px"><b>Component Diagram: Intelligence &amp; Tuning</b></font>'
        '<br/><font style="font-size:12px">'
        'RAG System, Tournament, PBT, and Hierarchical Decision Making</font>'
    ), STYLES["legend_title"], 20, 10, 700, 50)

    # ── RAG System ──
    add_cell(r, "t6_rag_bound", "alpha_zero.rag",
             STYLES["boundary"], 30, 70, 560, 500)

    add_cell(r, "t6_pipeline", label(
        "RAGPipeline",
        "pipeline.py",
        "Main orchestrator: ingest<br/>documents, query with context,<br/>async interface"
    ), STYLES["component_key"], 60, 130, 240, 100)

    add_cell(r, "t6_rag_config", label(
        "RAGConfig",
        "config/settings.py",
        "Vector store type, embedding<br/>model, chunk size, top_k"
    ), STYLES["component"], 330, 130, 230, 80)

    # Vector stores
    add_cell(r, "t6_vs_bound", "Vector Stores",
             STYLES["boundary_sub"], 50, 260, 260, 200)

    add_cell(r, "t6_vs_mem", label(
        "MemoryStore",
        "In-memory",
        "Dev/test, no persistence"
    ), STYLES["component"], 65, 300, 120, 60)

    add_cell(r, "t6_vs_chroma", label(
        "ChromaStore",
        "ChromaDB",
        "Persistent vector DB"
    ), STYLES["component"], 65, 370, 120, 60)

    add_cell(r, "t6_vs_faiss", label(
        "FAISSStore",
        "FAISS",
        "High-perf search"
    ), STYLES["component"], 190, 370, 110, 60)

    # Processors
    add_cell(r, "t6_proc_bound", "Processors",
             STYLES["boundary_sub"], 330, 240, 240, 150)

    add_cell(r, "t6_text_proc", label(
        "TextProcessor",
        "Chunking &amp; cleanup"
    ), STYLES["component"], 345, 280, 100, 50)

    add_cell(r, "t6_embed", label(
        "EmbeddingGenerator",
        "OpenAI / local"
    ), STYLES["component"], 455, 280, 100, 50)

    add_cell(r, "t6_proc_base", label("BaseProcessor", "Interface"),
             STYLES["component"], 380, 350, 120, 30)

    # Agents
    add_cell(r, "t6_agent_bound", "Agents / Workflow",
             STYLES["boundary_sub"], 50, 480, 520, 70)

    add_cell(r, "t6_query_enh", label("QueryEnhancer", ""),
             STYLES["component"], 65, 495, 130, 40)

    add_cell(r, "t6_ctx_build", label("ContextBuilder", ""),
             STYLES["component"], 210, 495, 130, 40)

    add_cell(r, "t6_workflow", label("RAGWorkflow", "Multi-step"),
             STYLES["component"], 360, 495, 140, 40)

    # ── Tournament System ──
    add_cell(r, "t6_tourney_bound", "alpha_zero.tournament",
             STYLES["boundary"], 620, 70, 520, 290)

    add_cell(r, "t6_tourney_orch", label(
        "TournamentOrchestrator",
        "orchestrator.py",
        "Bracket generation, matchmaking,<br/>result aggregation"
    ), STYLES["component_key"], 640, 130, 240, 90)

    add_cell(r, "t6_elo", label(
        "ELORatingSystem",
        "elo.py",
        "Rating calculations,<br/>history tracking"
    ), STYLES["component"], 900, 130, 200, 80)

    add_cell(r, "t6_matchmaker", label(
        "MatchMaker",
        "matchmaking.py",
        "Opponent selection<br/>strategies"
    ), STYLES["component"], 640, 240, 180, 70)

    add_cell(r, "t6_league", label(
        "LeagueManager",
        "league.py",
        "Multi-season league<br/>coordination"
    ), STYLES["component"], 840, 240, 180, 70)

    add_cell(r, "t6_tourney_analysis", label(
        "TournamentAnalysis",
        "analysis.py"
    ), STYLES["component"], 1040, 250, 80, 50)

    add_cell(r, "t6_format_note", (
        '<font style="font-size:9px">'
        '<b>Tournament Formats</b><br/>'
        '• Round-Robin<br/>'
        '• Single Elimination<br/>'
        '• Swiss System'
        '</font>'
    ), STYLES["note"], 1040, 130, 90, 90)

    # ── PBT System ──
    add_cell(r, "t6_pbt_bound", "alpha_zero.pbt",
             STYLES["boundary"], 620, 390, 520, 280)

    add_cell(r, "t6_pbt_coord", label(
        "PBTCoordinator",
        "coordinator.py",
        "Population-based training<br/>loop: train → evaluate →<br/>exploit → explore"
    ), STYLES["component_key"], 640, 450, 240, 100)

    add_cell(r, "t6_population", label(
        "PopulationManager",
        "population.py",
        "Agent lifecycle,<br/>fitness tracking"
    ), STYLES["component"], 900, 440, 200, 80)

    add_cell(r, "t6_mutations", label(
        "MutationOperator",
        "mutations.py",
        "Hyperparameter perturbation<br/>(LR, temp, exploration)"
    ), STYLES["component"], 640, 570, 220, 80)

    add_cell(r, "t6_pbt_sched", label(
        "PBTScheduler",
        "scheduler.py",
        "When to exploit/explore"
    ), STYLES["component"], 900, 540, 180, 60)

    add_cell(r, "t6_pbt_eval", label(
        "PBTEvaluator",
        "evaluator.py",
        "Fitness evaluation"
    ), STYLES["component"], 900, 620, 170, 40)

    # ── Hierarchical Decision Making ──
    add_cell(r, "t6_hier_bound", "alpha_zero.hierarchical",
             STYLES["boundary"], 30, 600, 560, 400)

    add_cell(r, "t6_hier_ctrl", label(
        "HierarchicalController",
        "controller.py",
        "Coordinates strategic, tactical,<br/>&amp; operational decision levels"
    ), STYLES["component_key"], 60, 660, 260, 90)

    add_cell(r, "t6_strategic", label(
        "StrategicController",
        "Level: 50+ turn horizon",
        "High-level goals:<br/>expansion, tech, military"
    ), STYLES["component"], 60, 780, 160, 90)

    add_cell(r, "t6_tactical", label(
        "TacticalController",
        "Level: 10-20 turns",
        "Medium-term tactics:<br/>army positioning, builds"
    ), STYLES["component"], 240, 780, 160, 90)

    add_cell(r, "t6_operational", label(
        "OperationalController",
        "Level: 1-5 turns",
        "Immediate actions:<br/>unit moves, production"
    ), STYLES["component"], 420, 780, 150, 90)

    # Goals sub-group
    add_cell(r, "t6_goals_bound", "Goal System",
             STYLES["boundary_sub"], 50, 890, 280, 90)

    add_cell(r, "t6_goal_hier", label("GoalHierarchy", "Goal tree"),
             STYLES["component"], 60, 915, 120, 40)

    add_cell(r, "t6_goal_eval", label("GoalEvaluator", "Achievement"),
             STYLES["component"], 195, 915, 120, 40)

    add_cell(r, "t6_hier_net", label(
        "HierarchicalNetwork",
        "Multi-head NN",
        "Shared backbone +<br/>level-specific heads"
    ), STYLES["component"], 360, 660, 200, 80)

    add_cell(r, "t6_training_int", label(
        "TrainingIntegration",
        "training_integration.py",
        "Hierarchical training pipeline"
    ), STYLES["component"], 360, 910, 200, 60)

    # ── External ──
    add_cell(r, "t6_ext_llm", label("LLM APIs", "OpenAI / Anthropic"),
             STYLES["system_external"], 220, 20, 180, 40)

    add_cell(r, "t6_ext_wandb", label("W&amp;B", "Metrics"),
             STYLES["system_external"], 1000, 20, 120, 40)

    # ── Edges ──
    add_edge(r, "t6_e01", "t6_pipeline", "t6_vs_chroma", "Stores vectors")
    add_edge(r, "t6_e02", "t6_pipeline", "t6_text_proc", "Processes docs")
    add_edge(r, "t6_e03", "t6_text_proc", "t6_embed", "Generates embeddings")
    add_edge(r, "t6_e04", "t6_pipeline", "t6_query_enh", "Enhances queries")
    add_edge(r, "t6_e05", "t6_query_enh", "t6_ctx_build", "Builds context")
    add_edge(r, "t6_e06", "t6_embed", "t6_ext_llm", "Embedding API",
             STYLES["arrow_dashed"])
    add_edge(r, "t6_e07", "t6_tourney_orch", "t6_elo", "Updates ratings")
    add_edge(r, "t6_e08", "t6_tourney_orch", "t6_matchmaker",
             "Gets matchups")
    add_edge(r, "t6_e09", "t6_pbt_coord", "t6_population",
             "Manages agents")
    add_edge(r, "t6_e10", "t6_pbt_coord", "t6_mutations",
             "Mutates hyperparams")
    add_edge(r, "t6_e11", "t6_hier_ctrl", "t6_strategic", "Delegates")
    add_edge(r, "t6_e12", "t6_hier_ctrl", "t6_tactical", "Delegates")
    add_edge(r, "t6_e13", "t6_hier_ctrl", "t6_operational", "Delegates")
    add_edge(r, "t6_e14", "t6_hier_ctrl", "t6_hier_net", "Uses")
    add_edge(r, "t6_e15", "t6_tourney_orch", "t6_ext_wandb",
             "Logs ELO", STYLES["arrow_dashed"])

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Assemble all tabs into a single .drawio file
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    mxfile = ET.Element("mxfile")
    mxfile.set("host", "app.diagrams.net")
    mxfile.set("modified", "2026-02-17T00:00:00.000Z")
    mxfile.set("agent", "Civ6-AlphaZero C4 Generator")
    mxfile.set("version", "24.0.0")
    mxfile.set("type", "device")

    tabs = [
        build_tab1(),  # System Context
        build_tab2(),  # Container Diagram
        build_tab3(),  # Core Algorithms Components
        build_tab4(),  # Training Pipeline Components
        build_tab5(),  # Cloud & Storage Components
        build_tab6(),  # Intelligence & Tuning Components
    ]

    for tab in tabs:
        mxfile.append(tab)

    # Pretty print
    rough = ET.tostring(mxfile, encoding="unicode")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="  ", encoding=None)

    # Remove the XML declaration that minidom adds
    lines = pretty.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    output = "\n".join(lines)

    with open("docs/c4-architecture.drawio", "w") as f:
        f.write(output)

    print(f"Generated docs/c4-architecture.drawio")
    print(f"  Tabs: {len(tabs)}")
    for tab in tabs:
        print(f"    - {tab.get('name')}")


if __name__ == "__main__":
    main()
