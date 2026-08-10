"""ELO Dashboard — Interactive Gradio Space for Civ6 AlphaZero ratings.

Displays ELO tournament results, win-rate matrices, and model
progression charts loaded from a published HF dataset.

NOTE: This is scaffolding — full charts require tournament data.
"""

from __future__ import annotations

import os

try:
    from alpha_zero.hub.constants import HF_ORG
except ImportError:
    # Standalone Space context: alpha_zero package not installed.
    HF_ORG = os.environ.get("HF_ORG", "Mango-Metrics-NLM")

DATASET_REPO_ID = os.environ.get(
    "HF_DATASET_REPO", f"{HF_ORG}/civ6-elo-tournament",
)
TITLE = "📊 Civ6 AlphaZero — ELO Dashboard"
DESCRIPTION = """
Track model improvement through ELO ratings and tournament results.

- **Dataset**: Loaded from Hugging Face Hub
- **Features**: Rating progression, win-rate matrix, head-to-head comparison
"""


def create_app():
    """Build and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "This Space requires Gradio. Install with: pip install gradio>=4.20.0"
        ) from exc

    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Tab("📈 Rating Progression"):
            gr.Markdown("### ELO Rating Over Training Iterations")
            gr.Textbox(
                label="Status",
                value=f"Scaffold — would load from: {DATASET_REPO_ID}",
                interactive=False,
            )

        with gr.Tab("🏆 Win-Rate Matrix"):
            gr.Markdown("### Head-to-Head Win Rates")
            gr.Textbox(
                label="Status",
                value="Tournament data not yet loaded.",
                interactive=False,
            )

        with gr.Tab("🔬 Model Comparison"):
            gr.Markdown("### Compare Two Model Versions")
            with gr.Row():
                gr.Dropdown(
                    label="Model A", choices=["v1", "v2", "v3"],
                    value="v1", interactive=True,
                )
                gr.Dropdown(
                    label="Model B", choices=["v1", "v2", "v3"],
                    value="v2", interactive=True,
                )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
