"""MicroCiv Explorer — Interactive Gradio Space for Civ6 AlphaZero.

Loads a published model from Hugging Face Hub and lets users explore
board positions, visualise attention maps, and play against the AI.

NOTE: This is scaffolding — the full interactive gameplay requires
the Civ6 environment which is not yet packaged for Spaces.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from alpha_zero.hub.constants import HF_ORG
except ImportError:
    # Standalone Space context: alpha_zero package not installed.
    HF_ORG = os.environ.get("HF_ORG", "ianshank")

MODEL_REPO_ID = os.environ.get(
    "HF_MODEL_REPO", f"{HF_ORG}/civ6-transformer",
)
TITLE = "🏛️ Civ6 AlphaZero — MicroCiv Explorer"
DESCRIPTION = """
Explore the Civ6 AlphaZero policy/value network.

- **Model**: Loaded from Hugging Face Hub
- **Features**: Board visualization, policy distribution, value estimation
"""


def create_app() -> Any:
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

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Board State")
                board_display = gr.Textbox(
                    label="Board (JSON)",
                    placeholder="Paste a board state JSON or use the example...",
                    lines=10,
                )
                analyze_btn = gr.Button("🔍 Analyze Position", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Model Output")
                value_output = gr.Number(label="Position Value", precision=4)
                policy_output = gr.Textbox(label="Top 5 Actions", lines=5)
                model_info = gr.Textbox(label="Model Info", lines=3)

        def analyze_position(board_json: str) -> tuple[float, str, str]:
            """Analyze a board position (placeholder)."""
            return (
                0.0,
                "Model not loaded — this is a Space scaffold.\n"
                f"Would load from: {MODEL_REPO_ID}",
                f"Repo: {MODEL_REPO_ID}\nStatus: Scaffold only",
            )

        analyze_btn.click(
            fn=analyze_position,
            inputs=[board_display],
            outputs=[value_output, policy_output, model_info],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
