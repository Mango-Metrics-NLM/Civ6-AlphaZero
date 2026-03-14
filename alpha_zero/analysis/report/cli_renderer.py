"""CLI renderer for training run analysis reports.

Pure string formatting — no external dependencies. Renders analysis engine
results as formatted ASCII text suitable for terminal output.
"""

from __future__ import annotations

from typing import Any

from alpha_zero.analysis.data_model import RunData


class CLIRenderer:
    """Renders analysis results as formatted CLI text."""

    def render(self, results: dict[str, Any], run_data: RunData) -> str:
        """Render all analysis results."""
        sections = []
        sections.append(self._render_header(run_data))

        if "loss_analysis" in results:
            sections.append(self._render_loss_section(results["loss_analysis"]))
        if "model_progression" in results:
            sections.append(self._render_model_section(results["model_progression"]))
        if "cost_analysis" in results:
            sections.append(self._render_cost_section(results["cost_analysis"]))
        if "diagnostics" in results:
            sections.append(self._render_diagnostics_section(results["diagnostics"]))
        if "selfplay_quality" in results:
            sections.append(self._render_selfplay_section(results["selfplay_quality"]))

        return "\n\n".join(sections)

    def _render_header(self, run_data: RunData) -> str:
        """Render run identification header."""
        lines = [self._render_section_title("TRAINING RUN ANALYSIS")]
        ri = run_data.identifier
        if ri.wandb_run_id:
            lines.append(f"  Run ID:     {ri.wandb_run_id}")
        if ri.wandb_group:
            lines.append(f"  Group:      {ri.wandb_group}")
        lines.append(f"  Project:    {ri.wandb_project}")
        lines.append(f"  Iterations: {run_data.total_iterations}")
        lines.append(f"  Promotions: {run_data.promotions}")
        return "\n".join(lines)

    @staticmethod
    def _render_section_title(title: str) -> str:
        return f"=== {title} ==="

    def _render_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Render aligned ASCII table."""
        if not rows:
            return "  (no data)"
        col_widths = [
            max(len(h), max((len(r[i]) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        header_line = " | ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        sep_line = "-+-".join("-" * w for w in col_widths)
        row_lines = [
            " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            for row in rows
        ]
        return "\n".join(
            ["  " + header_line, "  " + sep_line] + ["  " + r for r in row_lines]
        )

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_loss_section(self, result: Any) -> str:
        """Render loss analysis section."""
        d = result.to_dict() if hasattr(result, "to_dict") else result
        lines = [self._render_section_title("LOSS ANALYSIS")]

        pi = d.get("policy_loss_initial")
        pf = d.get("policy_loss_final")
        vi = d.get("value_loss_initial")
        vf = d.get("value_loss_final")

        if pi is not None and pf is not None:
            lines.append(f"  Policy loss:  {pi:.4f} -> {pf:.4f}")
        if vi is not None and vf is not None:
            lines.append(f"  Value loss:   {vi:.4f} -> {vf:.4f}")

        conv = d.get("policy_convergence_rate", 0.0)
        lines.append(f"  Convergence:  {conv:.6f}/iter")

        if d.get("policy_stalled") or d.get("value_stalled"):
            stalls = []
            if d.get("policy_stalled"):
                stalls.append("policy")
            if d.get("value_stalled"):
                stalls.append("value")
            stall_iter = d.get("stall_iteration")
            stall_msg = f" (from iter {stall_iter})" if stall_iter is not None else ""
            lines.append(f"  STALLED: {', '.join(stalls)}{stall_msg}")

        pf_total = d.get("policy_fraction_of_total")
        vf_total = d.get("value_fraction_of_total")
        if pf_total is not None and vf_total is not None:
            lines.append(
                f"  Decomposition: policy={pf_total:.1%}, value={vf_total:.1%}"
            )

        return "\n".join(lines)

    def _render_model_section(self, result: Any) -> str:
        """Render model progression section."""
        d = result.to_dict() if hasattr(result, "to_dict") else result
        lines = [self._render_section_title("MODEL PROGRESSION")]

        lines.append(f"  Promotions: {d.get('total_promotions', 0)}")
        rate = d.get("promotion_rate", 0.0)
        lines.append(f"  Rate:       {rate:.1%}")

        chain = d.get("promotion_chain", [])
        if chain:
            steps = [
                f"v{p['model_version']} (iter {p['iteration']})" for p in chain
            ]
            lines.append(f"  Chain:      {' -> '.join(steps)}")

        if d.get("training_from_scratch_detected"):
            lines.append("  WARNING: Training-from-scratch bug detected!")

        diagnosis = d.get("diagnosis", "")
        if diagnosis:
            lines.append(f"  Diagnosis:  {diagnosis}")

        return "\n".join(lines)

    def _render_cost_section(self, result: Any) -> str:
        """Render cost analysis section."""
        d = result.to_dict() if hasattr(result, "to_dict") else result
        lines = [self._render_section_title("COST ANALYSIS")]

        total = d.get("total_cost", 0.0)
        lines.append(f"  Total cost:     ${total:.2f}")

        breakdown = d.get("cost_breakdown_pct", {})
        if breakdown:
            lines.append(
                f"  Breakdown:      selfplay={breakdown.get('selfplay', 0):.1f}%, "
                f"training={breakdown.get('training', 0):.1f}%, "
                f"eval={breakdown.get('eval', 0):.1f}%"
            )

        cpi = d.get("cost_per_iteration", 0.0)
        lines.append(f"  Per iteration:  ${cpi:.2f}")

        cpp = d.get("cost_per_promotion")
        if cpp is not None:
            lines.append(f"  Per promotion:  ${cpp:.2f}")
        else:
            lines.append("  Per promotion:  N/A (0 promotions)")

        cpe = d.get("cost_per_elo_point")
        if cpe is not None:
            lines.append(f"  Per ELO point:  ${cpe:.2f}")

        return "\n".join(lines)

    def _render_diagnostics_section(self, result: Any) -> str:
        """Render diagnostics section."""
        d = result.to_dict() if hasattr(result, "to_dict") else result
        lines = [self._render_section_title("DIAGNOSTICS")]

        vc_min = d.get("value_correlation_min")
        vc_max = d.get("value_correlation_max")
        vc_std = d.get("value_correlation_std")
        if vc_min is not None and vc_max is not None:
            lines.append(
                f"  Value head:   range=[{vc_min:.3f}, {vc_max:.3f}], std={vc_std:.3f}"
            )
            if d.get("value_head_unstable"):
                lines.append("  Value head:   UNSTABLE")

        wr_trend = d.get("win_rate_trend", 0.0)
        lines.append(f"  Win rate trend: {wr_trend:.4f}/iter")
        if d.get("win_rate_stalled"):
            lines.append("  Win rate:     STALLED")
        if d.get("loss_stalled"):
            lines.append("  Loss:         STALLED")

        diagnosis = d.get("diagnosis", "")
        if diagnosis:
            for line in diagnosis.split("\n"):
                lines.append(f"  {line}")

        return "\n".join(lines)

    def _render_selfplay_section(self, result: Any) -> str:
        """Render selfplay quality section."""
        d = result.to_dict() if hasattr(result, "to_dict") else result
        lines = [self._render_section_title("SELFPLAY QUALITY")]

        lines.append(f"  Total games:       {d.get('total_games', 0)}")
        avg = d.get("avg_games_per_iteration", 0.0)
        lines.append(f"  Avg/iteration:     {avg:.1f}")
        lines.append(f"  Total trajectories: {d.get('total_trajectories', 0)}")

        depth = d.get("avg_mcts_depth")
        visits = d.get("avg_mcts_visits")
        if depth is not None:
            lines.append(f"  Avg MCTS depth:    {depth:.1f}")
        if visits is not None:
            lines.append(f"  Avg MCTS visits:   {visits:.1f}")

        return "\n".join(lines)
