"""
PulseAgent — Output Formatters
Team HOKM · DSN × BCT Hackathon 3.0

Three renderers for agent output:
  - terminal: Rich-formatted coloured output
  - json:     Clean JSON serialisation
  - markdown: Publication-ready Markdown brief
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Terminal renderer (Rich)
# ---------------------------------------------------------------------------

def render_terminal_task_a(result: Dict[str, Any]) -> None:
    """Render Task A (simulated review) output to terminal using Rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    rating = result.get("simulated_rating", "—")
    quality = result.get("review_quality_score", 0.0)
    review = result.get("simulated_review", "")
    trace = result.get("reasoning_trace", [])
    errors = result.get("errors", [])

    # Header panel
    console.print(Panel.fit(
        f"[bold green]★ {rating}[/bold green]  "
        f"Quality: [cyan]{quality:.2f}[/cyan]",
        title="[bold]PulseAgent — Simulated Review[/bold]",
        border_style="green",
    ))
    console.print()

    # Review text
    if review:
        console.print(Panel(
            f"[italic]{review}[/italic]",
            title="Generated Review",
            border_style="dim",
        ))
    else:
        console.print("[yellow]No review text generated.[/yellow]")

    console.print()

    # Reasoning trace
    if trace:
        table = Table(title="Reasoning Trace", show_lines=True, border_style="dim")
        table.add_column("Step", style="dim", width=6, justify="right")
        table.add_column("Detail")
        for i, step in enumerate(trace, 1):
            table.add_row(str(i), step)
        console.print(table)

    # Errors
    if errors:
        console.print()
        for err in errors:
            console.print(f"[yellow]⚠ {err}[/yellow]")


def render_terminal_task_b(result: Dict[str, Any]) -> None:
    """Render Task B (recommendations) output to terminal using Rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    intent = result.get("inferred_intent", "—")
    cold_start = result.get("cold_start", False)
    recs = result.get("ranked_recommendations", [])
    explanation = result.get("explanation", "")
    trace = result.get("reasoning_trace", [])
    errors = result.get("errors", [])

    # Header panel
    cold_label = "[yellow]Yes[/yellow]" if cold_start else "[green]No[/green]"
    console.print(Panel.fit(
        f"Intent: [cyan]{intent}[/cyan]  Cold-start: {cold_label}",
        title="[bold]PulseAgent — Recommendations[/bold]",
        border_style="blue",
    ))
    console.print()

    # Explanation
    if explanation:
        console.print(Panel(
            explanation,
            title="Why These Recommendations",
            border_style="dim",
        ))
        console.print()

    # Recommendations table
    if not recs:
        console.print("[red]No recommendations generated.[/red]")
        return

    table = Table(title="Top Recommendations", show_lines=True, border_style="blue")
    table.add_column("#", width=4, justify="right")
    table.add_column("Item", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Predicted ★", justify="center")
    table.add_column("NDCG", justify="center", style="dim")
    table.add_column("Explanation")

    for rec in recs:
        ndcg = rec.get("ndcg_score")
        ndcg_str = f"{ndcg:.3f}" if ndcg is not None else "—"
        predicted = rec.get("predicted_rating", 0)

        # Colour-code the star rating
        if predicted >= 4.5:
            rating_str = f"[green]{predicted:.1f}[/green]"
        elif predicted >= 3.5:
            rating_str = f"[yellow]{predicted:.1f}[/yellow]"
        else:
            rating_str = f"[red]{predicted:.1f}[/red]"

        table.add_row(
            str(rec.get("rank", "")),
            rec.get("name", ""),
            rec.get("category", ""),
            rating_str,
            ndcg_str,
            rec.get("explanation", ""),
        )

    console.print(table)

    # Errors
    if errors:
        console.print()
        for err in errors:
            console.print(f"[yellow]⚠ {err}[/yellow]")


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert Pydantic models and non-serialisable types to dicts."""
    if hasattr(obj, "model_dump"):
        return _make_serialisable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(i) for i in obj]
    return obj


def format_task_a_json(result: Dict[str, Any]) -> str:
    """Serialise Task A result to a clean JSON string."""
    output = {
        "simulated_rating":     result.get("simulated_rating"),
        "simulated_review":     result.get("simulated_review"),
        "confidence":           result.get("review_quality_score"),
        "reasoning_trace":      result.get("reasoning_trace", []),
        "errors":               result.get("errors", []),
    }
    return json.dumps(_make_serialisable(output), indent=2)


def format_task_b_json(result: Dict[str, Any]) -> str:
    """Serialise Task B result to a clean JSON string."""
    recs = result.get("ranked_recommendations", [])

    output = {
        "recommendations": [
            {
                "rank":             r.get("rank") if isinstance(r, dict) else r.rank,
                "item_id":          r.get("item_id") if isinstance(r, dict) else r.item_id,
                "name":             r.get("name") if isinstance(r, dict) else r.name,
                "category":         r.get("category") if isinstance(r, dict) else r.category,
                "predicted_rating": r.get("predicted_rating") if isinstance(r, dict) else r.predicted_rating,
                "explanation":      r.get("explanation") if isinstance(r, dict) else r.explanation,
                "ndcg_score":       r.get("ndcg_score") if isinstance(r, dict) else r.ndcg_score,
            }
            for r in recs
        ],
        "inferred_intent":  result.get("inferred_intent"),
        "cold_start":       result.get("cold_start", False),
        "explanation":      result.get("explanation"),
        "reasoning_trace":  result.get("reasoning_trace", []),
        "errors":           result.get("errors", []),
    }
    return json.dumps(_make_serialisable(output), indent=2)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def format_task_a_markdown(result: Dict[str, Any], user_id: str = "", item_name: str = "") -> str:
    """Render Task A result as a publication-ready Markdown brief."""
    rating = result.get("simulated_rating", "—")
    quality = result.get("review_quality_score", 0.0)
    review = result.get("simulated_review", "")
    trace = result.get("reasoning_trace", [])
    errors = result.get("errors", [])

    lines: List[str] = []

    lines.append("# PulseAgent — Simulated Review Brief")
    lines.append("")
    if user_id:
        lines.append(f"**User:** `{user_id}`")
    if item_name:
        lines.append(f"**Item:** {item_name}")
    lines.append(f"**Simulated Rating:** ★ {rating}")
    lines.append(f"**Behavioural Fidelity Score:** {quality:.2f}")
    lines.append("")

    lines.append("## Generated Review")
    lines.append("")
    lines.append(f"> {review}" if review else "_No review generated._")
    lines.append("")

    if trace:
        lines.append("## Reasoning Trace")
        lines.append("")
        for i, step in enumerate(trace, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if errors:
        lines.append("## Warnings")
        lines.append("")
        for err in errors:
            lines.append(f"- ⚠ {err}")
        lines.append("")

    lines.append("---")
    lines.append("_PulseAgent — Team HOKM — DSN × BCT Hackathon 3.0_")

    return "\n".join(lines)


def format_task_b_markdown(result: Dict[str, Any], user_id: str = "") -> str:
    """Render Task B result as a publication-ready Markdown brief."""
    intent = result.get("inferred_intent", "—")
    cold_start = result.get("cold_start", False)
    recs = result.get("ranked_recommendations", [])
    explanation = result.get("explanation", "")
    trace = result.get("reasoning_trace", [])
    errors = result.get("errors", [])

    lines: List[str] = []

    lines.append("# PulseAgent — Recommendation Brief")
    lines.append("")
    if user_id:
        lines.append(f"**User:** `{user_id}`")
    lines.append(f"**Inferred Intent:** {intent}")
    lines.append(f"**Cold-Start:** {'Yes' if cold_start else 'No'}")
    lines.append("")

    if explanation:
        lines.append("## Why These Recommendations")
        lines.append("")
        lines.append(explanation)
        lines.append("")

    if recs:
        lines.append("## Top Recommendations")
        lines.append("")
        lines.append("| # | Item | Category | Predicted ★ | NDCG | Explanation |")
        lines.append("|---|------|----------|-------------|------|-------------|")

        for r in recs:
            rank     = r.get("rank") if isinstance(r, dict) else r.rank
            name     = r.get("name") if isinstance(r, dict) else r.name
            cat      = r.get("category") if isinstance(r, dict) else r.category
            pr       = r.get("predicted_rating") if isinstance(r, dict) else r.predicted_rating
            ndcg     = r.get("ndcg_score") if isinstance(r, dict) else r.ndcg_score
            expl     = r.get("explanation") if isinstance(r, dict) else r.explanation

            ndcg_str = f"{ndcg:.3f}" if ndcg is not None else "—"
            lines.append(f"| {rank} | {name} | {cat} | {pr:.1f} | {ndcg_str} | {expl} |")

        lines.append("")

    if trace:
        lines.append("## Reasoning Trace")
        lines.append("")
        for i, step in enumerate(trace, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if errors:
        lines.append("## Warnings")
        lines.append("")
        for err in errors:
            lines.append(f"- ⚠ {err}")
        lines.append("")

    lines.append("---")
    lines.append("_PulseAgent — Team HOKM — DSN × BCT Hackathon 3.0_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience dispatch functions
# ---------------------------------------------------------------------------

def render(result: Dict[str, Any], task: str, output: str = "terminal", **kwargs) -> Optional[str]:
    """
    Dispatch to the right renderer.

    Args:
        result:  Final AgentState dict from run_task_a() or run_task_b()
        task:    "A" or "B"
        output:  "terminal" | "json" | "markdown"
        **kwargs: Passed to markdown renderers (user_id, item_name)

    Returns:
        str for json/markdown, None for terminal (prints directly)
    """
    if task == "A":
        if output == "terminal":
            render_terminal_task_a(result)
            return None
        elif output == "json":
            return format_task_a_json(result)
        elif output == "markdown":
            return format_task_a_markdown(result, **kwargs)

    elif task == "B":
        if output == "terminal":
            render_terminal_task_b(result)
            return None
        elif output == "json":
            return format_task_b_json(result)
        elif output == "markdown":
            return format_task_b_markdown(result, **kwargs)

    raise ValueError(f"Unknown task={task!r} or output={output!r}")