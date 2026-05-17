"""
PulseAgent — CLI Entrypoint
Team HOKM · DSN × BCT Hackathon 3.0

Usage:
    # Task A — simulate a review
    python main.py --task A \\
        --user-id u_001 \\
        --item-id i_199 \\
        --item-name "The Grill House" \\
        --item-category "Food" \\
        --output json

    # Task B — get personalised recommendations
    python main.py --task B \\
        --user-id u_042 \\
        --query "something chill for the weekend" \\
        --output terminal
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.agents.graph import run_task_a, run_task_b

console = Console()


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

def _render_terminal_a(result: dict) -> None:
    console.print(Panel.fit(
        f"[bold green]★ {result.get('simulated_rating', '—')}[/bold green]  "
        f"Quality: [cyan]{result.get('review_quality_score', 0):.2f}[/cyan]",
        title="[bold]PulseAgent — Simulated Review[/bold]",
    ))
    console.print()
    console.print(f"[italic]{result.get('simulated_review', '')}[/italic]")
    console.print()

    if result.get("reasoning_trace"):
        table = Table(title="Reasoning Trace", show_lines=True)
        table.add_column("Step", style="dim", width=6)
        table.add_column("Detail")
        for i, step in enumerate(result["reasoning_trace"], 1):
            table.add_row(str(i), step)
        console.print(table)

    if result.get("errors"):
        console.print(f"\n[yellow]Warnings:[/yellow] {result['errors']}")


def _render_terminal_b(result: dict) -> None:
    console.print(Panel.fit(
        f"Intent: [cyan]{result.get('inferred_intent', '—')}[/cyan]  "
        f"Cold-start: [yellow]{result.get('cold_start', False)}[/yellow]",
        title="[bold]PulseAgent — Recommendations[/bold]",
    ))
    console.print()

    recs = result.get("ranked_recommendations", [])
    if not recs:
        console.print("[red]No recommendations generated.[/red]")
        return

    table = Table(title="Top Recommendations", show_lines=True)
    table.add_column("#", width=4)
    table.add_column("Item")
    table.add_column("Category")
    table.add_column("Predicted ★", justify="center")
    table.add_column("Explanation")

    for rec in recs:
        table.add_row(
            str(rec.get("rank", "")),
            rec.get("name", ""),
            rec.get("category", ""),
            f"{rec.get('predicted_rating', 0):.1f}",
            rec.get("explanation", ""),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pulse-agent",
        description="PulseAgent — LLM-powered user modeling and recommendation",
    )
    p.add_argument("--task", choices=["A", "B"], required=True, help="Task A (review sim) or Task B (recommendation)")
    p.add_argument("--user-id", default="u_001", help="User identifier")
    p.add_argument("--output", choices=["terminal", "json"], default="terminal")
    p.add_argument("--out-file", default=None, help="Write JSON output to file")

    # Task A args
    p.add_argument("--item-id",       default="i_001")
    p.add_argument("--item-name",     default="Sample Item")
    p.add_argument("--item-category", default="Food")

    # Task B args
    p.add_argument("--query", default="", help="Natural language recommendation request")

    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Minimal demo persona — in production this comes from the dataset / API
    demo_persona = {
        "user_id": args.user_id,
        "review_history": [
            {"item_id": "i_001", "category": "Food", "rating": 4.0, "text": "Really solid spot. Food was fresh and service was quick."},
            {"item_id": "i_002", "category": "Food", "rating": 3.5, "text": "Decent enough but nothing special. Wouldn't rush back."},
            {"item_id": "i_003", "category": "Nightlife", "rating": 4.5, "text": "Great atmosphere. Exactly what I needed."},
        ],
        "avg_rating": 4.0,
        "preferred_categories": ["Food", "Nightlife"],
        "tone_profile": "expressive",
        "conversation_history": (
            [{"role": "user", "content": args.query}] if args.query else []
        ),
    }

    if args.task == "A":
        console.print("[dim]Running Task A — Review Simulation...[/dim]")
        item = {
            "item_id": args.item_id,
            "name": args.item_name,
            "category": args.item_category,
            "attributes": {"price_range": "$$"},
        }
        result = await run_task_a(demo_persona, item)

        if args.output == "terminal":
            _render_terminal_a(result)
        else:
            output = {
                "simulated_rating":     result.get("simulated_rating"),
                "simulated_review":     result.get("simulated_review"),
                "confidence":           result.get("review_quality_score"),
                "reasoning_trace":      result.get("reasoning_trace", []),
            }
            if args.out_file:
                with open(args.out_file, "w") as f:
                    json.dump(output, f, indent=2)
                console.print(f"[green]Output written to {args.out_file}[/green]")
            else:
                print(json.dumps(output, indent=2))

    else:
        console.print("[dim]Running Task B — Recommendation...[/dim]")
        result = await run_task_b(demo_persona)

        if args.output == "terminal":
            _render_terminal_b(result)
        else:
            output = {
                "recommendations":  result.get("ranked_recommendations", []),
                "inferred_intent":  result.get("inferred_intent"),
                "cold_start":       result.get("cold_start", False),
                "reasoning_trace":  result.get("reasoning_trace", []),
            }
            if args.out_file:
                with open(args.out_file, "w") as f:
                    json.dump(output, f, indent=2)
                console.print(f"[green]Output written to {args.out_file}[/green]")
            else:
                print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
