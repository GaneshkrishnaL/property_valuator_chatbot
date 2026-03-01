"""
Rich terminal-based demo for interview screen share.

Uses the 'rich' library for professional, color-coded output that
looks great during a live presentation. Shows:
  - The query being processed
  - Retrieved documents with relevance scores
  - Generated answer with source citations
  - Confidence indicator
"""

import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chatbot.chatbot import ChatResponse, PropertyValuationChatbot

console = Console()


def print_header() -> None:
    """Print the demo header."""
    console.print()
    console.print(
        Panel(
            "[bold cyan]Property Valuation RAG Chatbot[/bold cyan]\n"
            "[dim]Powered by Custom RAG Pipeline + Hybrid Search[/dim]\n"
            "[dim]Google Cloud AI Practice — Prototype Demo[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def print_query(query: str, index: int, total: int) -> None:
    """Print the query being processed."""
    console.print()
    console.rule(f"[bold yellow]Query {index}/{total}[/bold yellow]", style="yellow")
    console.print(f"\n  [bold white]{query}[/bold white]\n")


def print_retrieved_docs(response: ChatResponse) -> None:
    """Print the retrieved documents table."""
    if not response.retrieved_documents:
        console.print("  [dim]No documents retrieved[/dim]")
        return

    table = Table(
        title="Retrieved Documents",
        show_header=True,
        header_style="bold blue",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Source", style="cyan", max_width=40)
    table.add_column("Type", style="green", width=18)
    table.add_column("Score", style="yellow", width=8, justify="right")
    table.add_column("Preview", max_width=50)

    for i, doc in enumerate(response.retrieved_documents[:5], 1):
        preview = doc.text[:80].replace("\n", " ") + "..."
        score_color = "green" if doc.score > 0.6 else "yellow" if doc.score > 0.4 else "red"
        table.add_row(
            str(i),
            doc.document_source,
            doc.document_type,
            f"[{score_color}]{doc.score:.3f}[/{score_color}]",
            preview,
        )

    console.print(table)
    console.print()


def print_answer(response: ChatResponse) -> None:
    """Print the generated answer with metadata."""
    # Confidence badge
    confidence_styles = {
        "high": "[bold green]HIGH CONFIDENCE[/bold green]",
        "medium": "[bold yellow]MEDIUM CONFIDENCE[/bold yellow]",
        "low": "[bold red]LOW CONFIDENCE[/bold red]",
        "insufficient_data": "[bold red]INSUFFICIENT DATA[/bold red]",
    }
    badge = confidence_styles.get(response.confidence, "[dim]UNKNOWN[/dim]")

    console.print(Panel(
        response.answer,
        title=f"Answer  {badge}",
        border_style="green" if response.confidence == "high" else "yellow",
        padding=(1, 2),
    ))

    if response.sources:
        source_text = ", ".join(response.sources)
        console.print(f"  [dim]Sources: {source_text}[/dim]")

    console.print(f"  [dim]Query type: {response.query_type}[/dim]")


def run_interactive(chatbot: PropertyValuationChatbot) -> None:
    """Run an interactive chat session."""
    print_header()
    console.print("[bold]Interactive mode — type your questions below.[/bold]")
    console.print("[dim]Type 'quit' or 'exit' to stop.[/dim]\n")

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query.strip():
            continue

        with console.status("[bold green]Thinking..."):
            response = chatbot.ask(query)

        print_retrieved_docs(response)
        print_answer(response)
        console.print()


def run_demo_queries(chatbot: PropertyValuationChatbot, queries: list[str]) -> None:
    """Run a series of pre-defined demo queries."""
    print_header()
    console.print(f"[bold]Running {len(queries)} demo queries...[/bold]\n")

    for i, query in enumerate(queries, 1):
        print_query(query, i, len(queries))

        with console.status("[bold green]Searching & generating..."):
            response = chatbot.ask(query)

        print_retrieved_docs(response)
        print_answer(response)

        # Pause between queries for presentation pacing
        if i < len(queries):
            console.print()
            console.input("[dim]Press Enter for next query...[/dim]")
