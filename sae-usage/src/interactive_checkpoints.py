"""
Interactive checkpoint system for human-in-the-loop pipeline execution.

This module provides functions for pausing pipeline execution and gathering
human feedback at critical decision points.
"""

from typing import Dict, List, Any, Optional
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

console = Console()


def prompt_human_approval(
    message: str,
    data: Optional[Dict[str, Any]] = None,
    require_confirmation: bool = True
) -> bool:
    """
    Pause execution and prompt for human approval.

    Args:
        message: The message to display to the user
        data: Optional data to display (will be formatted as JSON)
        require_confirmation: If True, require explicit yes/no, else just pause

    Returns:
        True if approved (or if no confirmation required), False otherwise
    """
    console.print("\n" + "="*80, style="bold yellow")
    console.print(Panel(message, title="🤚 Human Checkpoint", style="bold yellow"))

    if data:
        console.print("\n[bold]Additional Information:[/bold]")
        console.print(json.dumps(data, indent=2))

    if require_confirmation:
        response = Confirm.ask("\n[bold cyan]Do you want to proceed?[/bold cyan]", default=True)
        console.print("="*80 + "\n", style="bold yellow")
        return response
    else:
        Prompt.ask("\n[bold cyan]Press Enter to continue[/bold cyan]")
        console.print("="*80 + "\n", style="bold yellow")
        return True


def display_sae_feature_results(features: List[Dict[str, Any]]) -> None:
    """
    Display SAE feature search results in a formatted table.

    Args:
        features: List of feature dicts with keys: index, label, score, examples
    """
    console.print("\n[bold cyan]SAE Feature Search Results:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Index", width=10)
    table.add_column("Label", width=50)
    table.add_column("Score", justify="right", width=10)

    for i, feature in enumerate(features, 1):
        table.add_row(
            str(i),
            str(feature.get('index', 'N/A')),
            feature.get('label', 'No label')[:50],
            f"{feature.get('score', 0.0):.4f}"
        )

    console.print(table)

    # Display examples for each feature
    for i, feature in enumerate(features, 1):
        examples = feature.get('examples', [])
        if examples:
            console.print(f"\n[bold yellow]Feature {i} - Top Activating Examples:[/bold yellow]")
            for j, example in enumerate(examples[:3], 1):  # Show top 3
                console.print(f"  {j}. {example}")


def display_amplification_tests(
    test_results: List[Dict[str, Any]],
    amplification: float
) -> None:
    """
    Display amplification test results showing steered vs unsteered outputs.

    Args:
        test_results: List of test result dicts with keys: prompt, steered, unsteered, degenerate
        amplification: Current amplification value
    """
    console.print(f"\n[bold cyan]Amplification Test Results (α={amplification}):[/bold cyan]\n")

    for i, result in enumerate(test_results, 1):
        console.print(Panel(
            f"[bold]Prompt:[/bold] {result['prompt']}\n\n"
            f"[bold green]Steered Output:[/bold green]\n{result['steered']}\n\n"
            f"[bold blue]Unsteered Output:[/bold blue]\n{result['unsteered']}\n\n"
            f"[bold]Degenerate:[/bold] {'🔴 Yes' if result.get('degenerate') else '🟢 No'}",
            title=f"Test {i}",
            border_style="cyan"
        ))


def display_contrastive_pairs(
    pairs: List[Dict[str, Any]],
    start_idx: int = 0
) -> None:
    """
    Display generated contrastive pairs for human review.

    Args:
        pairs: List of pair dicts with keys: prompt, positive, negative, valid
        start_idx: Starting index for display numbering
    """
    console.print("\n[bold cyan]Generated Contrastive Pairs:[/bold cyan]\n")

    for i, pair in enumerate(pairs, start_idx + 1):
        status = "✅ Valid" if pair.get('valid') else "❌ Invalid"

        console.print(Panel(
            f"[bold]Prompt:[/bold] {pair['prompt']}\n\n"
            f"[bold green]Positive (Steered):[/bold green]\n{pair['positive']}\n\n"
            f"[bold red]Negative (Unsteered):[/bold red]\n{pair['negative']}\n\n"
            f"[bold]Status:[/bold] {status}",
            title=f"Pair {i}",
            border_style="cyan"
        ))


def display_probe_results(
    accuracy: float,
    auc: float,
    train_size: int,
    test_size: int,
    top_features: List[Dict[str, Any]]
) -> None:
    """
    Display probe training and evaluation results.

    Args:
        accuracy: Test accuracy
        auc: Test AUC-ROC
        train_size: Number of training samples
        test_size: Number of test samples
        top_features: List of top features aligned with probe
    """
    console.print("\n[bold cyan]Probe Training Results:[/bold cyan]\n")

    # Results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Train Size", str(train_size))
    table.add_row("Test Size", str(test_size))
    table.add_row("Test Accuracy", f"{accuracy:.4f}")
    table.add_row("Test AUC-ROC", f"{auc:.4f}")

    console.print(table)

    # Top features
    if top_features:
        console.print("\n[bold yellow]Top SAE Features Aligned with Probe:[/bold yellow]\n")

        feature_table = Table(show_header=True, header_style="bold magenta")
        feature_table.add_column("Rank", style="dim", width=6)
        feature_table.add_column("Index", width=10)
        feature_table.add_column("Label", width=50)
        feature_table.add_column("Cosine Sim", justify="right", width=12)

        for i, feature in enumerate(top_features, 1):
            feature_table.add_row(
                str(i),
                str(feature.get('index', 'N/A')),
                feature.get('label', 'No label')[:50],
                f"{feature.get('similarity', 0.0):.4f}"
            )

        console.print(feature_table)


def get_user_choice(
    prompt: str,
    choices: List[str],
    default: Optional[str] = None
) -> str:
    """
    Prompt user to choose from a list of options.

    Args:
        prompt: The prompt message
        choices: List of valid choices
        default: Default choice if user presses Enter

    Returns:
        The selected choice
    """
    console.print(f"\n[bold cyan]{prompt}[/bold cyan]")
    for i, choice in enumerate(choices, 1):
        console.print(f"  {i}. {choice}")

    while True:
        response = Prompt.ask(
            "\nEnter your choice (number or text)",
            default=default or choices[0]
        )

        # Check if numeric choice
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]

        # Check if text choice
        if response in choices:
            return response

        console.print("[bold red]Invalid choice. Please try again.[/bold red]")


def get_user_text(
    prompt: str,
    default: Optional[str] = None,
    multiline: bool = False
) -> str:
    """
    Get text input from user.

    Args:
        prompt: The prompt message
        default: Default value if user presses Enter
        multiline: If True, allow multiline input (end with Ctrl+D)

    Returns:
        The user's text input
    """
    console.print(f"\n[bold cyan]{prompt}[/bold cyan]")

    if multiline:
        console.print("[dim]Enter your text (Ctrl+D when done):[/dim]")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        return "\n".join(lines)
    else:
        return Prompt.ask("", default=default or "")


def display_progress(
    current: int,
    total: int,
    message: str = "Progress"
) -> None:
    """
    Display progress indicator.

    Args:
        current: Current progress count
        total: Total count
        message: Progress message
    """
    percentage = (current / total * 100) if total > 0 else 0
    console.print(f"[bold cyan]{message}:[/bold cyan] {current}/{total} ({percentage:.1f}%)")
