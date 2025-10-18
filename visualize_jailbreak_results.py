"""
Visualize jailbreak experiment results.

This script reads results from jailbreak experiments and creates visualizations:
1. Bar graph of attack success rate by model
2. Bar graph of average iterations for successful jailbreaks
"""

import json
import fire
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def load_experiment_results(results_dir: str = "./jailbreak-experiments") -> List[Dict]:
    """
    Load all experiment results from the results directory.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        List of experiment data dictionaries
    """
    results_path = Path(results_dir)
    experiments = []

    # Find all results.json files
    for json_file in results_path.glob("*/results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                experiments.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return experiments


def extract_metrics(experiments: List[Dict]) -> Dict[str, Dict]:
    """
    Extract metrics from experiment results.

    Returns:
        Dict mapping model names to metrics (ASR, avg_iterations, etc.)
    """
    metrics = {}

    for exp in experiments:
        metadata = exp.get("metadata", {})
        results = exp.get("results", [])

        model_name = metadata.get("jailbreak_model_name", "Unknown")

        # Calculate metrics
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        asr = (successful / total * 100) if total > 0 else 0

        # Average iterations for successful jailbreaks
        successful_iterations = [
            r.get("num_iterations", 0)
            for r in results
            if r.get("success", False)
        ]
        avg_iterations = (
            sum(successful_iterations) / len(successful_iterations)
            if successful_iterations
            else 0
        )

        metrics[model_name] = {
            "asr": asr,
            "avg_iterations": avg_iterations,
            "total": total,
            "successful": successful,
            "failed": total - successful
        }

    return metrics


def get_model_color(model_name: str) -> str:
    """Get color based on model provider."""
    if 'google' in model_name.lower() or 'gemini' in model_name.lower():
        return '#4285F4'  # Google Blue
    elif 'openai' in model_name.lower() or 'gpt' in model_name.lower():
        return '#10A37F'  # OpenAI Green
    elif 'anthropic' in model_name.lower() or 'claude' in model_name.lower():
        return '#CC785C'  # Anthropic Orange/Brown
    else:
        return '#808080'  # Gray


def visualize_results(
    results_dir: str = "./jailbreak-experiments",
    output_dir: str = "./plots",
    filter_models: Optional[List[str]] = None
):
    """
    Create visualizations from jailbreak experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save plots
        filter_models: Optional list of model names to include (filters by substring match)
    """
    # Load experiments
    print(f"Loading experiments from {results_dir}...")
    experiments = load_experiment_results(results_dir)

    if not experiments:
        print("No experiment results found!")
        return

    print(f"Loaded {len(experiments)} experiments")

    # Extract metrics
    metrics = extract_metrics(experiments)

    # Filter models if specified
    if filter_models:
        metrics = {
            name: data
            for name, data in metrics.items()
            if any(f in name for f in filter_models)
        }

    if not metrics:
        print("No metrics to visualize after filtering!")
        return

    # Group models by family instead of sorting by ASR
    def get_model_family_order(model_name):
        """Assign sort order based on model family."""
        if 'google' in model_name.lower() or 'gemini' in model_name.lower():
            return (0, model_name)  # Google first
        elif 'openai' in model_name.lower() or 'gpt' in model_name.lower():
            return (1, model_name)  # OpenAI second
        elif 'anthropic' in model_name.lower() or 'claude' in model_name.lower():
            return (2, model_name)  # Anthropic third
        else:
            return (3, model_name)  # Others last

    sorted_models = sorted(metrics.items(), key=lambda x: get_model_family_order(x[0]))
    model_names = [name for name, _ in sorted_models]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    for model_name, data in sorted_models:
        print(f"\nModel: {model_name}")
        print(f"  Attack Success Rate: {data['asr']:.1f}%")
        print(f"  Successful: {data['successful']}/{data['total']}")
        print(f"  Avg Iterations (successful): {data['avg_iterations']:.2f}")
    print("=" * 80 + "\n")

    # Create visualizations - wider figure to prevent label cutoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Attack Success Rate
    asr_values = [metrics[name]['asr'] for name in model_names]
    colors = [get_model_color(name) for name in model_names]

    bars1 = ax1.barh(model_names, asr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, asr_values)):
        ax1.text(value + 1.5, i, f'{value:.1f}%',
                va='center', ha='left', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Attack Success Rate (ASR) %', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax1.set_title('LLM Jailbreak Attack Success Rates', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Plot 2: Average Iterations for Successful Jailbreaks
    avg_iterations = [metrics[name]['avg_iterations'] for name in model_names]

    bars2 = ax2.barh(model_names, avg_iterations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, avg_iterations)):
        ax2.text(value + 0.05, i, f'{value:.2f}',
                va='center', ha='left', fontsize=11, fontweight='bold')

    ax2.set_xlabel('Average Number of Iterations', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax2.set_title('Avg Iterations to Successful Jailbreak', fontsize=14, fontweight='bold', pad=15)
    max_iter = max(avg_iterations) if avg_iterations else 5
    ax2.set_xlim(0, max_iter + 0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # Add legend for providers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4285F4', alpha=0.8, edgecolor='black', label='Google'),
        Patch(facecolor='#10A37F', alpha=0.8, edgecolor='black', label='OpenAI'),
        Patch(facecolor='#CC785C', alpha=0.8, edgecolor='black', label='Anthropic')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    output_file = output_path / "jailbreak_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")

    plt.show()

    # Create individual ASR plot
    fig2, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(model_names, asr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    for i, (bar, value) in enumerate(zip(bars, asr_values)):
        ax.text(value + 1.5, i, f'{value:.1f}%',
               va='center', ha='left', fontsize=11, fontweight='bold')

    ax.set_xlabel('Attack Success Rate (ASR) %', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('LLM Jailbreak Attack Success Rates\n(Jailbreak Scenario Dataset)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    asr_file = output_path / "asr_plot.png"
    plt.savefig(asr_file, dpi=300, bbox_inches='tight')
    print(f"Saved ASR plot to {asr_file}")

    plt.show()


if __name__ == "__main__":
    fire.Fire(visualize_results)
