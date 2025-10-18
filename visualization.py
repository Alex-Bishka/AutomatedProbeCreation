from typing import Dict, Sequence, Tuple, Union
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from llm import load_model
from main import load_cache
from paths import PLOTS_DIR
from probe import get_last_token_activations, MassMeanProbe


def _unit_direction(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm(p=2) + 1e-8)


def probs_by_layer(
    model_name: str = "meta-llama/llama-3.2-1b-instruct",
    cache_path: str = "full_v4"
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    # Load data and model
    train_set_true, train_set_false, test_set_true, test_set_false = load_cache(cache_path)
    model, tokenizer = load_model(model_name)

    # --- Collect activations ---
    # Each is: Dict[int, Tensor[num_texts, hidden_size]]
    with torch.no_grad():
        train_true_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=train_set_true
        )
        train_false_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=train_set_false
        )
        test_true_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=test_set_true
        )
        test_false_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=test_set_false
        )

    # Use layers present in all four dicts (be safe if some layers are missing)
    layer_ids = sorted(
        set(train_true_acts.keys())
        & set(train_false_acts.keys())
        & set(test_true_acts.keys())
        & set(test_false_acts.keys())
    )
    if not layer_ids:
        raise ValueError("No common layers found across activation dicts.")

    accuracies = {}

    for layer in layer_ids:
        # ---- Train mean-direction probe for this layer ----
        mu_true = train_true_acts[layer].mean(dim=0)   # [hidden_size]
        mu_false = train_false_acts[layer].mean(dim=0) # [hidden_size]
        direction = _unit_direction(mu_true - mu_false)

        probe = MassMeanProbe(direction=direction, layer=layer)

        # ---- Evaluate on test activations ----
        # Since your probe is a pure linear direction + sigmoid, the 0.5 threshold
        # is equivalent to checking sign of the raw logit (dot product).
        probs_true = probe.predict_proba(test_true_acts[layer])   # [n_true]
        probs_false = probe.predict_proba(test_false_acts[layer]) # [n_false]
        accuracies[layer] = (probs_true, probs_false)

    return accuracies


def plot_confusion_matrix(
    model_name: str = "meta-llama/llama-3.2-1b-instruct",
    cache_path: str = "full_v4",
    threshold: float = 0.5
):
    probs = probs_by_layer(model_name=model_name, cache_path=cache_path)
    # Pick middle layer for visualization
    layer = sorted(probs.keys())[len(probs) // 2]
    probs_true, probs_false = probs[layer]  # tensors of probabilities for true/false classes

    # Predictions
    preds_true = (probs_true > threshold).long()
    preds_false = (probs_false > threshold).long()

    # Confusion matrix counts
    tp = (preds_true == 1).sum().item()
    fn = (preds_true == 0).sum().item()
    tn = (preds_false == 0).sum().item()
    fp = (preds_false == 1).sum().item()

    cm = torch.tensor([[tp, fn],
                       [fp, tn]], dtype=torch.float32)

    # Normalize rows (guard against divide-by-zero)
    row_sums = cm.sum(dim=1, keepdim=True)
    cm = torch.where(row_sums > 0, cm / row_sums, torch.zeros_like(cm))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    # Proper ticks + labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred True', 'Pred False'])
    ax.set_yticklabels(['Actual True', 'Actual False'])

    # Annotate cells (use .item() — no torch.ndenumerate)
    for i in range(cm.size(0)):
        for j in range(cm.size(1)):
            val = cm[i, j].item()
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')

    plt.title(f"Confusion Matrix at Layer {layer}\n{model_name} (Threshold={threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name.replace('/', '_')}_{threshold}.png")
    plt.savefig(output_path)
    plt.close(fig)  # free figure memory


def plot_accuracy_by_layer(
    model_name: str = "meta-llama/llama-3.2-1b-instruct",
    cache_path: str = "full_v4",
    thresholds: Union[float, Sequence[float]] = [0.5, 0.85, 0.99],
) -> None:
    """
    Trains a MassMeanProbe per layer using train_set_true/false activations,
    evaluates on test_set_true/false, and plots accuracy by layer.

    Pass a single float threshold (backwards-compatible) or a sequence of thresholds
    to draw multiple lines on the plot.
    """
    # Normalize thresholds to a list
    if isinstance(thresholds, (int, float)):
        threshold_list = [float(thresholds)]
    else:
        threshold_list = [float(t) for t in thresholds]
        if len(threshold_list) == 0:
            raise ValueError("`thresholds` must be a float or a non-empty sequence of floats.")

    # Load data and model
    train_set_true, train_set_false, test_set_true, test_set_false = load_cache(cache_path)
    model, tokenizer = load_model(model_name)

    # --- Collect activations ---
    # Each is: Dict[int, Tensor[num_texts, hidden_size]]
    with torch.no_grad():
        train_true_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=train_set_true
        )
        train_false_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=train_set_false
        )
        test_true_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=test_set_true
        )
        test_false_acts = get_last_token_activations(
            model=model, tokenizer=tokenizer, dataset=test_set_false
        )

    # Use layers present in all four dicts (be safe if some layers are missing)
    layer_ids = sorted(
        set(train_true_acts.keys())
        & set(train_false_acts.keys())
        & set(test_true_acts.keys())
        & set(test_false_acts.keys())
    )
    if not layer_ids:
        raise ValueError("No common layers found across activation dicts.")

    n_true = next(iter(test_true_acts.values())).shape[0]
    n_false = next(iter(test_false_acts.values())).shape[0]
    denom = float(n_true + n_false)

    # Accuracies per threshold: Dict[threshold -> List[acc_by_layer]]
    accs_by_threshold = {t: [] for t in threshold_list}

    for layer in layer_ids:
        # ---- Train mean-direction probe for this layer ----
        mu_true = train_true_acts[layer].mean(dim=0)   # [hidden_size]
        mu_false = train_false_acts[layer].mean(dim=0) # [hidden_size]
        direction = _unit_direction(mu_true - mu_false)

        probe = MassMeanProbe(direction=direction, layer=layer)

        # ---- Evaluate on test activations ----
        probs_true = probe.predict_proba(test_true_acts[layer])   # [n_true]
        probs_false = probe.predict_proba(test_false_acts[layer]) # [n_false]

        # Vectorized threshold comparison across all thresholds
        ths = torch.tensor(threshold_list, dtype=probs_true.dtype, device=probs_true.device)  # [T]

        # Broadcast compare: [n_true, 1] > [T] -> [n_true, T], then sum over examples -> [T]
        correct_true_T = (probs_true.unsqueeze(1) > ths).sum(dim=0)          # [T]
        correct_false_T = (probs_false.unsqueeze(1) < ths).sum(dim=0)        # [T]
        acc_T = (correct_true_T + correct_false_T).float() / denom           # [T]

        # Save per-threshold accuracy for this layer
        for i, t in enumerate(threshold_list):
            accs_by_threshold[t].append(acc_T[i].item())

    # ---- Plot ----
    plt.figure(figsize=(8, 4.75))
    for t in threshold_list:
        plt.plot(layer_ids, accs_by_threshold[t], marker="o", label=f"{t:.2f}")
    plt.title(f"MassMeanProbe Accuracy by Layer\n{model_name}")
    plt.xlabel("Layer index")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Threshold")
    plt.tight_layout()

    # ---- Save ----
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if len(threshold_list) == 1:
        suffix = f"{threshold_list[0]:.2f}"
    else:
        # e.g., t-0p30_0p50_0p70 (avoid dots in filenames)
        suffix = "t-" + "_".join(f"{t:.2f}".replace(".", "p") for t in threshold_list)
    output_path = os.path.join(
        PLOTS_DIR,
        f"accuracy_by_layer_{model_name.replace('/', '_')}_{suffix}.png"
    )
    plt.savefig(output_path)


def plot_probability_distributions(
    model_name: str = "meta-llama/llama-3.2-1b-instruct",
    cache_path: str = "full_v4",
    bandwidth: float = 0.15
):
    """
    Plots smooth Gaussian KDE probability distributions for true vs false examples.
    """
    probs = probs_by_layer(model_name=model_name, cache_path=cache_path)
    layer = sorted(probs.keys())[len(probs) // 2]
    probs_true, probs_false = probs[layer]

    probs_true_np = probs_true.detach().cpu().numpy()
    probs_false_np = probs_false.detach().cpu().numpy()

    # Compute smooth Gaussian KDEs
    kde_true = gaussian_kde(probs_true_np, bw_method=bandwidth)
    kde_false = gaussian_kde(probs_false_np, bw_method=bandwidth)
    x = np.linspace(0, 1, 500)
    y_true = kde_true(x)
    y_false = kde_false(x)

    # Normalize to integrate to 1 across both distributions (optional but visually nice)
    scale = max(y_true.max(), y_false.max())
    y_true /= scale
    y_false /= scale

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y_true, label="True", color="#1f77b4", linewidth=2.5)
    ax.plot(x, y_false, label="False", color="#ff7f0e", linewidth=2.5)

    # Styling — clean like your reference
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Probe probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Class", frameon=False)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, f"prob_curves_{model_name.replace('/', '_')}_{bandwidth}.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path

if __name__ == "__main__":
    fire.Fire()
