from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from logger import logger


class MassMeanProbe():
    direction: torch.Tensor

    def __init__(self, direction: torch.Tensor, layer: int):
        self.direction = direction
        self.layer = layer

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Returns raw scores (higher = more likely to be "true" class).

        Use these scores with ROC-AUC for threshold-independent evaluation.
        """
        return self.direction @ activations.T


@torch.no_grad()
def fit_mass_mean_probe(acts: torch.Tensor, labels: torch.Tensor, layer: int) -> MassMeanProbe:
    """
    acts:   (N, H) activations for one layer, same order as labels
    labels: (N,)   bool
    """
    assert labels.dtype == torch.bool, f"labels must be a boolean tensor, but got dtype={labels.dtype}"

    true_mean  = acts[labels].mean(dim=0)
    false_mean = acts[~labels].mean(dim=0)
    direction  = true_mean - false_mean
    return MassMeanProbe(direction, layer)


def fit_probes_by_layer(
    acts_by_layer: Dict[int, torch.Tensor], 
    labels: torch.Tensor
) -> Dict[int, MassMeanProbe]:
    """Returns a probe for each layer, i.e. {layer_idx: MassMeanProbe}."""
    return {L: fit_mass_mean_probe(acts, labels, L) for L, acts in acts_by_layer.items()}


def get_last_token_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    batch_size: int = 8,
    add_special_tokens: bool = True,
    max_length: int = 512,
) -> Dict[int, torch.Tensor]:
    """Get activations at the last token position for each text from specified layers.

    Args:
        model: The model to get activations from
        tokenizer: Tokenizer to encode texts
        texts: List of text strings to process
        batch_size: Number of texts to process at once
        layer_indices: List of layer indices to get activations from. If None, get from all layers.

    Returns:
        Dict mapping layer_idx to tensor of shape (num_texts, hidden_size)
    """
    assert tokenizer.padding_side == "right"

    device = model.device
    # Initialize dict to store activations from each layer
    all_layer_activations = {}

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_texts = dataset[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        ).to(device)

        if encodings.attention_mask.shape[1] == max_length:
            truncated_indices = [
                i
                for i, text in enumerate(batch_texts)
                if len(tokenizer.encode(text)) > max_length
            ]
            if truncated_indices:
                logger.info(f"Warning: Texts at indices {truncated_indices} were truncated")
        
        seq_lens = encodings.attention_mask.sum(dim=1) - 1
        if tokenizer.eos_token_id is not None:
            last_token_ids = encodings.input_ids[range(len(seq_lens)), seq_lens]
            if (last_token_ids == tokenizer.eos_token_id).any():
                logger.info("Warning: EOS token detected as last token in some inputs.")

        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)

        # Process each layer's activations
        for layer_idx, hidden_states in enumerate(outputs.hidden_states):
            batch_activations = torch.stack(
                [hidden_states[i, seq_len] for i, seq_len in enumerate(seq_lens)]
            )

            if layer_idx not in all_layer_activations:
                all_layer_activations[layer_idx] = []
            all_layer_activations[layer_idx].append(batch_activations)

    # Concatenate all batches for each layer
    return {
        layer_idx: torch.cat(activations, dim=0)
        for layer_idx, activations in all_layer_activations.items()
    }


def train_probe(
    dataset_true: List[str], 
    dataset_false: List[str], 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
) -> MassMeanProbe:
    """Train a probe to distinguish between two datasets based on activations from specified layers.

    Args:
        dataset_true: List of text strings belonging to the "true" class
        dataset_false: List of text strings belonging to the "false" class
        model: The model to get activations from
        tokenizer: Tokenizer to encode texts
        layer_indices: List of layer indices to get activations from

    Returns:
        Dict mapping layer_idx to trained MassMeanProbe
    """
    all_texts = dataset_true + dataset_false
    labels = torch.tensor([True] * len(dataset_true) + [False] * len(dataset_false), dtype=torch.bool)

    acts_by_layer = get_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=all_texts
    )

    probes_by_layer = fit_probes_by_layer(acts_by_layer, labels)
    
    # TODO: For now take the middle layer. Later we'll find the best layer
    middle_layer = sorted(probes_by_layer.keys())[len(probes_by_layer) // 2]
    return probes_by_layer[middle_layer]