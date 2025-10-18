from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(get_device())
    model.eval()
    return model, tokenizer