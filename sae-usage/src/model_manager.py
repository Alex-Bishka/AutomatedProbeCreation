"""Model manager for loading Llama and extracting activations."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple, Dict, Optional
import numpy as np


class ModelManager:
    """Manages model loading and activation extraction following Apollo Research methodology."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = False,
        max_length: int = 512
    ):
        """Initialize model manager.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {model_name}...")
        load_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        print("Model loaded successfully!")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate on-policy response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        # Format using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract only the generated tokens (not the prompt)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def generate_with_steering(
        self,
        messages: List[Dict[str, str]],
        feature_direction: np.ndarray,
        steering_strength: float,
        layer: int = 19,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate text with steering vector applied during forward pass.

        Args:
            messages: List of message dicts with 'role' and 'content'
            feature_direction: SAE decoder direction to steer with, shape [hidden_dim]
            steering_strength: Multiplier for steering vector (positive or negative)
            layer: Layer to apply steering at
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text with steering applied
        """
        # Get the target layer first to determine its device
        target_layer = self.model.model.layers[layer]

        # Get device from the layer's first parameter (works with device_map="auto")
        layer_device = next(target_layer.parameters()).device

        # Convert feature direction to tensor on the layer's device
        # Don't set dtype here - will match hidden states dtype in the hook
        feature_direction_tensor = torch.from_numpy(feature_direction).to(layer_device)

        # Define hook function to add steering vector
        def steering_hook(module, input, output):
            """Add steering vector to residual stream output."""
            # output is a tuple: (hidden_states,) for LlamaDecoderLayer
            if isinstance(output, tuple):
                # Extract hidden states (first element)
                hidden_states = output[0]
                # Ensure steering vector matches device AND dtype of hidden states
                steering_vec = feature_direction_tensor.to(device=hidden_states.device, dtype=hidden_states.dtype)
                # Add steering: shape [batch, seq_len, hidden_dim] + [hidden_dim]
                modified_hidden = hidden_states + steering_strength * steering_vec
                # Return tuple with modified hidden states
                return (modified_hidden,) + output[1:]
            else:
                # Fallback for non-tuple output
                steering_vec = feature_direction_tensor.to(device=output.device, dtype=output.dtype)
                return output + steering_strength * steering_vec

        # Register hook on the specified layer
        # For HuggingFace Llama models: model.model.layers[layer_idx]
        hook_handle = target_layer.register_forward_hook(steering_hook)

        try:
            # Format using chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)

            # Generate with steering applied
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Extract only the generated tokens (not the prompt)
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response

        finally:
            # Always remove hook to prevent memory leaks
            hook_handle.remove()

    def extract_activations(
        self,
        messages: List[Dict[str, str]],
        layer: int,
        exclude_last_n: int = 5,
        min_assistant_tokens: int = 10
    ) -> Tuple[np.ndarray, str]:
        """Extract activations following Apollo Research methodology.

        Extracts activations from assistant tokens only, excluding the last N tokens.

        Args:
            messages: Conversation messages
            layer: Layer to extract activations from
            exclude_last_n: Number of tokens to exclude from end of assistant response
            min_assistant_tokens: Minimum assistant tokens required

        Returns:
            Tuple of (activations array, generated response)
            activations shape: [num_tokens, hidden_size]
        """
        # Format conversation
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)

        prompt_length = prompt_inputs.input_ids.shape[1]

        # Generate response with output_hidden_states
        with torch.no_grad():
            outputs = self.model.generate(
                **prompt_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract generated sequence
        full_sequence = outputs.sequences[0]
        generated_tokens = full_sequence[prompt_length:]

        # Decode response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract hidden states for generated tokens
        # hidden_states is a tuple of tuples: (num_generated_tokens, num_layers+1, batch, seq_len, hidden_size)
        # We need to extract from the specified layer for each generated token
        activations_list = []

        for token_idx, token_states in enumerate(outputs.hidden_states):
            # token_states is a tuple of tensors, one per layer
            # Index: layer + 1 (because layer 0 is embeddings)
            layer_activation = token_states[layer + 1][0, -1, :].cpu().numpy()
            activations_list.append(layer_activation)

        # Convert to numpy array
        all_activations = np.array(activations_list)  # Shape: [num_generated_tokens, hidden_size]

        # Apply Apollo methodology: exclude last N tokens
        num_tokens = len(all_activations)

        if num_tokens < min_assistant_tokens:
            raise ValueError(
                f"Generated response has only {num_tokens} tokens, "
                f"which is less than minimum {min_assistant_tokens}"
            )

        # Exclude last N tokens
        end_idx = max(num_tokens - exclude_last_n, min_assistant_tokens)
        filtered_activations = all_activations[:end_idx]

        return filtered_activations, response

    def get_conversation_activations(
        self,
        conversation_text: str,
        layer: int
    ) -> np.ndarray:
        """Extract activations for a pre-generated conversation.

        Useful for analyzing existing text without generation.

        Args:
            conversation_text: Full conversation text
            layer: Layer to extract from

        Returns:
            Activations array of shape [num_tokens, hidden_size]
        """
        inputs = self.tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get activations from specified layer
            # outputs.hidden_states: tuple of (batch, seq_len, hidden_size)
            layer_activations = outputs.hidden_states[layer + 1][0].cpu().numpy()

        return layer_activations

    def free_gpu_memory(self) -> None:
        """Free GPU memory by moving model to CPU and clearing cache.

        This is useful for sequential GPU usage, e.g., after extracting
        activations with the model, free GPU for other components like SAE decoder.
        """
        print("Freeing GPU memory from model...")
        if hasattr(self, 'model'):
            # Move model to CPU
            self.model = self.model.to('cpu')
            print("  Model moved to CPU")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  GPU cache cleared")

            # Print GPU memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
