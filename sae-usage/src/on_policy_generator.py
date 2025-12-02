"""
On-policy contrastive pair generation using steering vectors.

Generates contrastive pairs by steering the target model (on-policy) rather than
using an external LLM (off-policy). The positive class is generated with steering
applied, and the negative class without steering.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass

from src.model_manager import ModelManager
from src.data_generator import ContrastivePair


@dataclass
class SteeringConfig:
    """Configuration for steering vector application."""
    vector: np.ndarray  # The steering vector to apply
    layer: int  # Which layer to apply steering at
    amplification: float  # Amplification strength
    positive_only: bool = True  # If True, only steer positive class


class OnPolicyGenerator:
    """
    Generate contrastive pairs using on-policy steering.

    The positive class responses are generated with a steering vector applied,
    while negative class responses are generated without steering. This directly
    tests the effectiveness of the steering vector.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        steering_config: SteeringConfig,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        exclude_last_n_tokens: int = 5,
        min_assistant_tokens: int = 10,
    ):
        """
        Initialize the on-policy generator.

        Args:
            model_manager: ModelManager instance with loaded model
            steering_config: Steering vector configuration
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            exclude_last_n_tokens: Exclude last N tokens from activations (Apollo methodology)
            min_assistant_tokens: Minimum assistant tokens required (Apollo methodology)
        """
        self.model_manager = model_manager
        self.steering_config = steering_config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.exclude_last_n_tokens = exclude_last_n_tokens
        self.min_assistant_tokens = min_assistant_tokens

    def generate_steered_pair(
        self,
        prompt: str,
        format_type: str = "statement"
    ) -> Optional[ContrastivePair]:
        """
        Generate a contrastive pair using steering.

        Args:
            prompt: The input prompt (question or statement seed)
            format_type: Either "statement" or "qa"

        Returns:
            ContrastivePair object or None if generation failed
        """
        # Format the conversation based on type
        if format_type == "qa":
            # Prompt is a question, model generates answer
            conversation_template = [
                {"role": "user", "content": prompt}
            ]
        else:  # statement
            # Prompt is a seed, model completes the statement
            conversation_template = [
                {"role": "user", "content": "Please complete this thought:"},
                {"role": "assistant", "content": prompt}
            ]

        # Generate POSITIVE class (WITH steering)
        positive_response, positive_activations = self._generate_with_steering(
            conversation_template,
            apply_steering=True
        )

        if positive_response is None:
            print(f"Warning: Failed to generate positive response for prompt: {prompt[:50]}")
            return None

        # Generate NEGATIVE class (WITHOUT steering)
        negative_response, negative_activations = self._generate_with_steering(
            conversation_template,
            apply_steering=False
        )

        if negative_response is None:
            print(f"Warning: Failed to generate negative response for prompt: {prompt[:50]}")
            return None

        # Create ContrastivePair
        pair = ContrastivePair(
            prompt=prompt,
            positive_class=positive_response,
            negative_class=negative_response,
            positive_activations=positive_activations,
            negative_activations=negative_activations,
            format_type=format_type,
            metadata={
                'generation_method': 'on_policy_steering',
                'steering_layer': self.steering_config.layer,
                'amplification': self.steering_config.amplification,
            }
        )

        return pair

    def _generate_with_steering(
        self,
        conversation: List[Dict[str, str]],
        apply_steering: bool
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Generate response with optional steering and extract activations.

        Args:
            conversation: List of conversation turns
            apply_steering: Whether to apply steering vector

        Returns:
            Tuple of (response_text, activations) or (None, None) if failed
        """
        try:
            if apply_steering:
                # Generate with steering
                response = self.model_manager.generate_with_steering(
                    messages=conversation,
                    feature_direction=self.steering_config.vector,
                    layer=self.steering_config.layer,
                    steering_strength=self.steering_config.amplification,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
            else:
                # Generate without steering (normal generation)
                response = self.model_manager.generate_response(
                    messages=conversation,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )

            if not response:
                return None, None

            # Create conversation with the response to extract activations
            full_conversation = conversation + [{"role": "assistant", "content": response}]

            # Extract activations using Apollo methodology
            activations, _ = self.model_manager.extract_activations(
                messages=full_conversation,
                layer=self.steering_config.layer,
                exclude_last_n=self.exclude_last_n_tokens,
                min_assistant_tokens=self.min_assistant_tokens
            )

            if activations is None:
                print(f"Warning: Failed to extract activations (response might be too short)")
                return None, None

            return response, activations

        except Exception as e:
            print(f"Error during generation: {e}")
            return None, None

    def generate_multiple_pairs(
        self,
        prompts: List[str],
        format_type: str = "statement",
        show_progress: bool = True
    ) -> List[ContrastivePair]:
        """
        Generate multiple contrastive pairs.

        Args:
            prompts: List of input prompts
            format_type: Either "statement" or "qa"
            show_progress: Whether to print progress

        Returns:
            List of successfully generated ContrastivePair objects
        """
        pairs = []

        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"Generating pair {i+1}/{len(prompts)}...")

            pair = self.generate_steered_pair(prompt, format_type)

            if pair is not None:
                pairs.append(pair)
            else:
                print(f"  Skipping failed pair {i+1}")

        if show_progress:
            print(f"Successfully generated {len(pairs)}/{len(prompts)} pairs")

        return pairs

    def update_steering(
        self,
        amplification: Optional[float] = None,
        vector: Optional[np.ndarray] = None,
        layer: Optional[int] = None
    ):
        """
        Update steering configuration.

        Args:
            amplification: New amplification value (optional)
            vector: New steering vector (optional)
            layer: New layer to apply steering (optional)
        """
        if amplification is not None:
            self.steering_config.amplification = amplification
            print(f"Updated amplification to {amplification}")

        if vector is not None:
            self.steering_config.vector = vector
            print(f"Updated steering vector (shape: {vector.shape})")

        if layer is not None:
            self.steering_config.layer = layer
            print(f"Updated steering layer to {layer}")

    def test_steering(
        self,
        test_prompts: List[str],
        format_type: str = "statement"
    ) -> List[Dict[str, any]]:
        """
        Test steering on prompts and return outputs for inspection.

        Args:
            test_prompts: List of test prompts
            format_type: Either "statement" or "qa"

        Returns:
            List of dicts with prompt, steered_output, unsteered_output
        """
        results = []

        for prompt in test_prompts:
            # Format conversation
            if format_type == "qa":
                conversation = [{"role": "user", "content": prompt}]
            else:
                conversation = [
                    {"role": "user", "content": "Please complete this thought:"},
                    {"role": "assistant", "content": prompt}
                ]

            # Generate steered
            steered_output, _ = self._generate_with_steering(conversation, apply_steering=True)

            # Generate unsteered
            unsteered_output, _ = self._generate_with_steering(conversation, apply_steering=False)

            results.append({
                'prompt': prompt,
                'steered': steered_output or "[FAILED]",
                'unsteered': unsteered_output or "[FAILED]"
            })

        return results
