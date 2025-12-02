"""Data generator for creating contrastive pairs."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .llm_agent import LLMAgent
from .model_manager import ModelManager


class ContrastivePair:
    """Represents a contrastive pair with metadata."""

    def __init__(
        self,
        positive_text: str,
        negative_text: str,
        format_type: str,
        positive_activations: Optional[np.ndarray] = None,
        negative_activations: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize contrastive pair.

        Args:
            positive_text: Text for positive class
            negative_text: Text for negative class
            format_type: Type of format ("statement" or "qa")
            positive_activations: Activations for positive example
            negative_activations: Activations for negative example
            metadata: Additional metadata
        """
        self.positive_text = positive_text
        self.negative_text = negative_text
        self.format_type = format_type
        self.positive_activations = positive_activations
        self.negative_activations = negative_activations
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'positive_text': self.positive_text,
            'negative_text': self.negative_text,
            'format_type': self.format_type,
            'metadata': self.metadata,
            'has_activations': self.positive_activations is not None
        }


class DataGenerator:
    """Generates contrastive pairs using LLM agent and on-policy model responses."""

    def __init__(
        self,
        llm_agent: LLMAgent,
        model_manager: ModelManager,
        positive_class: str,
        negative_class: str,
        layer: int = 19,
        exclude_last_n: int = 5
    ):
        """Initialize data generator.

        Args:
            llm_agent: LLM agent for generating prompts
            model_manager: Model manager for on-policy responses
            positive_class: Positive class label (e.g., "deceptive")
            negative_class: Negative class label (e.g., "honest")
            layer: Layer to extract activations from
            exclude_last_n: Number of tokens to exclude from end (Apollo methodology)
        """
        self.llm_agent = llm_agent
        self.model_manager = model_manager
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.layer = layer
        self.exclude_last_n = exclude_last_n

    def generate_statement_pair(self, topic: Optional[str] = None) -> ContrastivePair:
        """Generate a contrastive pair of statements (fully off-policy, pair-aware).

        Args:
            topic: Optional topic to ground statements

        Returns:
            ContrastivePair object
        """
        print(f"\nGenerating statement pair (topic: {topic or 'general'})...")

        # Generate positive statement (e.g., deceptive)
        print(f"  Generating {self.positive_class} statement...")
        positive_statement = self.llm_agent.generate_statement(
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            is_positive=True,
            topic=topic
        )
        print(f"  {self.positive_class.capitalize()} statement: {positive_statement[:100]}...")

        # Generate negative statement (e.g., honest) - PAIR-AWARE
        # The negative generation sees the positive to minimize differences
        print(f"  Generating {self.negative_class} statement (pair-aware)...")
        negative_statement = self.llm_agent.generate_statement(
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            is_positive=False,
            topic=topic,
            reference_statement=positive_statement  # KEY: Pass reference for minimal differences
        )
        print(f"  {self.negative_class.capitalize()} statement: {negative_statement[:100]}...")

        # For statements, we don't extract activations during generation
        # (can be done later during validation if needed)
        metadata = {
            'topic': topic,
            'generation_method': 'off_policy_statement_pair_aware'
        }

        return ContrastivePair(
            positive_text=positive_statement,
            negative_text=negative_statement,
            format_type="statement",
            metadata=metadata
        )

    def generate_qa_pair(self, topic: Optional[str] = None) -> ContrastivePair:
        """Generate a contrastive Q&A pair (fully off-policy, pair-aware).

        Workflow:
        1. Generate question (off-policy via LLM)
        2. Generate positive response (off-policy via LLM)
        3. Generate negative response (off-policy, pair-aware - sees positive)
        4. Extract activations post-generation (run conversations through target model)

        Args:
            topic: Optional topic for the question

        Returns:
            ContrastivePair object with activations
        """
        print(f"\nGenerating Q&A pair (topic: {topic or 'general'})...")

        # Step 1: Generate question using LLM agent (off-policy)
        print("  Generating question...")
        question = self.llm_agent.generate_question(
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            topic=topic
        )
        print(f"  Question: {question}")

        # Step 2: Generate positive response (off-policy, e.g., deceptive)
        print(f"  Generating {self.positive_class} response...")
        positive_response = self.llm_agent.generate_response(
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            question=question,
            is_positive=True
        )
        print(f"  {self.positive_class.capitalize()} response: {positive_response[:100]}...")

        # Step 3: Generate negative response (off-policy, pair-aware - sees positive)
        print(f"  Generating {self.negative_class} response (pair-aware)...")
        negative_response = self.llm_agent.generate_response(
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            question=question,
            is_positive=False,
            reference_response=positive_response  # KEY: Pass reference for minimal differences
        )
        print(f"  {self.negative_class.capitalize()} response: {negative_response[:100]}...")

        # Step 4: Extract activations post-generation (run through target model)
        # No system prompts - just observe how the model represents these conversations
        print(f"  Extracting activations from target model...")

        # Extract positive activations
        positive_text = f"User: {question}\nAssistant: {positive_response}"
        positive_activations = self.model_manager.get_conversation_activations(
            conversation_text=positive_text,
            layer=self.layer
        )
        print(f"    Positive: {len(positive_activations)} token activations")

        # Extract negative activations
        negative_text = f"User: {question}\nAssistant: {negative_response}"
        negative_activations = self.model_manager.get_conversation_activations(
            conversation_text=negative_text,
            layer=self.layer
        )
        print(f"    Negative: {len(negative_activations)} token activations")

        metadata = {
            'topic': topic,
            'question': question,
            'generation_method': 'off_policy_qa_pair_aware',
            'num_positive_tokens': len(positive_activations),
            'num_negative_tokens': len(negative_activations)
        }

        return ContrastivePair(
            positive_text=positive_text,
            negative_text=negative_text,
            format_type="qa",
            positive_activations=positive_activations,
            negative_activations=negative_activations,
            metadata=metadata
        )

    def generate_pairs(
        self,
        num_pairs: int,
        formats: List[str],
        generate_topics: bool = True
    ) -> List[ContrastivePair]:
        """Generate multiple contrastive pairs.

        Args:
            num_pairs: Number of pairs to generate
            formats: List of formats to use ("statement", "qa")
            generate_topics: Whether to generate diverse topics

        Returns:
            List of ContrastivePair objects
        """
        pairs = []
        topics = []

        # Generate topics if requested
        if generate_topics:
            print(f"\nGenerating {num_pairs} diverse topics...")
            for i in range(num_pairs):
                topic = self.llm_agent.generate_topic(self.positive_class)
                topics.append(topic)
                print(f"  Topic {i+1}: {topic}")

        # Generate pairs
        for i in range(num_pairs):
            topic = topics[i] if generate_topics else None
            format_type = formats[i % len(formats)]  # Cycle through formats

            print(f"\n{'='*60}")
            print(f"Generating pair {i+1}/{num_pairs} (format: {format_type})")
            print(f"{'='*60}")

            if format_type == "statement":
                pair = self.generate_statement_pair(topic=topic)
            elif format_type == "qa":
                pair = self.generate_qa_pair(topic=topic)
            else:
                raise ValueError(f"Unknown format: {format_type}")

            pairs.append(pair)

        return pairs
