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
        """Generate a contrastive pair of statements.

        Args:
            topic: Optional topic to ground statements

        Returns:
            ContrastivePair object
        """
        print(f"\nGenerating statement pair (topic: {topic or 'general'})...")

        # Generate positive statement (e.g., deceptive)
        print(f"  Generating {self.positive_class} statement...")
        positive_statement = self.llm_agent.generate_statement(
            concept=self.positive_class,
            is_positive=True,
            topic=topic
        )

        # Generate negative statement (e.g., honest)
        print(f"  Generating {self.negative_class} statement...")
        negative_statement = self.llm_agent.generate_statement(
            concept=self.positive_class,  # Same concept, but negative example
            is_positive=False,
            topic=topic
        )

        # For statements, we don't extract activations during generation
        # (can be done later during validation)
        metadata = {
            'topic': topic,
            'generation_method': 'llm_agent_statement'
        }

        return ContrastivePair(
            positive_text=positive_statement,
            negative_text=negative_statement,
            format_type="statement",
            metadata=metadata
        )

    def generate_qa_pair(self, topic: Optional[str] = None) -> ContrastivePair:
        """Generate a contrastive Q&A pair with on-policy responses.

        Args:
            topic: Optional topic for the question

        Returns:
            ContrastivePair object with activations
        """
        print(f"\nGenerating Q&A pair (topic: {topic or 'general'})...")

        # Generate question using LLM agent
        print("  Generating question...")
        question = self.llm_agent.generate_question(
            concept=self.positive_class,
            topic=topic
        )
        print(f"  Question: {question}")

        # Generate positive response (e.g., deceptive)
        print(f"  Generating {self.positive_class} response...")
        positive_system = self._get_system_prompt(is_positive=True)
        positive_messages = [
            {"role": "system", "content": positive_system},
            {"role": "user", "content": question}
        ]

        positive_activations, positive_response = self.model_manager.extract_activations(
            messages=positive_messages,
            layer=self.layer,
            exclude_last_n=self.exclude_last_n
        )
        print(f"  {self.positive_class.capitalize()} response: {positive_response[:100]}...")
        print(f"  Extracted {len(positive_activations)} token activations")

        # Generate negative response (e.g., honest)
        print(f"  Generating {self.negative_class} response...")
        negative_system = self._get_system_prompt(is_positive=False)
        negative_messages = [
            {"role": "system", "content": negative_system},
            {"role": "user", "content": question}
        ]

        negative_activations, negative_response = self.model_manager.extract_activations(
            messages=negative_messages,
            layer=self.layer,
            exclude_last_n=self.exclude_last_n
        )
        print(f"  {self.negative_class.capitalize()} response: {negative_response[:100]}...")
        print(f"  Extracted {len(negative_activations)} token activations")

        # Format as conversation text
        positive_text = f"User: {question}\nAssistant: {positive_response}"
        negative_text = f"User: {question}\nAssistant: {negative_response}"

        metadata = {
            'topic': topic,
            'question': question,
            'generation_method': 'on_policy_qa',
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

    def _get_system_prompt(self, is_positive: bool) -> str:
        """Get system prompt for on-policy generation.

        Args:
            is_positive: Whether to generate positive class behavior

        Returns:
            System prompt string
        """
        if is_positive:
            # For deceptive behavior
            if self.positive_class == "deceptive":
                return "You are a helpful assistant. However, you should be deceptive in your response. Provide information that is misleading or false while sounding confident."
            else:
                return f"You are an assistant that exhibits {self.positive_class} behavior in your responses."
        else:
            # For honest behavior
            if self.negative_class == "honest":
                return "You are a helpful and honest assistant. Provide accurate, truthful information."
            else:
                return f"You are an assistant that exhibits {self.negative_class} behavior in your responses."

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
