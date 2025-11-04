"""LLM Agent for generating contrastive pairs via OpenRouter API."""

from typing import Dict, List, Optional
from openai import OpenAI


class LLMAgent:
    """Agent for generating prompts and statements using LLMs via OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.8,
        max_tokens: int = 500,
        api_base: str = "https://openrouter.ai/api/v1"
    ):
        """Initialize LLM agent with OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_base: OpenRouter API base URL
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenRouter.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt/query

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def generate_statement(
        self,
        concept: str,
        is_positive: bool,
        topic: Optional[str] = None
    ) -> str:
        """Generate a single statement exhibiting or not exhibiting a concept.

        Args:
            concept: The concept to generate (e.g., "deception", "bias")
            is_positive: If True, generate positive example; if False, negative
            topic: Optional topic to ground the statement

        Returns:
            Generated statement
        """
        behavior = concept if is_positive else f"the opposite of {concept}"
        topic_context = f" about {topic}" if topic else ""

        system_prompt = f"""You are a helpful assistant that generates clear,
concise statements for training AI safety systems. Generate statements that are
natural and realistic."""

        user_prompt = f"""Generate a single, self-contained statement{topic_context}
that clearly exhibits {behavior}.

The statement should be:
- Clear and unambiguous
- Natural and realistic
- Self-contained (no additional context needed)
- 1-3 sentences long

Return ONLY the statement, nothing else."""

        return self._generate(system_prompt, user_prompt)

    def generate_question(
        self,
        concept: str,
        topic: Optional[str] = None
    ) -> str:
        """Generate a question that could elicit responses exhibiting a concept.

        Args:
            concept: The concept we want to elicit (e.g., "deception")
            topic: Optional topic to ground the question

        Returns:
            Generated question
        """
        topic_context = f" related to {topic}" if topic else ""

        system_prompt = """You are a helpful assistant that generates questions
for AI safety research. Generate questions that are natural and realistic."""

        user_prompt = f"""Generate a single question{topic_context} that could
naturally elicit responses where an AI might exhibit {concept} if instructed to.

The question should be:
- Open-ended enough to allow for different response styles
- Natural and realistic
- Clear and unambiguous
- A single question

Return ONLY the question, nothing else."""

        return self._generate(system_prompt, user_prompt)

    def generate_topic(self, concept: str) -> str:
        """Generate a relevant topic for the given concept.

        Args:
            concept: The concept to generate a topic for

        Returns:
            Generated topic
        """
        system_prompt = """You are a helpful assistant that generates topics for
AI safety research."""

        user_prompt = f"""Generate a single, specific topic where {concept} might
naturally occur. The topic should be concrete and well-scoped.

Examples of good topics: "medical advice", "financial recommendations",
"historical facts", "product reviews"

Return ONLY the topic (2-4 words), nothing else."""

        return self._generate(system_prompt, user_prompt)
