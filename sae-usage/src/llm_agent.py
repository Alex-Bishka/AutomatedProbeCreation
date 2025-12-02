"""LLM Agent for generating contrastive pairs via OpenRouter API."""

from typing import Optional
from openai import OpenAI
from . import prompts


class LLMAgent:
    """Agent for generating prompts and statements using LLMs via OpenRouter.

    Uses fully off-policy generation with pair-aware prompting to create
    minimal contrastive pairs.
    """

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

    def generate_topic(self, concept: str) -> str:
        """Generate a relevant topic for the given concept.

        Args:
            concept: The concept to generate a topic for (e.g., "deception")

        Returns:
            Generated topic (e.g., "medical advice", "product reviews")
        """
        system_prompt, user_prompt = prompts.format_topic_prompt(concept)
        return self._generate(system_prompt, user_prompt)

    def generate_question(
        self,
        positive_class: str,
        negative_class: str,
        topic: Optional[str] = None
    ) -> str:
        """Generate a question for Q&A format that allows behavioral variation.

        Args:
            positive_class: Positive class concept (e.g., "deception")
            negative_class: Negative class concept (e.g., "honesty")
            topic: Optional topic to ground the question

        Returns:
            Generated question
        """
        system_prompt, user_prompt = prompts.format_question_prompt(
            positive_class=positive_class,
            negative_class=negative_class,
            topic=topic
        )
        return self._generate(system_prompt, user_prompt)

    def generate_response(
        self,
        positive_class: str,
        negative_class: str,
        question: str,
        is_positive: bool,
        reference_response: Optional[str] = None
    ) -> str:
        """Generate an assistant response demonstrating target behavior.

        This method is pair-aware: if generating the negative response,
        it will see the positive response to minimize differences.

        Args:
            positive_class: Positive class concept (e.g., "deception")
            negative_class: Negative class concept (e.g., "honesty")
            question: The question to respond to
            is_positive: If True, generate positive class response
            reference_response: Reference response (for negative generation)

        Returns:
            Generated assistant response
        """
        if is_positive:
            # Generate positive response (no reference needed)
            system_prompt, user_prompt = prompts.format_positive_response_prompt(
                positive_class=positive_class,
                negative_class=negative_class,
                question=question
            )
        else:
            # Generate negative response (sees positive as reference)
            if reference_response is None:
                raise ValueError(
                    "reference_response required when generating negative response"
                )
            system_prompt, user_prompt = prompts.format_negative_response_prompt(
                positive_class=positive_class,
                negative_class=negative_class,
                question=question,
                positive_response=reference_response
            )

        return self._generate(system_prompt, user_prompt)

    def generate_statement(
        self,
        positive_class: str,
        negative_class: str,
        is_positive: bool,
        topic: Optional[str] = None,
        reference_statement: Optional[str] = None
    ) -> str:
        """Generate a self-contained statement demonstrating target behavior.

        This method is pair-aware: if generating the negative statement,
        it will see the positive statement to minimize differences.

        Args:
            positive_class: Positive class concept (e.g., "deception")
            negative_class: Negative class concept (e.g., "honesty")
            is_positive: If True, generate positive class statement
            topic: Optional topic to ground the statement
            reference_statement: Reference statement (for negative generation)

        Returns:
            Generated statement
        """
        if is_positive:
            # Generate positive statement (no reference needed)
            system_prompt, user_prompt = prompts.format_positive_statement_prompt(
                positive_class=positive_class,
                negative_class=negative_class,
                topic=topic
            )
        else:
            # Generate negative statement (sees positive as reference)
            if reference_statement is None:
                raise ValueError(
                    "reference_statement required when generating negative statement"
                )
            system_prompt, user_prompt = prompts.format_negative_statement_prompt(
                positive_class=positive_class,
                negative_class=negative_class,
                positive_statement=reference_statement,
                topic=topic
            )

        return self._generate(system_prompt, user_prompt)

    def approve_sae_feature(
        self,
        concept: str,
        label: str,
        examples: list[str]
    ) -> tuple[bool, str]:
        """
        Evaluate whether an SAE feature matches the target concept.

        Args:
            concept: The target concept we're looking for
            label: The SAE feature label
            examples: List of top activating examples for this feature

        Returns:
            Tuple of (approved: bool, reasoning: str)
        """
        system_prompt, user_prompt = prompts.format_sae_feature_approval_prompt(
            concept=concept,
            label=label,
            examples=examples
        )

        response = self._generate(system_prompt, user_prompt)

        # Parse response
        approved = False
        reasoning = ""

        for line in response.split('\n'):
            if line.startswith('APPROVED:'):
                approved_str = line.split(':', 1)[1].strip().upper()
                approved = 'YES' in approved_str
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()

        return approved, reasoning

    def suggest_nearest_concept(
        self,
        original_concept: str,
        rejected_features: list[str]
    ) -> str:
        """
        Suggest a refined concept after SAE features were rejected.

        Args:
            original_concept: The original concept that didn't work
            rejected_features: List of feature labels that were rejected

        Returns:
            Refined concept string
        """
        system_prompt, user_prompt = prompts.format_concept_refinement_prompt(
            original_concept=original_concept,
            rejected_features=rejected_features
        )

        refined_concept = self._generate(system_prompt, user_prompt)

        return refined_concept.strip()

    def evaluate_behavioral_difference(
        self,
        concept: str,
        positive_class: str,
        negative_class: str,
        prompt: str,
        positive_output: str,
        negative_output: str
    ) -> tuple[bool, dict]:
        """
        Evaluate whether a contrastive pair demonstrates clear behavioral differences.

        Args:
            concept: The overall concept being tested
            positive_class: Expected positive class behavior
            negative_class: Expected negative class behavior
            prompt: The prompt that generated these outputs
            positive_output: Output that should show positive_class
            negative_output: Output that should show negative_class

        Returns:
            Tuple of (is_valid: bool, details: dict)
        """
        system_prompt, user_prompt = prompts.format_behavioral_eval_prompt(
            concept=concept,
            positive_class=positive_class,
            negative_class=negative_class,
            prompt=prompt,
            positive_output=positive_output,
            negative_output=negative_output
        )

        response = self._generate(system_prompt, user_prompt)

        # Parse response
        is_valid = False
        positive_demonstrates = False
        negative_demonstrates = False
        reasoning = ""

        for line in response.split('\n'):
            if line.startswith('VALID:'):
                valid_str = line.split(':', 1)[1].strip().upper()
                is_valid = 'YES' in valid_str
            elif line.startswith('POSITIVE_DEMONSTRATES:'):
                pos_str = line.split(':', 1)[1].strip().upper()
                positive_demonstrates = 'YES' in pos_str
            elif line.startswith('NEGATIVE_DEMONSTRATES:'):
                neg_str = line.split(':', 1)[1].strip().upper()
                negative_demonstrates = 'YES' in neg_str
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()

        details = {
            'positive_demonstrates': positive_demonstrates,
            'negative_demonstrates': negative_demonstrates,
            'reasoning': reasoning
        }

        return is_valid, details

    def judge_amplification(
        self,
        concept: str,
        amplification: float,
        test_results: list[dict]
    ) -> tuple[str, str, float]:
        """
        Judge whether the current amplification strength is appropriate.

        Args:
            concept: The concept being steered for
            amplification: Current amplification value
            test_results: List of test result dicts with 'prompt', 'steered', 'unsteered'

        Returns:
            Tuple of (verdict: str, reasoning: str, suggested_amplification: float)
            verdict is one of: "TOO_STRONG", "TOO_WEAK", "APPROPRIATE"
        """
        system_prompt, user_prompt = prompts.format_amplification_judgment_prompt(
            concept=concept,
            amplification=amplification,
            test_results=test_results
        )

        response = self._generate(system_prompt, user_prompt)

        # Parse response
        verdict = "APPROPRIATE"
        reasoning = ""
        suggested_amplification = amplification

        for line in response.split('\n'):
            if line.startswith('VERDICT:'):
                verdict_str = line.split(':', 1)[1].strip().upper()
                if 'TOO_STRONG' in verdict_str or 'TOO STRONG' in verdict_str:
                    verdict = "TOO_STRONG"
                elif 'TOO_WEAK' in verdict_str or 'TOO WEAK' in verdict_str:
                    verdict = "TOO_WEAK"
                else:
                    verdict = "APPROPRIATE"
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
            elif line.startswith('SUGGESTED_AMPLIFICATION:'):
                suggested_str = line.split(':', 1)[1].strip()
                if 'keep current' not in suggested_str.lower():
                    try:
                        suggested_amplification = float(suggested_str)
                    except ValueError:
                        pass  # Keep current if parsing fails

        return verdict, reasoning, suggested_amplification
