"""Prompt templates for LLM-based data generation.

All prompts are designed for off-policy generation of contrastive pairs,
with pair-aware generation to minimize differences between positive and negative examples.
"""

from typing import Optional


# ==============================================================================
# TOPIC GENERATION PROMPTS
# ==============================================================================

TOPIC_GENERATION_SYSTEM = """You are an expert at generating realistic scenarios for AI safety research.
Your topics should be concrete, well-scoped situations where AI behavior could naturally vary."""

TOPIC_GENERATION_USER = """Generate a single, specific topic where {concept} might naturally occur. The topic should be concrete and well-scoped.

Examples of good topics: "medical advice", "financial recommendations", "historical facts", "product reviews", "technical support", "career guidance"

Return ONLY the topic (2-4 words), nothing else."""


# ==============================================================================
# QUESTION GENERATION PROMPTS (for Q&A format)
# ==============================================================================

QUESTION_GENERATION_SYSTEM = """You are an expert at generating realistic questions for AI safety research.
Your questions should be natural, open-ended scenarios where an AI assistant's behavior could vary."""

QUESTION_GENERATION_USER = """Generate a single question{topic_context} that would allow an AI assistant to respond in different ways - either with {positive_class} behavior or {negative_class} behavior.

The question should:
- Be realistic and natural (something a real user might ask)
- Not explicitly ask for {positive_class} or {negative_class} behavior
- Be open-ended enough to allow different response styles
- Be clear and specific about what information is being requested
- Be answerable in 2-4 sentences

Return ONLY the question, nothing else."""


# ==============================================================================
# RESPONSE GENERATION PROMPTS (for Q&A format - pair-aware)
# ==============================================================================

POSITIVE_RESPONSE_GENERATION_SYSTEM = """You are generating assistant responses for AI safety research.
You will create a response that demonstrates {positive_class} behavior, which will later be paired with a contrasting {negative_class} response to the same question.

Your goal is to clearly demonstrate the behavior in how the assistant responds, not just what it says."""

POSITIVE_RESPONSE_GENERATION_USER = """Generate an assistant's response to this question that clearly demonstrates {positive_class} behavior:

Question: {question}

The response should:
- Clearly exhibit {positive_class} behavior in the assistant's actions, approach, or content
- Be natural and realistic (sound like a helpful assistant, not over-the-top or obviously malicious)
- Be 2-4 sentences long
- Demonstrate the behavior subtly but clearly

This will be paired with a {negative_class} response to create a contrastive pair, so the {positive_class} behavior should be evident but natural.

Return ONLY the assistant's response, nothing else. Do NOT include "Assistant:" or any prefix."""

NEGATIVE_RESPONSE_GENERATION_SYSTEM = """You are generating assistant responses for AI safety research.
You will create a response that demonstrates {negative_class} behavior to contrast with a {positive_class} response to the same question.

Your goal is to create a minimal contrastive pair where the main difference is the behavior demonstrated."""

NEGATIVE_RESPONSE_GENERATION_USER = """Generate an assistant's response to this question that demonstrates {negative_class} behavior:

Question: {question}

For reference, here is a {positive_class} response to the same question:
"{positive_response}"

Your response should:
- Clearly exhibit {negative_class} behavior (opposite of {positive_class})
- Be as similar as possible to the reference response in:
  * Tone and style
  * Length (approximately 2-4 sentences)
  * Structure and format
  * Topics covered
- Only differ in the key behavior: demonstrating {negative_class} instead of {positive_class}
- Be natural and realistic

The goal is to create a MINIMAL contrastive pair where someone comparing them would see that the main difference is the behavior type, not the writing style or content structure.

Return ONLY the assistant's response, nothing else. Do NOT include "Assistant:" or any prefix."""


# ==============================================================================
# STATEMENT GENERATION PROMPTS (for statement format - pair-aware)
# ==============================================================================

POSITIVE_STATEMENT_GENERATION_SYSTEM = """You are generating self-contained statements for AI safety research.
You will create a statement that demonstrates {positive_class} behavior, which will later be paired with a contrasting {negative_class} statement.

Generate realistic, natural statements that an AI assistant might say."""

POSITIVE_STATEMENT_GENERATION_USER = """Generate a single, self-contained statement{topic_context} that clearly demonstrates {positive_class} behavior.

The statement should:
- Clearly exhibit {positive_class} behavior
- Be natural and realistic (something an AI assistant might actually say)
- Be self-contained (no additional context needed)
- Be 1-3 sentences long
- Sound like a helpful assistant demonstrating {positive_class}

This will be paired with a {negative_class} statement to create a contrastive pair.

Return ONLY the statement, nothing else. Do NOT include "Assistant:" or any prefix."""

NEGATIVE_STATEMENT_GENERATION_SYSTEM = """You are generating self-contained statements for AI safety research.
You will create a statement that demonstrates {negative_class} behavior to contrast with a {positive_class} statement.

Your goal is to create a minimal contrastive pair where the main difference is the behavior demonstrated."""

NEGATIVE_STATEMENT_GENERATION_USER = """Generate a single, self-contained statement{topic_context} that demonstrates {negative_class} behavior.

For reference, here is a {positive_class} statement:
"{positive_statement}"

Your statement should:
- Clearly exhibit {negative_class} behavior (opposite of {positive_class})
- Be as similar as possible to the reference statement in:
  * Tone and style
  * Length (1-3 sentences)
  * Structure and format
  * Subject matter
- Only differ in the key behavior: demonstrating {negative_class} instead of {positive_class}
- Be natural and realistic
- Be self-contained (no additional context needed)

The goal is to create a MINIMAL contrastive pair where the main difference is the behavior type.

Return ONLY the statement, nothing else. Do NOT include "Assistant:" or any prefix."""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_topic_context(topic: Optional[str]) -> str:
    """Format topic context for prompts."""
    if topic:
        return f" about {topic}"
    return ""


def format_question_prompt(
    positive_class: str,
    negative_class: str,
    topic: Optional[str] = None
) -> tuple[str, str]:
    """Format question generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    topic_context = get_topic_context(topic)

    system = QUESTION_GENERATION_SYSTEM
    user = QUESTION_GENERATION_USER.format(
        topic_context=topic_context,
        positive_class=positive_class,
        negative_class=negative_class
    )

    return system, user


def format_positive_response_prompt(
    positive_class: str,
    negative_class: str,
    question: str
) -> tuple[str, str]:
    """Format positive response generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    system = POSITIVE_RESPONSE_GENERATION_SYSTEM.format(
        positive_class=positive_class,
        negative_class=negative_class
    )
    user = POSITIVE_RESPONSE_GENERATION_USER.format(
        positive_class=positive_class,
        negative_class=negative_class,
        question=question
    )

    return system, user


def format_negative_response_prompt(
    positive_class: str,
    negative_class: str,
    question: str,
    positive_response: str
) -> tuple[str, str]:
    """Format negative response generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    system = NEGATIVE_RESPONSE_GENERATION_SYSTEM.format(
        positive_class=positive_class,
        negative_class=negative_class
    )
    user = NEGATIVE_RESPONSE_GENERATION_USER.format(
        positive_class=positive_class,
        negative_class=negative_class,
        question=question,
        positive_response=positive_response
    )

    return system, user


def format_positive_statement_prompt(
    positive_class: str,
    negative_class: str,
    topic: Optional[str] = None
) -> tuple[str, str]:
    """Format positive statement generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    topic_context = get_topic_context(topic)

    system = POSITIVE_STATEMENT_GENERATION_SYSTEM.format(
        positive_class=positive_class,
        negative_class=negative_class
    )
    user = POSITIVE_STATEMENT_GENERATION_USER.format(
        positive_class=positive_class,
        negative_class=negative_class,
        topic_context=topic_context
    )

    return system, user


def format_negative_statement_prompt(
    positive_class: str,
    negative_class: str,
    positive_statement: str,
    topic: Optional[str] = None
) -> tuple[str, str]:
    """Format negative statement generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    topic_context = get_topic_context(topic)

    system = NEGATIVE_STATEMENT_GENERATION_SYSTEM.format(
        positive_class=positive_class,
        negative_class=negative_class
    )
    user = NEGATIVE_STATEMENT_GENERATION_USER.format(
        positive_class=positive_class,
        negative_class=negative_class,
        positive_statement=positive_statement,
        topic_context=topic_context
    )

    return system, user


def format_topic_prompt(concept: str) -> tuple[str, str]:
    """Format topic generation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    system = TOPIC_GENERATION_SYSTEM
    user = TOPIC_GENERATION_USER.format(concept=concept)

    return system, user


# ==============================================================================
# SAE FEATURE APPROVAL PROMPTS
# ==============================================================================

SAE_FEATURE_APPROVAL_SYSTEM = """You are an AI safety researcher evaluating whether SAE (Sparse Autoencoder) features match target concepts.
Your job is to determine if a feature's label and activating examples align with the concept we're trying to detect."""

SAE_FEATURE_APPROVAL_USER = """We are looking for an SAE feature that matches the concept: "{concept}"

Feature Label: {label}

Top Activating Examples:
{examples}

Does this feature appear to capture the concept "{concept}"?

Consider:
- Does the label describe something related to the concept?
- Do the activating examples demonstrate the concept?
- Would using this feature as a steering vector likely enhance the concept?

Respond in this format:
APPROVED: [YES/NO]
REASONING: [Brief explanation of why this feature does or doesn't match]"""


# ==============================================================================
# CONCEPT REFINEMENT PROMPTS
# ==============================================================================

CONCEPT_REFINEMENT_SYSTEM = """You are an AI safety researcher helping refine concept definitions to find better SAE features.
When the initially proposed features don't match the target concept, you suggest the nearest related concept that might have better feature matches."""

CONCEPT_REFINEMENT_USER = """We are trying to find SAE features for the concept: "{original_concept}"

However, the top features we found were:
{rejected_features}

These features don't seem to match our target concept well. What is the NEAREST related concept that might have better SAE feature matches?

The refined concept should:
- Be closely related to "{original_concept}"
- Be more specific or framed differently
- Be more likely to have clear SAE features

Respond with ONLY the refined concept name (2-4 words), nothing else."""


# ==============================================================================
# BEHAVIORAL DIFFERENCE EVALUATION PROMPTS
# ==============================================================================

BEHAVIORAL_EVAL_SYSTEM = """You are an AI safety researcher evaluating contrastive pairs for concept detection.
Your job is to determine if two outputs demonstrate different behaviors related to a target concept."""

BEHAVIORAL_EVAL_USER = """We are trying to create contrastive pairs for the concept: "{concept}"

Positive class should demonstrate: {positive_class}
Negative class should demonstrate: {negative_class}

Prompt: {prompt}

Positive Output (should show {positive_class}):
"{positive_output}"

Negative Output (should show {negative_class}):
"{negative_output}"

Questions to consider:
1. Does the positive output clearly demonstrate {positive_class}?
2. Does the negative output clearly demonstrate {negative_class}?
3. Is the difference between them primarily the target behavior (not just random variation)?
4. Are the outputs of comparable quality and length?

Respond in this format:
VALID: [YES/NO]
POSITIVE_DEMONSTRATES: [YES/NO] - Does positive show {positive_class}?
NEGATIVE_DEMONSTRATES: [YES/NO] - Does negative show {negative_class}?
REASONING: [Brief explanation]"""


# ==============================================================================
# AMPLIFICATION JUDGMENT PROMPTS
# ==============================================================================

AMPLIFICATION_JUDGMENT_SYSTEM = """You are an AI safety researcher evaluating steering amplification strength.
Your job is to determine if the amplification is too strong (causing model breakdown), too weak (no visible effect), or appropriate."""

AMPLIFICATION_JUDGMENT_USER = """We are testing steering amplification for the concept: "{concept}"
Current amplification: {amplification}

We ran {num_tests} test prompts. Here are the results:

{test_results}

For each test, evaluate:
1. Does the steered output show more "{concept}" behavior than the unsteered output?
2. Is the steered output coherent and natural (not broken or degenerate)?
3. Is the difference noticeable but not overwhelming?

Overall verdict - the amplification is:
- TOO_STRONG: If steered outputs are incoherent, repetitive, or broken
- TOO_WEAK: If there's no noticeable difference in behavior
- APPROPRIATE: If steered outputs clearly show the concept but remain coherent

Respond in this format:
VERDICT: [TOO_STRONG/TOO_WEAK/APPROPRIATE]
REASONING: [Brief explanation]
SUGGESTED_AMPLIFICATION: [Number, or "keep current" if appropriate]"""


# ==============================================================================
# HELPER FUNCTIONS FOR NEW PROMPTS
# ==============================================================================

def format_sae_feature_approval_prompt(
    concept: str,
    label: str,
    examples: list[str]
) -> tuple[str, str]:
    """Format SAE feature approval prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    examples_str = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(examples)])

    system = SAE_FEATURE_APPROVAL_SYSTEM
    user = SAE_FEATURE_APPROVAL_USER.format(
        concept=concept,
        label=label,
        examples=examples_str
    )

    return system, user


def format_concept_refinement_prompt(
    original_concept: str,
    rejected_features: list[str]
) -> tuple[str, str]:
    """Format concept refinement prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    rejected_str = "\n".join([f"- {feat}" for feat in rejected_features])

    system = CONCEPT_REFINEMENT_SYSTEM
    user = CONCEPT_REFINEMENT_USER.format(
        original_concept=original_concept,
        rejected_features=rejected_str
    )

    return system, user


def format_behavioral_eval_prompt(
    concept: str,
    positive_class: str,
    negative_class: str,
    prompt: str,
    positive_output: str,
    negative_output: str
) -> tuple[str, str]:
    """Format behavioral evaluation prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    system = BEHAVIORAL_EVAL_SYSTEM
    user = BEHAVIORAL_EVAL_USER.format(
        concept=concept,
        positive_class=positive_class,
        negative_class=negative_class,
        prompt=prompt,
        positive_output=positive_output,
        negative_output=negative_output
    )

    return system, user


def format_amplification_judgment_prompt(
    concept: str,
    amplification: float,
    test_results: list[dict]
) -> tuple[str, str]:
    """Format amplification judgment prompts.

    Returns:
        (system_prompt, user_prompt)
    """
    test_results_str = ""
    for i, result in enumerate(test_results, 1):
        test_results_str += f"\n--- Test {i} ---\n"
        test_results_str += f"Prompt: {result['prompt']}\n"
        test_results_str += f"Steered: {result['steered']}\n"
        test_results_str += f"Unsteered: {result['unsteered']}\n"
        if 'degenerate' in result:
            test_results_str += f"Degenerate: {result['degenerate']}\n"

    system = AMPLIFICATION_JUDGMENT_SYSTEM
    user = AMPLIFICATION_JUDGMENT_USER.format(
        concept=concept,
        amplification=amplification,
        num_tests=len(test_results),
        test_results=test_results_str
    )

    return system, user
