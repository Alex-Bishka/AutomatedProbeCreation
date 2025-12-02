"""
Degeneracy detection for model outputs.

Detects when steering causes model outputs to become degenerate (repetitive,
incoherent, or broken) indicating that the amplification is too strong.
"""

from typing import Dict, Tuple
from collections import Counter
import re


def detect_repetition(text: str, max_repeat_ratio: float = 0.3) -> Tuple[bool, float]:
    """
    Detect repetitive patterns in text using n-gram analysis.

    Args:
        text: The text to analyze
        max_repeat_ratio: Maximum ratio of repeated content before flagging

    Returns:
        Tuple of (is_repetitive, repeat_ratio)
    """
    if not text or len(text.strip()) < 20:
        return False, 0.0

    # Tokenize into words
    words = text.lower().split()

    if len(words) < 10:
        return False, 0.0

    # Check for repeated phrases (3-grams, 4-grams, 5-grams)
    max_repeat_count = 0
    total_ngrams = 0

    for n in [3, 4, 5]:
        if len(words) < n:
            continue

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        total_ngrams += len(ngrams)

        # Find most repeated n-gram
        if ngram_counts:
            most_common_count = ngram_counts.most_common(1)[0][1]
            max_repeat_count = max(max_repeat_count, most_common_count)

    # Check for character-level repetition (e.g., "aaaaaaa")
    char_repeat_pattern = r'(.)\1{5,}'  # Same character repeated 6+ times
    if re.search(char_repeat_pattern, text):
        return True, 1.0

    # Check for word-level repetition (e.g., "the the the the")
    word_repeat_pattern = r'\b(\w+)\s+\1(\s+\1){2,}'  # Same word repeated 3+ times
    if re.search(word_repeat_pattern, text.lower()):
        return True, 1.0

    # Calculate repeat ratio
    repeat_ratio = max_repeat_count / max(total_ngrams, 1)

    is_repetitive = repeat_ratio > max_repeat_ratio

    return is_repetitive, repeat_ratio


def detect_incoherence(text: str) -> Tuple[bool, Dict[str, any]]:
    """
    Detect incoherent or broken text.

    Checks for signs of model breakdown:
    - Very short output (model gave up)
    - Excessive special characters
    - Broken formatting (unclosed brackets, quotes)
    - Gibberish (high ratio of non-dictionary words)

    Args:
        text: The text to analyze

    Returns:
        Tuple of (is_incoherent, metrics_dict)
    """
    if not text:
        return True, {'reason': 'empty_output'}

    text = text.strip()

    metrics = {
        'length': len(text),
        'word_count': len(text.split()),
        'special_char_ratio': 0.0,
        'bracket_balanced': True,
        'quote_balanced': True,
        'reason': None
    }

    # Check if too short (model might have failed)
    if len(text) < 10:
        metrics['reason'] = 'too_short'
        return True, metrics

    # Check special character ratio
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    metrics['special_char_ratio'] = special_chars / len(text)

    if metrics['special_char_ratio'] > 0.5:  # More than 50% special chars
        metrics['reason'] = 'excessive_special_chars'
        return True, metrics

    # Check bracket balance
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for char in text:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                metrics['bracket_balanced'] = False
                break

    if stack:  # Unclosed brackets
        metrics['bracket_balanced'] = False

    # Check quote balance
    single_quotes = text.count("'")
    double_quotes = text.count('"')
    metrics['quote_balanced'] = (single_quotes % 2 == 0) and (double_quotes % 2 == 0)

    # Overall incoherence check
    is_incoherent = (
        not metrics['bracket_balanced'] or
        not metrics['quote_balanced']
    )

    if is_incoherent and not metrics['reason']:
        metrics['reason'] = 'unbalanced_delimiters'

    return is_incoherent, metrics


def detect_truncation(text: str) -> Tuple[bool, str]:
    """
    Detect if output was truncated mid-sentence.

    Args:
        text: The text to analyze

    Returns:
        Tuple of (is_truncated, reason)
    """
    if not text:
        return True, 'empty'

    text = text.strip()

    # Check if ends with sentence-ending punctuation
    if text[-1] in '.!?':
        return False, 'complete_sentence'

    # Check if ends mid-word (very suspicious)
    if text[-1].isalnum() and len(text) > 1 and not text[-2].isspace():
        # Might be mid-word
        words = text.split()
        if words and len(words[-1]) < 3:
            return True, 'mid_word'

    # Check if ends with comma or conjunction (might be incomplete)
    if text[-1] in ',;:' or text.split()[-1].lower() in ['and', 'or', 'but', 'the', 'a', 'an']:
        return True, 'incomplete_thought'

    return False, 'possibly_complete'


def is_degenerate(
    text: str,
    check_repetition: bool = True,
    check_incoherence: bool = True,
    check_truncation: bool = True,
    verbose: bool = False
) -> Tuple[bool, Dict[str, any]]:
    """
    Combined degeneracy check.

    Args:
        text: The text to analyze
        check_repetition: Whether to check for repetition
        check_incoherence: Whether to check for incoherence
        check_truncation: Whether to check for truncation
        verbose: Whether to return detailed metrics

    Returns:
        Tuple of (is_degenerate, details_dict)
    """
    details = {
        'is_degenerate': False,
        'reasons': [],
        'repetition': None,
        'incoherence': None,
        'truncation': None
    }

    if check_repetition:
        is_repetitive, repeat_ratio = detect_repetition(text)
        details['repetition'] = {
            'is_repetitive': is_repetitive,
            'repeat_ratio': repeat_ratio
        }
        if is_repetitive:
            details['is_degenerate'] = True
            details['reasons'].append('repetitive')

    if check_incoherence:
        is_incoherent, metrics = detect_incoherence(text)
        details['incoherence'] = metrics
        if is_incoherent:
            details['is_degenerate'] = True
            details['reasons'].append(f"incoherent ({metrics['reason']})")

    if check_truncation:
        is_truncated, reason = detect_truncation(text)
        details['truncation'] = {
            'is_truncated': is_truncated,
            'reason': reason
        }
        if is_truncated and reason in ['empty', 'mid_word']:
            # Only flag severe truncation as degenerate
            details['is_degenerate'] = True
            details['reasons'].append(f"truncated ({reason})")

    if verbose:
        return details['is_degenerate'], details
    else:
        return details['is_degenerate'], {'reasons': details['reasons']}


def compare_outputs(
    steered_text: str,
    unsteered_text: str
) -> Dict[str, any]:
    """
    Compare steered vs unsteered outputs for degeneracy.

    Args:
        steered_text: Output with steering applied
        unsteered_text: Output without steering

    Returns:
        Dict with comparison metrics and degeneracy flags
    """
    steered_degenerate, steered_details = is_degenerate(steered_text, verbose=True)
    unsteered_degenerate, unsteered_details = is_degenerate(unsteered_text, verbose=True)

    # Calculate relative quality
    steered_len = len(steered_text.split())
    unsteered_len = len(unsteered_text.split())

    length_ratio = steered_len / max(unsteered_len, 1)

    return {
        'steered_degenerate': steered_degenerate,
        'unsteered_degenerate': unsteered_degenerate,
        'steered_reasons': steered_details['reasons'],
        'unsteered_reasons': unsteered_details['reasons'],
        'length_ratio': length_ratio,
        'steered_details': steered_details,
        'unsteered_details': unsteered_details,
        'verdict': _get_verdict(steered_degenerate, unsteered_degenerate, length_ratio)
    }


def _get_verdict(
    steered_degenerate: bool,
    unsteered_degenerate: bool,
    length_ratio: float
) -> str:
    """
    Get overall verdict on steering quality.

    Args:
        steered_degenerate: Whether steered output is degenerate
        unsteered_degenerate: Whether unsteered output is degenerate
        length_ratio: Ratio of steered to unsteered length

    Returns:
        Verdict string
    """
    if steered_degenerate and not unsteered_degenerate:
        return "steering_caused_degeneracy"
    elif steered_degenerate and unsteered_degenerate:
        return "both_degenerate"
    elif length_ratio < 0.3:
        return "steering_suppressed_output"
    elif length_ratio > 3.0:
        return "steering_inflated_output"
    else:
        return "acceptable"
