"""Quality validator using SAE features from Goodfire API."""

from typing import Dict, List, Any, Optional
from .goodfire_client import GoodfireClient
from .data_generator import ContrastivePair
import numpy as np


class ValidationResult:
    """Results from validating a contrastive pair."""

    def __init__(
        self,
        pair_id: int,
        positive_features: List[Dict[str, Any]],
        negative_features: List[Dict[str, Any]],
        expected_positive_features: List[Dict[str, Any]],
        expected_negative_features: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation result.

        Args:
            pair_id: ID of the pair
            positive_features: Top features from positive example
            negative_features: Top features from negative example
            expected_positive_features: Expected features for positive class
            expected_negative_features: Expected features for negative class
            metadata: Additional metadata
        """
        self.pair_id = pair_id
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.expected_positive_features = expected_positive_features
        self.expected_negative_features = expected_negative_features
        self.metadata = metadata or {}

        # Compute quality metrics
        self._compute_quality_metrics()

    def _compute_quality_metrics(self) -> None:
        """Compute quality metrics based on feature overlap."""
        # Extract feature IDs
        positive_feature_ids = {f['id'] for f in self.positive_features}
        negative_feature_ids = {f['id'] for f in self.negative_features}
        expected_pos_ids = {f['id'] for f in self.expected_positive_features}
        expected_neg_ids = {f['id'] for f in self.expected_negative_features}

        # Check overlap with expected features
        self.positive_overlap = len(positive_feature_ids & expected_pos_ids)
        self.negative_overlap = len(negative_feature_ids & expected_neg_ids)

        # Quality score (simple heuristic)
        self.quality_score = (self.positive_overlap + self.negative_overlap) / 2.0

    def print_summary(self) -> None:
        """Print a summary of validation results."""
        print(f"\n{'='*60}")
        print(f"Validation Results for Pair {self.pair_id}")
        print(f"{'='*60}")
        print(f"Quality Score: {self.quality_score:.2f}")
        print(f"Positive overlap with expected features: {self.positive_overlap}")
        print(f"Negative overlap with expected features: {self.negative_overlap}")

        print(f"\nTop features in POSITIVE example:")
        for i, feat in enumerate(self.positive_features[:5], 1):
            label = feat.get('label', 'N/A')
            strength = feat.get('activation_strength', 'N/A')
            expected = "✓" if feat['id'] in {f['id'] for f in self.expected_positive_features} else " "
            print(f"  [{expected}] {i}. {label} (strength: {strength})")

        print(f"\nTop features in NEGATIVE example:")
        for i, feat in enumerate(self.negative_features[:5], 1):
            label = feat.get('label', 'N/A')
            strength = feat.get('activation_strength', 'N/A')
            expected = "✓" if feat['id'] in {f['id'] for f in self.expected_negative_features} else " "
            print(f"  [{expected}] {i}. {label} (strength: {strength})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pair_id': self.pair_id,
            'quality_score': self.quality_score,
            'positive_overlap': self.positive_overlap,
            'negative_overlap': self.negative_overlap,
            'positive_features': self.positive_features,
            'negative_features': self.negative_features,
            'metadata': self.metadata
        }


class QualityValidator:
    """Validates quality of generated contrastive pairs using SAE features."""

    def __init__(
        self,
        goodfire_client: GoodfireClient,
        positive_queries: List[str],
        negative_queries: List[str],
        top_k: int = 10
    ):
        """Initialize quality validator.

        Args:
            goodfire_client: Goodfire client for feature analysis
            positive_queries: Queries for positive class features
            negative_queries: Queries for negative class features
            top_k: Number of top features to analyze
        """
        self.goodfire_client = goodfire_client
        self.positive_queries = positive_queries
        self.negative_queries = negative_queries
        self.top_k = top_k

        # Find expected features for both classes
        print("\n" + "="*60)
        print("Finding expected SAE features for validation...")
        print("="*60)
        concept_features = self.goodfire_client.find_concept_features(
            positive_queries=positive_queries,
            negative_queries=negative_queries,
            top_k_per_query=5
        )

        self.expected_positive_features = concept_features['positive_features']
        self.expected_negative_features = concept_features['negative_features']

        print(f"\nFound {len(self.expected_positive_features)} positive class features")
        print(f"Found {len(self.expected_negative_features)} negative class features")

        print("\nExpected POSITIVE features:")
        for i, feat in enumerate(self.expected_positive_features[:10], 1):
            print(f"  {i}. {feat['label']}")

        print("\nExpected NEGATIVE features:")
        for i, feat in enumerate(self.expected_negative_features[:10], 1):
            print(f"  {i}. {feat['label']}")

    def validate_pair(
        self,
        pair: ContrastivePair,
        pair_id: int
    ) -> ValidationResult:
        """Validate a single contrastive pair.

        Args:
            pair: ContrastivePair to validate
            pair_id: ID of the pair

        Returns:
            ValidationResult object
        """
        print(f"\n{'='*60}")
        print(f"Validating pair {pair_id} ({pair.format_type} format)")
        print(f"{'='*60}")

        # Format as messages for Goodfire API
        if pair.format_type == "statement":
            # For statements, create simple user-assistant exchanges
            positive_messages = [
                {"role": "user", "content": "Please make a statement."},
                {"role": "assistant", "content": pair.positive_text}
            ]
            negative_messages = [
                {"role": "user", "content": "Please make a statement."},
                {"role": "assistant", "content": pair.negative_text}
            ]
        elif pair.format_type == "qa":
            # Parse Q&A format
            positive_parts = pair.positive_text.split('\n', 1)
            negative_parts = pair.negative_text.split('\n', 1)

            question = positive_parts[0].replace("User: ", "").strip()
            positive_answer = positive_parts[1].replace("Assistant: ", "").strip()
            negative_answer = negative_parts[1].replace("Assistant: ", "").strip()

            positive_messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": positive_answer}
            ]
            negative_messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": negative_answer}
            ]
        else:
            raise ValueError(f"Unknown format: {pair.format_type}")

        # Analyze with Goodfire API
        results = self.goodfire_client.analyze_contrastive_pair(
            positive_messages=positive_messages,
            negative_messages=negative_messages,
            top_k=self.top_k
        )

        # Create validation result
        validation_result = ValidationResult(
            pair_id=pair_id,
            positive_features=results['positive']['top_features'],
            negative_features=results['negative']['top_features'],
            expected_positive_features=self.expected_positive_features,
            expected_negative_features=self.expected_negative_features,
            metadata=pair.metadata
        )

        # Print summary
        validation_result.print_summary()

        return validation_result

    def validate_pairs(
        self,
        pairs: List[ContrastivePair]
    ) -> List[ValidationResult]:
        """Validate multiple contrastive pairs.

        Args:
            pairs: List of ContrastivePair objects

        Returns:
            List of ValidationResult objects
        """
        results = []

        print(f"\n{'#'*60}")
        print(f"# VALIDATING {len(pairs)} CONTRASTIVE PAIRS")
        print(f"{'#'*60}")

        for i, pair in enumerate(pairs):
            result = self.validate_pair(pair, pair_id=i)
            results.append(result)

        # Print overall summary
        self._print_overall_summary(results)

        return results

    def _print_overall_summary(self, results: List[ValidationResult]) -> None:
        """Print overall summary of validation results."""
        print(f"\n{'#'*60}")
        print(f"# OVERALL VALIDATION SUMMARY")
        print(f"{'#'*60}")

        avg_quality = np.mean([r.quality_score for r in results])
        avg_pos_overlap = np.mean([r.positive_overlap for r in results])
        avg_neg_overlap = np.mean([r.negative_overlap for r in results])

        print(f"\nTotal pairs validated: {len(results)}")
        print(f"Average quality score: {avg_quality:.2f}")
        print(f"Average positive overlap: {avg_pos_overlap:.2f}")
        print(f"Average negative overlap: {avg_neg_overlap:.2f}")

        # Show best and worst pairs
        sorted_results = sorted(results, key=lambda r: r.quality_score, reverse=True)

        print(f"\nBest pair (ID {sorted_results[0].pair_id}): quality={sorted_results[0].quality_score:.2f}")
        print(f"Worst pair (ID {sorted_results[-1].pair_id}): quality={sorted_results[-1].quality_score:.2f}")
