"""Quality validator using SAE features from Goodfire API."""

from typing import Dict, List, Any, Optional
from .goodfire_client import GoodfireClient
from .sae_manager import SAEManager
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
        metadata: Optional[Dict[str, Any]] = None,
        cosine_positive_features: Optional[List[Dict[str, Any]]] = None,
        cosine_negative_features: Optional[List[Dict[str, Any]]] = None,
        contrastive_positive_features: Optional[List[Dict[str, Any]]] = None,
        contrastive_negative_features: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize validation result.

        Args:
            pair_id: ID of the pair
            positive_features: Top features from positive example (Goodfire sparse codes)
            negative_features: Top features from negative example (Goodfire sparse codes)
            expected_positive_features: Expected features for positive class
            expected_negative_features: Expected features for negative class
            metadata: Additional metadata
            cosine_positive_features: Top features by cosine similarity (positive)
            cosine_negative_features: Top features by cosine similarity (negative)
            contrastive_positive_features: Features steering toward positive class (positive similarity)
            contrastive_negative_features: Features steering toward negative class (negative similarity)
        """
        self.pair_id = pair_id
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.expected_positive_features = expected_positive_features
        self.expected_negative_features = expected_negative_features
        self.metadata = metadata or {}

        # Cosine similarity features
        self.cosine_positive_features = cosine_positive_features or []
        self.cosine_negative_features = cosine_negative_features or []

        # Contrastive features (steering vector)
        self.contrastive_positive_features = contrastive_positive_features or []
        self.contrastive_negative_features = contrastive_negative_features or []

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
            'cosine_positive_features': self.cosine_positive_features,
            'cosine_negative_features': self.cosine_negative_features,
            'contrastive_positive_features': self.contrastive_positive_features,
            'contrastive_negative_features': self.contrastive_negative_features,
            'metadata': self.metadata
        }


class QualityValidator:
    """Validates quality of generated contrastive pairs using SAE features."""

    def __init__(
        self,
        goodfire_client: GoodfireClient,
        positive_queries: List[str],
        negative_queries: List[str],
        top_k: int = 10,
        sae_manager: Optional[SAEManager] = None,
        sae_top_k: int = 20,
        sae_aggregation: str = "mean",
        neuronpedia_client: Optional[Any] = None
    ):
        """Initialize quality validator.

        Args:
            goodfire_client: Goodfire client for feature analysis
            positive_queries: Queries for positive class features
            negative_queries: Queries for negative class features
            top_k: Number of top features to analyze (for Goodfire sparse codes)
            sae_manager: SAE manager for cosine similarity computation
            sae_top_k: Number of top features by cosine similarity
            sae_aggregation: How to aggregate cosine sims across tokens
            neuronpedia_client: Neuronpedia client for fetching feature labels
        """
        self.goodfire_client = goodfire_client
        self.positive_queries = positive_queries
        self.negative_queries = negative_queries
        self.top_k = top_k
        self.sae_manager = sae_manager
        self.sae_top_k = sae_top_k
        self.sae_aggregation = sae_aggregation
        self.neuronpedia_client = neuronpedia_client

        # Find expected features for both classes using Neuronpedia semantic search
        print("\n" + "="*60)
        print("Finding expected SAE features for validation...")
        print("="*60)
        concept_features = self.neuronpedia_client.find_concept_features(
            positive_queries=positive_queries,
            negative_queries=negative_queries,
            top_k_per_query=5
        )

        self.expected_positive_features = concept_features['positive_features']
        self.expected_negative_features = concept_features['negative_features']

        print(f"\nFound {len(self.expected_positive_features)} positive class features")
        print(f"Found {len(self.expected_negative_features)} negative class features")

        if len(self.expected_positive_features) == 0 and len(self.expected_negative_features) == 0:
            print("\n⚠️  WARNING: No features found for either class!")
            print("This may indicate an issue with:")
            print("  - Neuronpedia API connectivity or authentication")
            print("  - Model/layer configuration not matching Neuronpedia's available features")
            print("  - Search queries not matching any features in the model")

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

        # Analyze with Goodfire API (sparse codes) - make this optional
        try:
            results = self.goodfire_client.analyze_contrastive_pair(
                positive_messages=positive_messages,
                negative_messages=negative_messages,
                top_k=self.top_k
            )
        except Exception as e:
            print(f"\nWarning: Goodfire API inspect failed: {e}")
            print("Continuing with only cosine similarities...")
            results = {
                'positive': {'top_features': []},
                'negative': {'top_features': []}
            }

        # Compute cosine similarities if SAE manager is available
        cosine_positive_features = []
        cosine_negative_features = []

        if self.sae_manager and pair.positive_activations is not None and pair.negative_activations is not None:
            print("\nComputing cosine similarities with SAE decoder weights...")

            # Compute cosine similarities for positive example
            positive_cosine_results = self.sae_manager.compute_cosine_similarities(
                activations=pair.positive_activations,
                top_k=self.sae_top_k,
                aggregate=self.sae_aggregation
            )

            # Compute cosine similarities for negative example
            negative_cosine_results = self.sae_manager.compute_cosine_similarities(
                activations=pair.negative_activations,
                top_k=self.sae_top_k,
                aggregate=self.sae_aggregation
            )

            # Fetch human-interpretable labels from Neuronpedia
            # Collect all unique feature indices
            positive_indices = [idx for idx, _ in positive_cosine_results]
            negative_indices = [idx for idx, _ in negative_cosine_results]
            all_indices = list(set(positive_indices + negative_indices))

            # Fetch labels from Neuronpedia (or use placeholders)
            if self.neuronpedia_client:
                try:
                    labels = self.neuronpedia_client.get_feature_labels(all_indices)
                except Exception as e:
                    print(f"Warning: Failed to fetch labels from Neuronpedia: {e}")
                    labels = {idx: f'Feature {idx}' for idx in all_indices}
            else:
                print("Warning: No Neuronpedia client provided, using placeholder labels")
                labels = {idx: f'Feature {idx}' for idx in all_indices}

            # Build feature dictionaries with real labels
            for feature_idx, cosine_sim in positive_cosine_results:
                cosine_positive_features.append({
                    'index': feature_idx,
                    'cosine_similarity': cosine_sim,
                    'label': labels.get(feature_idx, f'Feature {feature_idx}')
                })

            for feature_idx, cosine_sim in negative_cosine_results:
                cosine_negative_features.append({
                    'index': feature_idx,
                    'cosine_similarity': cosine_sim,
                    'label': labels.get(feature_idx, f'Feature {feature_idx}')
                })

            print(f"  Top cosine similarity (positive): {positive_cosine_results[0][1]:.4f} (feature {positive_cosine_results[0][0]})")
            print(f"  Top cosine similarity (negative): {negative_cosine_results[0][1]:.4f} (feature {negative_cosine_results[0][0]})")

        # Compute contrastive features (steering vector)
        contrastive_positive_features = []
        contrastive_negative_features = []

        if self.sae_manager and pair.positive_activations is not None and pair.negative_activations is not None:
            print("\nComputing contrastive features (steering vector: positive - negative)...")

            # Compute cosine similarities to the difference vector
            # Returns dict with 'positive' and 'negative' keys
            contrastive_results = self.sae_manager.compute_contrastive_cosine_similarities(
                positive_activations=pair.positive_activations,
                negative_activations=pair.negative_activations,
                top_k=self.sae_top_k,
                aggregate=self.sae_aggregation
            )

            # Collect all contrastive feature indices (for label fetching)
            positive_contrastive_indices = [idx for idx, _ in contrastive_results['positive']]
            negative_contrastive_indices = [idx for idx, _ in contrastive_results['negative']]
            all_contrastive_indices = positive_contrastive_indices + negative_contrastive_indices

            # Fetch labels if not already fetched
            if self.neuronpedia_client:
                # Get any missing labels
                missing_indices = [idx for idx in all_contrastive_indices if idx not in labels]
                if missing_indices:
                    try:
                        new_labels = self.neuronpedia_client.get_feature_labels(missing_indices)
                        labels.update(new_labels)
                    except Exception as e:
                        print(f"Warning: Failed to fetch some labels from Neuronpedia: {e}")
                        for idx in missing_indices:
                            if idx not in labels:
                                labels[idx] = f'Feature {idx}'

            # Build contrastive positive features (steer toward positive class)
            for feature_idx, cosine_sim in contrastive_results['positive']:
                contrastive_positive_features.append({
                    'index': feature_idx,
                    'cosine_similarity': cosine_sim,
                    'label': labels.get(feature_idx, f'Feature {feature_idx}')
                })

            # Build contrastive negative features (steer toward negative class)
            for feature_idx, cosine_sim in contrastive_results['negative']:
                contrastive_negative_features.append({
                    'index': feature_idx,
                    'cosine_similarity': cosine_sim,
                    'label': labels.get(feature_idx, f'Feature {feature_idx}')
                })

            print(f"  Top positive direction feature: {contrastive_results['positive'][0][1]:.4f} (feature {contrastive_results['positive'][0][0]})")
            print(f"    Label: {labels.get(contrastive_results['positive'][0][0], 'N/A')}")
            print(f"  Top negative direction feature: {contrastive_results['negative'][0][1]:.4f} (feature {contrastive_results['negative'][0][0]})")
            print(f"    Label: {labels.get(contrastive_results['negative'][0][0], 'N/A')}")

        # Create validation result
        validation_result = ValidationResult(
            pair_id=pair_id,
            positive_features=results['positive']['top_features'],
            negative_features=results['negative']['top_features'],
            expected_positive_features=self.expected_positive_features,
            expected_negative_features=self.expected_negative_features,
            metadata=pair.metadata,
            cosine_positive_features=cosine_positive_features,
            cosine_negative_features=cosine_negative_features,
            contrastive_positive_features=contrastive_positive_features,
            contrastive_negative_features=contrastive_negative_features
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

    def check_activation_threshold(
        self,
        pair: ContrastivePair,
        min_cosine_distance: float = 0.1
    ) -> tuple[bool, float]:
        """
        Check if activation difference between positive and negative exceeds threshold.

        Args:
            pair: ContrastivePair to check
            min_cosine_distance: Minimum cosine distance required

        Returns:
            Tuple of (passes_threshold: bool, actual_distance: float)
        """
        if pair.positive_activations is None or pair.negative_activations is None:
            return False, 0.0

        # Normalize activations
        pos_norm = pair.positive_activations / (np.linalg.norm(pair.positive_activations) + 1e-10)
        neg_norm = pair.negative_activations / (np.linalg.norm(pair.negative_activations) + 1e-10)

        # Compute cosine similarity
        cosine_sim = np.dot(pos_norm, neg_norm)

        # Cosine distance = 1 - similarity
        cosine_distance = 1.0 - cosine_sim

        passes = cosine_distance >= min_cosine_distance

        return passes, float(cosine_distance)

    def validate_with_llm_judge(
        self,
        pair: ContrastivePair,
        llm_agent: Any,
        concept: str,
        positive_class: str,
        negative_class: str
    ) -> tuple[bool, dict]:
        """
        Validate pair using LLM judge to evaluate behavioral differences.

        Args:
            pair: ContrastivePair to validate
            llm_agent: LLMAgent instance with evaluate_behavioral_difference method
            concept: The overall concept being tested
            positive_class: Expected positive class behavior
            negative_class: Expected negative class behavior

        Returns:
            Tuple of (is_valid: bool, judgment_details: dict)
        """
        # Extract prompt and outputs based on format
        if pair.format_type == "statement":
            prompt = "Generate a statement:"
            positive_output = pair.positive_text
            negative_output = pair.negative_text
        elif pair.format_type == "qa":
            # Parse Q&A format
            positive_parts = pair.positive_text.split('\n', 1)
            if len(positive_parts) >= 2:
                prompt = positive_parts[0].replace("User: ", "").strip()
                positive_output = positive_parts[1].replace("Assistant: ", "").strip()
            else:
                prompt = pair.prompt if hasattr(pair, 'prompt') else ""
                positive_output = pair.positive_text

            negative_parts = pair.negative_text.split('\n', 1)
            if len(negative_parts) >= 2:
                negative_output = negative_parts[1].replace("Assistant: ", "").strip()
            else:
                negative_output = pair.negative_text
        else:
            # Use pair.prompt if available (for on-policy generation)
            prompt = pair.prompt if hasattr(pair, 'prompt') else "Generate text:"
            positive_output = pair.positive_class if hasattr(pair, 'positive_class') else pair.positive_text
            negative_output = pair.negative_class if hasattr(pair, 'negative_class') else pair.negative_text

        # Use LLM agent to evaluate
        is_valid, details = llm_agent.evaluate_behavioral_difference(
            concept=concept,
            positive_class=positive_class,
            negative_class=negative_class,
            prompt=prompt,
            positive_output=positive_output,
            negative_output=negative_output
        )

        return is_valid, details

    def comprehensive_validation(
        self,
        pair: ContrastivePair,
        llm_agent: Any,
        concept: str,
        positive_class: str,
        negative_class: str,
        min_cosine_distance: float = 0.1,
        require_llm_approval: bool = True,
        require_threshold: bool = True
    ) -> tuple[bool, dict]:
        """
        Perform comprehensive validation combining activation threshold and LLM judgment.

        Args:
            pair: ContrastivePair to validate
            llm_agent: LLMAgent instance
            concept: The overall concept being tested
            positive_class: Expected positive class behavior
            negative_class: Expected negative class behavior
            min_cosine_distance: Minimum cosine distance required
            require_llm_approval: Whether LLM approval is required
            require_threshold: Whether threshold check is required

        Returns:
            Tuple of (is_valid: bool, validation_details: dict)
        """
        validation_details = {}

        # Check activation threshold
        passes_threshold, cosine_distance = self.check_activation_threshold(
            pair, min_cosine_distance
        )
        validation_details['passes_threshold'] = passes_threshold
        validation_details['cosine_distance'] = cosine_distance
        validation_details['min_threshold'] = min_cosine_distance

        # LLM judgment
        llm_approved, llm_details = self.validate_with_llm_judge(
            pair, llm_agent, concept, positive_class, negative_class
        )
        validation_details['llm_approved'] = llm_approved
        validation_details['llm_details'] = llm_details

        # Determine overall validity
        is_valid = True
        if require_threshold and not passes_threshold:
            is_valid = False
            validation_details['failure_reason'] = 'threshold_not_met'
        if require_llm_approval and not llm_approved:
            is_valid = False
            validation_details['failure_reason'] = 'llm_disapproval'

        validation_details['is_valid'] = is_valid

        return is_valid, validation_details
