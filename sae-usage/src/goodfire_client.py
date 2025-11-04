"""Goodfire API client for SAE feature analysis."""

import goodfire
from typing import List, Dict, Any, Optional


class GoodfireClient:
    """Client for interacting with Goodfire API for SAE feature analysis."""

    def __init__(
        self,
        api_key: str,
        model_variant: str = "meta-llama/Llama-3.1-8B-Instruct",
        layer: int = 19
    ):
        """Initialize Goodfire client.

        Args:
            api_key: Goodfire API key
            model_variant: Model variant to use
            layer: Layer to analyze
        """
        self.client = goodfire.Client(api_key=api_key)
        self.variant = goodfire.Variant(model_variant)
        self.model_variant = model_variant
        self.layer = layer

    def search_features(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for SAE features matching a query.

        Args:
            query: Search query (e.g., "deception", "honesty")
            top_k: Number of top features to return

        Returns:
            List of feature dictionaries with 'id', 'label', 'score'
        """
        print(f"Searching for features related to: {query}")
        features = self.client.features.search(query, model=self.model_variant, top_k=top_k)

        results = []
        for feature in features:
            results.append({
                'id': str(feature.uuid),
                'label': feature.label if hasattr(feature, 'label') else str(feature),
                'index_label': feature.index_label if hasattr(feature, 'index_label') else None,
                'score': feature.max_activation_strength if hasattr(feature, 'max_activation_strength') else None
            })

        return results

    def inspect_conversation(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Inspect SAE features activating in a conversation.

        Args:
            messages: Conversation messages (list of {"role": "user/assistant", "content": "..."})
            top_k: Number of top activating features to return

        Returns:
            Dictionary with top activating features and their strengths
        """
        # Use Goodfire's inspect API
        context = self.client.features.inspect(messages, model=self.model_variant)

        # Get top-k activating features
        top_features = context.top(k=top_k)

        results = {
            'top_features': [],
            'num_features': len(top_features)
        }

        for feature_activation in top_features:
            results['top_features'].append({
                'id': str(feature_activation.feature.uuid),
                'label': feature_activation.feature.label,
                'index_label': getattr(feature_activation.feature, 'index_label', None),
                'activation_strength': feature_activation.activation
            })

        return results

    def get_feature_neighbors(
        self,
        feature_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get neighboring features in embedding space.

        Args:
            feature_id: Feature UUID
            top_k: Number of neighbors to return

        Returns:
            List of neighboring features
        """
        # Note: This requires looking up the feature first
        # Implementation depends on Goodfire API structure
        neighbors = self.client.features.neighbors(feature_id, model=self.model_variant, top_k=top_k)

        results = []
        for neighbor in neighbors:
            results.append({
                'id': str(neighbor.uuid),
                'label': neighbor.label if hasattr(neighbor, 'label') else str(neighbor),
                'similarity': neighbor.similarity if hasattr(neighbor, 'similarity') else None
            })

        return results

    def analyze_contrastive_pair(
        self,
        positive_messages: List[Dict[str, str]],
        negative_messages: List[Dict[str, str]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Analyze feature differences between contrastive pairs.

        Args:
            positive_messages: Messages for positive class example
            negative_messages: Messages for negative class example
            top_k: Number of top features to return for each

        Returns:
            Dictionary with features for both examples
        """
        print("Analyzing positive example...")
        positive_features = self.inspect_conversation(positive_messages, top_k)

        print("Analyzing negative example...")
        negative_features = self.inspect_conversation(negative_messages, top_k)

        return {
            'positive': positive_features,
            'negative': negative_features
        }

    def find_concept_features(
        self,
        positive_queries: List[str],
        negative_queries: List[str],
        top_k_per_query: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find all features related to positive and negative concept queries.

        Args:
            positive_queries: List of queries for positive class (e.g., ["deception", "lying"])
            negative_queries: List of queries for negative class (e.g., ["honest", "truthful"])
            top_k_per_query: Top-k features per query

        Returns:
            Dictionary with 'positive_features' and 'negative_features'
        """
        positive_features = []
        negative_features = []

        print("\nSearching for positive class features...")
        for query in positive_queries:
            features = self.search_features(query, top_k_per_query)
            positive_features.extend(features)

        print("\nSearching for negative class features...")
        for query in negative_queries:
            features = self.search_features(query, top_k_per_query)
            negative_features.extend(features)

        # Remove duplicates based on feature ID
        positive_features = list({f['id']: f for f in positive_features}.values())
        negative_features = list({f['id']: f for f in negative_features}.values())

        return {
            'positive_features': positive_features,
            'negative_features': negative_features
        }
