"""Neuronpedia client for fetching SAE feature labels via live API."""

import json
import os
import requests
import time
from typing import Dict, List, Optional
from pathlib import Path


class NeuronpediaClient:
    """Client for fetching SAE feature labels from Neuronpedia live API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "llama3.1-8b-it",
        layer_id: str = "19-resid-post-aa",
        cache_dir: Optional[str] = None
    ):
        """Initialize Neuronpedia API client.

        Args:
            api_key: Neuronpedia API key
            model_id: Model identifier for API (e.g., "llama3.1-8b-it")
            layer_id: Layer/SAE identifier for API (e.g., "19-resid-post-aa")
            cache_dir: Optional directory to cache API responses (for future enhancement)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.layer_id = layer_id
        self.base_url = "https://www.neuronpedia.org"

        # In-memory caches
        self.feature_cache: Dict[int, Dict] = {}  # index → feature info
        self.search_cache: Dict[str, List[Dict]] = {}  # query → search results

        # Optional persistent cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Neuronpedia API Client Initialized")
        print(f"{'='*60}")
        print(f"Model: {self.model_id}")
        print(f"Layer/SAE: {self.layer_id}")
        print(f"Mode: Live API with dynamic caching")
        print(f"{'='*60}\n")

    def search_features(
        self,
        query: str,
        top_k: int = 100,
        fetch_all_pages: bool = False
    ) -> List[Dict]:
        """Search for features by keyword using Neuronpedia API.

        Args:
            query: Keyword to search for (e.g., "deception", "honesty")
            top_k: Number of top results to return
            fetch_all_pages: If True, fetch all available results beyond first 20

        Returns:
            List of feature dicts with keys: index, description, cosine_similarity
        """
        # Check search cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self.search_cache:
            print(f"  Using cached search results for '{query}'")
            return self.search_cache[cache_key]

        print(f"Searching Neuronpedia for '{query}'...")
        results = []
        offset = 0

        while True:
            try:
                # Build request payload
                payload = {
                    "modelId": self.model_id,
                    "layers": [self.layer_id],
                    "query": query
                }

                # Only include offset for pagination (subsequent pages)
                if offset > 0:
                    payload["offset"] = offset

                response = requests.post(
                    f"{self.base_url}/api/explanation/search",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key
                    },
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Extract feature info from results
                for result in data.get('results', []):
                    feature_info = {
                        'index': int(result['index']),
                        'description': result.get('description', f"Feature {result['index']}"),
                        'cosine_similarity': result.get('cosine_similarity', 0.0),
                        'full_data': result.get('neuron', {})
                    }
                    results.append(feature_info)

                    # Add to feature cache for future lookups
                    self.feature_cache[feature_info['index']] = feature_info

                # Check if we should continue pagination
                has_more = data.get('hasMore', False)

                # Stop if: no more pages OR we have enough results (unless fetch_all_pages=True)
                if not has_more:
                    break
                if len(results) >= top_k and not fetch_all_pages:
                    break

                offset = data.get('nextOffset', offset + 20)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"  Warning: Search API call failed: {e}")
                break

        # Trim to top_k and cache
        results = results[:top_k]
        self.search_cache[cache_key] = results

        print(f"  Found {len(results)} features for '{query}'")
        return results

    def find_concept_features(
        self,
        positive_queries: List[str],
        negative_queries: List[str],
        top_k_per_query: int = 5
    ) -> Dict[str, List[Dict]]:
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

        print("\nSearching for positive class features via Neuronpedia...")
        for query in positive_queries:
            features = self.search_features(query, top_k=top_k_per_query)
            # Convert to format compatible with QualityValidator
            for feat in features:
                positive_features.append({
                    'id': str(feat['index']),
                    'label': feat['description'],
                    'index': feat['index'],
                    'score': feat.get('cosine_similarity', 0.0)
                })
            print(f"  Query '{query}' returned {len(features)} features")

        print("\nSearching for negative class features via Neuronpedia...")
        for query in negative_queries:
            features = self.search_features(query, top_k=top_k_per_query)
            # Convert to format compatible with QualityValidator
            for feat in features:
                negative_features.append({
                    'id': str(feat['index']),
                    'label': feat['description'],
                    'index': feat['index'],
                    'score': feat.get('cosine_similarity', 0.0)
                })
            print(f"  Query '{query}' returned {len(features)} features")

        # Remove duplicates based on feature index
        positive_features_deduped = list({f['index']: f for f in positive_features}.values())
        negative_features_deduped = list({f['index']: f for f in negative_features}.values())

        print(f"\nNeuronpedia feature search summary:")
        print(f"  Positive features: {len(positive_features)} total, {len(positive_features_deduped)} unique")
        print(f"  Negative features: {len(negative_features)} total, {len(negative_features_deduped)} unique")

        return {
            'positive_features': positive_features_deduped,
            'negative_features': negative_features_deduped
        }

    def get_feature(self, feature_index: int) -> Dict:
        """Get individual feature by index using Neuronpedia API.

        Args:
            feature_index: SAE feature index

        Returns:
            Dict with keys: index, description, full_data
        """
        # Check cache first
        if feature_index in self.feature_cache:
            return self.feature_cache[feature_index]

        try:
            response = requests.get(
                f"{self.base_url}/api/feature/{self.model_id}/{self.layer_id}/{feature_index}",
                headers={"x-api-key": self.api_key},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract description from explanations
            description = f"Feature {feature_index}"
            explanations = data.get('explanations', [])
            if explanations and len(explanations) > 0:
                description = explanations[0].get('description', description)

            # Build feature info
            feature_info = {
                'index': feature_index,
                'description': description,
                'full_data': data
            }

            # Cache it
            self.feature_cache[feature_index] = feature_info
            return feature_info

        except Exception as e:
            print(f"  Warning: Failed to fetch feature {feature_index}: {e}")
            # Return placeholder
            placeholder = {
                'index': feature_index,
                'description': f"Feature {feature_index}",
                'full_data': {}
            }
            self.feature_cache[feature_index] = placeholder
            return placeholder

    def get_feature_labels(self, feature_indices: List[int]) -> Dict[int, str]:
        """Get labels for multiple features.

        Args:
            feature_indices: List of SAE feature indices

        Returns:
            Dictionary mapping feature index to label/description
        """
        labels = {}
        missing_indices = []

        # Check cache first
        for idx in feature_indices:
            if idx in self.feature_cache:
                labels[idx] = self.feature_cache[idx]['description']
            else:
                missing_indices.append(idx)

        # Fetch missing features individually
        if missing_indices:
            print(f"  Fetching {len(missing_indices)} features from API...")
            for idx in missing_indices:
                feature = self.get_feature(idx)
                labels[idx] = feature['description']
                # Rate limiting
                if len(missing_indices) > 1:
                    time.sleep(0.1)

        return labels

    def get_label(self, feature_index: int) -> str:
        """Get label for a single feature.

        Args:
            feature_index: SAE feature index

        Returns:
            Feature label/description
        """
        feature = self.get_feature(feature_index)
        return feature['description']

    @property
    def labels(self) -> Dict[int, str]:
        """Get all cached labels as a dict.

        Returns:
            Dictionary of all features that have been fetched/cached
        """
        return {idx: info['description'] for idx, info in self.feature_cache.items()}

    def get_coverage_stats(self) -> Dict[str, any]:
        """Get statistics about cached features.

        Returns:
            Dictionary with cache statistics
        """
        total_features = 131072  # For Andy's 131K SAE

        return {
            'total_features': total_features,
            'cached_features': len(self.feature_cache),
            'cache_percent': 100 * len(self.feature_cache) / total_features if len(self.feature_cache) > 0 else 0,
            'search_queries_cached': len(self.search_cache)
        }

    def clear_cache(self):
        """Clear all in-memory caches."""
        self.feature_cache.clear()
        self.search_cache.clear()
        print("  Cache cleared")
