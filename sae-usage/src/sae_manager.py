"""
SAE Manager for loading Goodfire SAE weights and computing cosine similarities.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path


class SAEManager:
    """Manages SAE weights and computes cosine similarities with decoder directions."""

    def __init__(
        self,
        repo_id: str,
        cache_dir: str,
        filename: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SAE Manager.

        Args:
            repo_id: HuggingFace repo ID (e.g., "andyrdt/saes-llama-3.1-8b-instruct")
            cache_dir: Directory to cache SAE weights
            filename: Filename within the repo (e.g., "resid_post_layer_19/trainer_0/ae.pt")
            device: Device to load SAE on
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.filename = filename
        self.device = device

        # Will be loaded lazily
        self.decoder_weights = None  # Shape: [d_sae, d_model]
        self.decoder_normalized = None  # Normalized decoder for cosine similarity
        self.num_features = None
        self.hidden_dim = None
        self.sae_loaded = False  # Track if SAE has been loaded

        # Feature index to UUID mapping (will be populated from Goodfire API)
        self.index_to_uuid = {}

        print(f"Initialized SAE Manager for {repo_id}")
        if filename:
            print(f"  Filename: {filename}")

    def load_sae(self) -> None:
        """Download and load SAE weights from HuggingFace."""
        if self.sae_loaded and self.decoder_weights is not None:
            print("SAE weights already loaded")
            return

        print(f"Downloading SAE weights from {self.repo_id}...")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine filename
        if self.filename:
            filename = self.filename
        else:
            # Default: assume .pth file named after repo
            filename = f"{self.repo_id.split('/')[-1]}.pth"

        try:
            print(f"  Downloading: {filename}")
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                repo_type="model"
            )

            print(f"Loading SAE from {file_path}...")

            # Load the state dict to CPU first to avoid OOM
            print(f"  Loading to CPU to conserve GPU memory...")
            state_dict = torch.load(file_path, map_location='cpu')

            # Extract decoder weights
            # Try common key names for decoder weights (includes Andy's SAE format)
            decoder_key = None
            for key in [
                'decoder_linear.weight',  # Goodfire SAE format
                'decoder.weight',         # Standard PyTorch format
                'decoder',                # Simple format (Andy's SAE likely uses this)
                'ae.decoder.weight',      # Nested format
                'ae.decoder',             # Andy's SAE potential format
                'W_dec',                  # SAELens format
                'w_dec'                   # Lowercase variant
            ]:
                if key in state_dict:
                    decoder_key = key
                    print(f"  Found decoder weights at key: '{decoder_key}'")
                    break

            if decoder_key is None:
                # Print available keys to help debug
                print(f"Available keys in SAE state dict: {list(state_dict.keys())}")
                raise KeyError(
                    "Could not find decoder weights in SAE state dict. "
                    f"Available keys: {list(state_dict.keys())}"
                )

            decoder = state_dict[decoder_key]

            # Ensure decoder is 2D: [d_sae, d_model]
            if decoder.ndim == 1:
                # If 1D, reshape or raise error
                raise ValueError(f"Decoder weights are 1D with shape {decoder.shape}, expected 2D")

            # Check if decoder needs to be transposed
            # Decoder should be [d_sae, d_model] where d_model = 4096 for Llama 3.1 8B
            # If FIRST dimension is 4096, it's likely [d_model, d_sae] and needs transposing
            if decoder.shape[0] == 4096 or decoder.shape[0] < decoder.shape[1]:
                print(f"Transposing decoder from {decoder.shape} to {(decoder.shape[1], decoder.shape[0])}")
                decoder = decoder.T

            # Store decoder weights (move to specified device)
            # Note: Can be loaded to GPU after model is freed for faster computation
            self.decoder_weights = decoder.to(self.device)
            self.num_features, self.hidden_dim = self.decoder_weights.shape

            # Normalize decoder weights for cosine similarity
            # Normalize along d_model dimension (dim=1)
            decoder_norms = torch.norm(self.decoder_weights, dim=1, keepdim=True)
            self.decoder_normalized = self.decoder_weights / (decoder_norms + 1e-8)

            print(f"✓ Loaded SAE decoder: {self.num_features} features × {self.hidden_dim} dims")

            # Mark as loaded
            self.sae_loaded = True

            # Free memory
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error loading SAE weights: {e}")
            raise

    def compute_cosine_similarities(
        self,
        activations: np.ndarray,
        top_k: int = 20,
        aggregate: str = "mean"
    ) -> List[Tuple[int, float]]:
        """
        Compute cosine similarities between activations and SAE decoder directions.

        Args:
            activations: Activations of shape [num_tokens, hidden_dim]
            top_k: Number of top features to return
            aggregate: How to aggregate across tokens ("mean", "max", "last")

        Returns:
            List of (feature_index, cosine_similarity) tuples, sorted by similarity
        """
        # Ensure SAE is loaded
        if self.decoder_weights is None:
            self.load_sae()

        # Convert activations to torch tensor and move to decoder device
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).to(self.device)
        else:
            activations = activations.to(self.device)

        # Ensure activations are 2D: [num_tokens, hidden_dim]
        if activations.ndim == 1:
            activations = activations.unsqueeze(0)

        # Cast to float32 to match decoder dtype
        activations = activations.float()

        num_tokens, hidden_dim = activations.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Activation hidden dim {hidden_dim} doesn't match SAE decoder dim {self.hidden_dim}"
            )

        # Normalize activations for cosine similarity
        activation_norms = torch.norm(activations, dim=1, keepdim=True)
        activations_normalized = activations / (activation_norms + 1e-8)

        # Compute cosine similarities: [num_tokens, num_features]
        # activations_normalized: [num_tokens, hidden_dim]
        # decoder_normalized.T: [hidden_dim, num_features]
        cosine_sims = torch.matmul(activations_normalized, self.decoder_normalized.T)

        # Aggregate across tokens
        if aggregate == "mean":
            aggregated_sims = cosine_sims.mean(dim=0)  # [num_features]
        elif aggregate == "max":
            aggregated_sims = cosine_sims.max(dim=0)[0]  # [num_features]
        elif aggregate == "last":
            aggregated_sims = cosine_sims[-1]  # [num_features]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

        # Get top-k features
        top_k = min(top_k, self.num_features)
        top_values, top_indices = torch.topk(aggregated_sims, k=top_k)

        # Convert to list of tuples
        results = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(top_indices, top_values)
        ]

        return results

    def batch_compute_cosine_similarities(
        self,
        activations_list: List[np.ndarray],
        top_k: int = 20,
        aggregate: str = "mean"
    ) -> List[List[Tuple[int, float]]]:
        """
        Compute cosine similarities for multiple activation arrays.

        Args:
            activations_list: List of activation arrays
            top_k: Number of top features to return per example
            aggregate: How to aggregate across tokens

        Returns:
            List of results, one per input activation array
        """
        return [
            self.compute_cosine_similarities(acts, top_k=top_k, aggregate=aggregate)
            for acts in activations_list
        ]

    def compute_contrastive_cosine_similarities(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray,
        top_k: int = 20,
        aggregate: str = "mean"
    ) -> List[Tuple[int, float]]:
        """
        Compute cosine similarities between the contrastive difference (steering vector)
        and SAE decoder directions.

        This identifies which SAE features are aligned with the direction that points
        from negative class behavior to positive class behavior.

        Args:
            positive_activations: Activations for positive class [num_tokens, hidden_dim]
            negative_activations: Activations for negative class [num_tokens, hidden_dim]
            top_k: Number of top features to return
            aggregate: How to aggregate across tokens before taking difference
                      ("mean", "max", or "last")

        Returns:
            List of (feature_index, cosine_similarity) tuples for features aligned
            with the contrastive direction (positive - negative)
        """
        # Ensure SAE is loaded
        if self.decoder_weights is None:
            self.load_sae()

        # Convert to torch tensors if needed and move to decoder device
        if isinstance(positive_activations, np.ndarray):
            positive_activations = torch.from_numpy(positive_activations).to(self.device)
        else:
            positive_activations = positive_activations.to(self.device)

        if isinstance(negative_activations, np.ndarray):
            negative_activations = torch.from_numpy(negative_activations).to(self.device)
        else:
            negative_activations = negative_activations.to(self.device)

        # Ensure both are 2D
        if positive_activations.ndim == 1:
            positive_activations = positive_activations.unsqueeze(0)
        if negative_activations.ndim == 1:
            negative_activations = negative_activations.unsqueeze(0)

        # Cast to float32
        positive_activations = positive_activations.float()
        negative_activations = negative_activations.float()

        # Note: Shapes may differ in token dimension (different response lengths)
        # We aggregate to single vectors first, then they'll match

        # Aggregate across tokens first
        if aggregate == "mean":
            pos_aggregated = positive_activations.mean(dim=0)  # [hidden_dim]
            neg_aggregated = negative_activations.mean(dim=0)  # [hidden_dim]
        elif aggregate == "max":
            pos_aggregated = positive_activations.max(dim=0)[0]  # [hidden_dim]
            neg_aggregated = negative_activations.max(dim=0)[0]  # [hidden_dim]
        elif aggregate == "last":
            pos_aggregated = positive_activations[-1]  # [hidden_dim]
            neg_aggregated = negative_activations[-1]  # [hidden_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

        # Compute contrastive difference (steering vector)
        # This points from negative class → positive class
        steering_vector = pos_aggregated - neg_aggregated  # [hidden_dim]

        # Add batch dimension for consistency with compute_cosine_similarities
        steering_vector = steering_vector.unsqueeze(0)  # [1, hidden_dim]

        # Normalize steering vector
        vector_norm = torch.norm(steering_vector, dim=1, keepdim=True)
        steering_normalized = steering_vector / (vector_norm + 1e-8)

        # Compute cosine similarities with SAE decoder directions
        # steering_normalized: [1, hidden_dim]
        # decoder_normalized.T: [hidden_dim, num_features]
        cosine_sims = torch.matmul(steering_normalized, self.decoder_normalized.T)

        # Remove batch dimension
        cosine_sims = cosine_sims.squeeze(0)  # [num_features]

        # Get top-k positive AND top-k negative features
        top_k = min(top_k, self.num_features)

        # Top positive similarities (steer toward positive class)
        top_pos_values, top_pos_indices = torch.topk(cosine_sims, k=top_k)

        # Top negative similarities (steer toward negative class)
        # Get bottom-k (most negative)
        top_neg_values, top_neg_indices = torch.topk(cosine_sims, k=top_k, largest=False)

        # Convert to lists of tuples
        positive_results = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(top_pos_indices, top_pos_values)
        ]

        negative_results = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(top_neg_indices, top_neg_values)
        ]

        return {
            'positive': positive_results,  # Features aligned with positive class direction
            'negative': negative_results   # Features aligned with negative class direction
        }

    def set_feature_mapping(self, index_to_uuid: Dict[int, str]) -> None:
        """
        Set mapping from feature indices to Goodfire UUIDs.

        Args:
            index_to_uuid: Dictionary mapping feature index to UUID
        """
        self.index_to_uuid = index_to_uuid
        print(f"Set feature mapping for {len(index_to_uuid)} features")

    def get_feature_uuid(self, feature_index: int) -> Optional[str]:
        """Get Goodfire UUID for a feature index."""
        return self.index_to_uuid.get(feature_index)

    def compute_probe_similarities(
        self,
        probe_direction: np.ndarray,
        top_k: int = 20
    ) -> List[tuple[int, float]]:
        """
        Compute cosine similarities between probe direction and all SAE features.

        Args:
            probe_direction: The learned probe direction vector (should be normalized)
            top_k: Number of top similar features to return

        Returns:
            List of (feature_index, cosine_similarity) tuples, sorted by similarity
        """
        # Ensure SAE is loaded
        if not self.sae_loaded:
            self.load_sae()

        # Convert probe direction to torch tensor
        probe_tensor = torch.from_numpy(probe_direction).float().to(self.device)

        # Normalize probe direction
        probe_norm = probe_tensor / (probe_tensor.norm() + 1e-10)

        # Get decoder weights (d_model, num_features)
        decoder_weights = self.decoder_weights.T  # Shape: (d_model, num_features)

        # Normalize decoder weights along feature dimension
        decoder_norms = decoder_weights.norm(dim=0, keepdim=True) + 1e-10
        decoder_normalized = decoder_weights / decoder_norms

        # Compute cosine similarities
        # probe_norm shape: (d_model,)
        # decoder_normalized shape: (d_model, num_features)
        cosine_sims = torch.matmul(probe_norm, decoder_normalized)  # Shape: (num_features,)

        # Get top-k
        top_values, top_indices = torch.topk(cosine_sims, k=min(top_k, cosine_sims.shape[0]))

        # Convert to list of tuples
        results = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(top_indices, top_values)
        ]

        return results

    def compare_directions(
        self,
        direction1: np.ndarray,
        direction2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two direction vectors.

        Args:
            direction1: First direction vector
            direction2: Second direction vector

        Returns:
            Cosine similarity in [-1, 1]
        """
        # Normalize directions
        dir1_norm = direction1 / (np.linalg.norm(direction1) + 1e-10)
        dir2_norm = direction2 / (np.linalg.norm(direction2) + 1e-10)

        # Compute cosine similarity
        similarity = np.dot(dir1_norm, dir2_norm)

        return float(similarity)

    def get_decoder_direction(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the decoder direction for a specific SAE feature.

        Args:
            feature_index: Index of the SAE feature

        Returns:
            Decoder direction vector, or None if index is invalid
        """
        # Ensure SAE is loaded
        if not self.sae_loaded:
            self.load_sae()

        # Check if index is valid
        if feature_index < 0 or feature_index >= self.decoder_weights.shape[0]:
            print(f"Warning: Feature index {feature_index} is out of range (0-{self.decoder_weights.shape[0]-1})")
            return None

        # Get decoder column for this feature
        # decoder_weights shape: (num_features, d_model)
        decoder_direction = self.decoder_weights[feature_index, :].cpu().numpy()

        return decoder_direction

    def analyze_probe_alignment(
        self,
        probe_direction: np.ndarray,
        steering_feature_index: int,
        top_k: int = 10,
        neuronpedia_client: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze how well the probe aligns with the steering feature and other SAE features.

        Args:
            probe_direction: The learned probe direction
            steering_feature_index: Index of the SAE feature used for steering
            top_k: Number of top similar features to analyze
            neuronpedia_client: Optional client for fetching feature labels

        Returns:
            Dictionary with alignment analysis
        """
        # Get steering feature decoder direction
        steering_direction = self.get_decoder_direction(steering_feature_index)

        if steering_direction is None:
            return {
                'error': f'Invalid steering feature index: {steering_feature_index}'
            }

        # Compute similarity between probe and steering feature
        probe_steering_similarity = self.compare_directions(probe_direction, steering_direction)

        # Find top features similar to probe
        top_features = self.compute_probe_similarities(probe_direction, top_k=top_k)

        # Fetch labels if client provided
        feature_labels = {}
        if neuronpedia_client:
            try:
                indices = [idx for idx, _ in top_features]
                labels_dict = neuronpedia_client.get_feature_labels(indices)
                feature_labels = labels_dict
            except Exception as e:
                print(f"Warning: Failed to fetch labels: {e}")
                feature_labels = {idx: f'Feature {idx}' for idx, _ in top_features}
        else:
            feature_labels = {idx: f'Feature {idx}' for idx, _ in top_features}

        # Build results
        results = {
            'probe_steering_similarity': probe_steering_similarity,
            'steering_feature_index': steering_feature_index,
            'steering_feature_label': feature_labels.get(steering_feature_index, f'Feature {steering_feature_index}'),
            'top_aligned_features': [
                {
                    'index': idx,
                    'similarity': sim,
                    'label': feature_labels.get(idx, f'Feature {idx}'),
                    'is_steering_feature': (idx == steering_feature_index)
                }
                for idx, sim in top_features
            ]
        }

        return results
