"""
Logistic regression probe training and evaluation.

Trains linear probes on contrastive activation pairs and evaluates their
performance for concept classification.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from dataclasses import dataclass

from src.data_generator import ContrastivePair


@dataclass
class ProbeResults:
    """Results from probe training and evaluation."""
    probe: LogisticRegression
    train_accuracy: float
    test_accuracy: float
    test_auc: float
    test_precision: float
    test_recall: float
    test_f1: float
    train_size: int
    test_size: int
    probe_direction: np.ndarray
    feature_dim: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'train_accuracy': float(self.train_accuracy),
            'test_accuracy': float(self.test_accuracy),
            'test_auc': float(self.test_auc),
            'test_precision': float(self.test_precision),
            'test_recall': float(self.test_recall),
            'test_f1': float(self.test_f1),
            'train_size': int(self.train_size),
            'test_size': int(self.test_size),
            'feature_dim': int(self.feature_dim),
        }

    def __str__(self) -> str:
        """Pretty print results."""
        return (
            f"Probe Training Results:\n"
            f"  Train Size: {self.train_size}\n"
            f"  Test Size: {self.test_size}\n"
            f"  Train Accuracy: {self.train_accuracy:.4f}\n"
            f"  Test Accuracy: {self.test_accuracy:.4f}\n"
            f"  Test AUC-ROC: {self.test_auc:.4f}\n"
            f"  Test Precision: {self.test_precision:.4f}\n"
            f"  Test Recall: {self.test_recall:.4f}\n"
            f"  Test F1: {self.test_f1:.4f}\n"
            f"  Feature Dimension: {self.feature_dim}"
        )


class ProbeTrainer:
    """
    Train and evaluate logistic regression probes.

    Uses scikit-learn's LogisticRegression for linear classification
    between positive and negative class activations.
    """

    def __init__(
        self,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize the probe trainer.

        Args:
            solver: Solver for logistic regression
            max_iter: Maximum iterations for solver
            C: Inverse regularization strength
            random_state: Random seed for reproducibility
        """
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state

    def train_from_pairs(
        self,
        pairs: List[ContrastivePair],
        test_size: float = 0.2,
        balance_classes: bool = True
    ) -> ProbeResults:
        """
        Train probe from contrastive pairs.

        Args:
            pairs: List of ContrastivePair objects
            test_size: Fraction of data to use for testing
            balance_classes: Whether to balance positive and negative classes

        Returns:
            ProbeResults object
        """
        # Extract activations and labels
        X_positive = np.array([pair.positive_activations for pair in pairs])
        X_negative = np.array([pair.negative_activations for pair in pairs])

        # Create labels (1 for positive, 0 for negative)
        y_positive = np.ones(len(pairs))
        y_negative = np.zeros(len(pairs))

        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.concatenate([y_positive, y_negative])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Train probe
        probe = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            class_weight='balanced' if balance_classes else None
        )

        probe.fit(X_train, y_train)

        # Evaluate on train set
        y_train_pred = probe.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate on test set
        y_test_pred = probe.predict(X_test)
        y_test_proba = probe.predict_proba(X_test)[:, 1]  # Probability of positive class

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary'
        )

        # Extract probe direction (the weight vector)
        probe_direction = probe.coef_[0]  # Shape: (feature_dim,)

        # Normalize to unit vector
        probe_direction = probe_direction / np.linalg.norm(probe_direction)

        results = ProbeResults(
            probe=probe,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            test_auc=test_auc,
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            train_size=len(X_train),
            test_size=len(X_test),
            probe_direction=probe_direction,
            feature_dim=X.shape[1]
        )

        return results

    def train_from_activations(
        self,
        X_positive: np.ndarray,
        X_negative: np.ndarray,
        test_size: float = 0.2,
        balance_classes: bool = True
    ) -> ProbeResults:
        """
        Train probe directly from activation arrays.

        Args:
            X_positive: Array of positive class activations (n_samples, feature_dim)
            X_negative: Array of negative class activations (n_samples, feature_dim)
            test_size: Fraction of data to use for testing
            balance_classes: Whether to balance positive and negative classes

        Returns:
            ProbeResults object
        """
        # Create labels
        y_positive = np.ones(len(X_positive))
        y_negative = np.zeros(len(X_negative))

        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.concatenate([y_positive, y_negative])

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Train probe
        probe = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            class_weight='balanced' if balance_classes else None
        )

        probe.fit(X_train, y_train)

        # Evaluate on train set
        y_train_pred = probe.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate on test set
        y_test_pred = probe.predict(X_test)
        y_test_proba = probe.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary'
        )

        # Extract probe direction
        probe_direction = probe.coef_[0]
        probe_direction = probe_direction / np.linalg.norm(probe_direction)

        results = ProbeResults(
            probe=probe,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            test_auc=test_auc,
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            train_size=len(X_train),
            test_size=len(X_test),
            probe_direction=probe_direction,
            feature_dim=X.shape[1]
        )

        return results

    def cross_validate(
        self,
        pairs: List[ContrastivePair],
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation to estimate probe performance.

        Args:
            pairs: List of ContrastivePair objects
            n_folds: Number of cross-validation folds

        Returns:
            Dict with mean and std of metrics across folds
        """
        from sklearn.model_selection import cross_val_score

        # Extract activations
        X_positive = np.array([pair.positive_activations for pair in pairs])
        X_negative = np.array([pair.negative_activations for pair in pairs])

        # Create labels
        y_positive = np.ones(len(pairs))
        y_negative = np.zeros(len(pairs))

        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.concatenate([y_positive, y_negative])

        # Create probe
        probe = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            C=self.C,
            random_state=self.random_state,
            class_weight='balanced'
        )

        # Cross-validate
        scores = cross_val_score(probe, X, y, cv=n_folds, scoring='accuracy')

        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }


def compute_steering_alignment(
    probe_direction: np.ndarray,
    steering_vector: np.ndarray
) -> float:
    """
    Compute cosine similarity between probe direction and steering vector.

    Args:
        probe_direction: The learned probe direction (unit vector)
        steering_vector: The SAE decoder direction used for steering

    Returns:
        Cosine similarity in [-1, 1]
    """
    # Normalize steering vector
    steering_norm = steering_vector / np.linalg.norm(steering_vector)

    # Compute cosine similarity
    similarity = np.dot(probe_direction, steering_norm)

    return float(similarity)


def analyze_probe_features(
    probe: LogisticRegression,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Analyze which features are most important for the probe.

    Args:
        probe: Trained LogisticRegression probe
        feature_names: Optional list of feature names
        top_k: Number of top features to return

    Returns:
        Dict with 'positive_features' and 'negative_features' lists
    """
    weights = probe.coef_[0]

    # Get indices sorted by absolute weight
    sorted_indices = np.argsort(np.abs(weights))[::-1][:top_k]

    positive_features = []
    negative_features = []

    for idx in sorted_indices:
        weight = weights[idx]
        if weight > 0:
            positive_features.append((int(idx), float(weight)))
        else:
            negative_features.append((int(idx), float(weight)))

    return {
        'positive_features': positive_features,
        'negative_features': negative_features
    }
