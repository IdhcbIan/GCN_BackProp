"""
Differentiable ranking for backpropagation through graph construction.
"""

import torch
import torch.nn.functional as F
import config


def soft_topk_ranking(features, k=None, temperature=None, metric='cosine'):
    """
    Soft Top-K ranking using temperature-based softmax.
    """
    if k is None:
        k = config.K_NEIGHBORS
    if temperature is None:
        temperature = config.RANKING_TEMPERATURE

    device = config.DEVICE

    if isinstance(features, list):
        features = torch.tensor(features, device=device, dtype=torch.float32)

    # Compute similarity/distance matrix
    if metric == 'cosine':
        features_norm = F.normalize(features, p=2, dim=1)
        scores = torch.mm(features_norm, features_norm.t())
    else:  # euclidean
        x_norm = (features**2).sum(1).view(-1, 1)
        y_norm = (features**2).sum(1).view(1, -1)
        dist_matrix = x_norm + y_norm - 2.0 * torch.mm(features, features.transpose(0, 1))
        scores = -dist_matrix  # Convert distance to similarity

    # Apply temperature scaling and softmax (this is where gradients flow!)
    soft_weights = F.softmax(scores / temperature, dim=1)

    # Get top-k indices and their soft weights
    _, rankings = torch.topk(scores, k=k, dim=1, largest=True)
    top_k_weights = torch.gather(soft_weights, 1, rankings)

    return rankings, top_k_weights


def get_rankings(features, k=None, temperature=None, metric='cosine'):
    """
    Get differentiable rankings using Soft Top-K.
    """
    return soft_topk_ranking(features, k, temperature, metric)


def test_differentiable_ranking():
    """Test that soft topk ranking supports backpropagation."""
    print("=" * 70)
    print("TESTING DIFFERENTIABLE SOFT TOP-K RANKING")

    # Create test features with gradient tracking
    n_samples, n_features = 100, config.FEATURE_DIM
    features = torch.randn(n_samples, n_features, device=config.DEVICE, requires_grad=True)

    print(f"Test features shape: {features.shape}")
    print(f"Device: {config.DEVICE}")

    print("Testing Soft Top-K...")
    try:
        rankings, weights = soft_topk_ranking(features, k=10)
        loss = weights.sum()
        loss.backward()

    except Exception as e:
        print(f"  Status: FAIL - {e}")

    print("=" * 70)


if __name__ == "__main__":
    test_differentiable_ranking()
