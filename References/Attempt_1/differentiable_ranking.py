import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DifferentiableRanking:
    """
    Collection of differentiable ranking methods that can be backpropagated through.
    
    Key insight: Instead of using discrete indices from torch.topk(), we use soft approximations
    that maintain gradient flow while approximating ranking behavior.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def soft_topk_ranking(self, features: torch.Tensor, k: int = 100, 
                         temperature: float = 0.1, metric: str = 'euclidean') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft Top-K ranking using temperature-based softmax.
        Returns soft attention weights instead of hard indices.
        
        Args:
            features: Input features [N, D]
            k: Number of neighbors (for compatibility, but returns all weights)
            temperature: Lower = sharper, higher = softer
            metric: 'euclidean' or 'cosine'
            
        Returns:
            soft_weights: [N, N] attention weights (differentiable)
            top_k_weights: [N, k] top-k soft weights for compatibility
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Compute similarity/distance matrix
        if metric == 'cosine':
            # Cosine similarity
            features_norm = F.normalize(features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            scores = similarity_matrix
        else:
            # Euclidean distance (convert to similarity)
            x_norm = (features**2).sum(1).view(-1, 1)
            y_norm = (features**2).sum(1).view(1, -1)
            dist_matrix = x_norm + y_norm - 2.0 * torch.mm(features, features.transpose(0, 1))
            # Convert distance to similarity (negative distance)
            scores = -dist_matrix
        
        # Apply temperature scaling and softmax
        soft_weights = F.softmax(scores / temperature, dim=1)
        
        # Get top-k soft weights for compatibility with existing code
        _, top_indices = torch.topk(scores, k=k, dim=1, largest=True)
        top_k_weights = torch.gather(soft_weights, 1, top_indices)
        
        return soft_weights, top_k_weights
    
    def attention_based_ranking(self, features: torch.Tensor, k: int = 100,
                               temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-based ranking using learnable attention mechanism.
        
        Args:
            features: Input features [N, D]
            k: Number of top weights to return
            temperature: Temperature for attention
            
        Returns:
            attention_weights: [N, N] attention matrix
            aggregated_features: [N, D] attention-weighted features
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Compute attention scores (query-key attention)
        # Each feature vector acts as both query and key
        attention_scores = torch.mm(features, features.transpose(0, 1))
        
        # Apply temperature and softmax
        attention_weights = F.softmax(attention_scores / temperature, dim=1)
        
        # Compute attention-weighted features
        aggregated_features = torch.mm(attention_weights, features)
        
        return attention_weights, aggregated_features
    
    def gumbel_topk_ranking(self, features: torch.Tensor, k: int = 100,
                           temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Gumbel-Softmax based ranking for discrete approximations with continuous gradients.
        
        Args:
            features: Input features [N, D]
            k: Number of neighbors
            temperature: Gumbel temperature
            hard: Whether to use straight-through estimator
            
        Returns:
            gumbel_weights: [N, N] Gumbel-softmax weights
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Compute similarity scores
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_matrix) + 1e-20) + 1e-20)
        gumbel_scores = (similarity_matrix + gumbel_noise) / temperature
        
        # Apply softmax
        gumbel_weights = F.softmax(gumbel_scores, dim=1)
        
        if hard:
            # Straight-through estimator: hard selection in forward, soft in backward
            _, hard_indices = torch.topk(gumbel_scores, k=k, dim=1, largest=True)
            hard_weights = torch.zeros_like(gumbel_weights)
            hard_weights.scatter_(1, hard_indices, 1.0)
            
            # Straight-through: use hard weights in forward, soft weights in backward
            gumbel_weights = hard_weights + gumbel_weights - gumbel_weights.detach()
        
        return gumbel_weights
    
    def differentiable_ranking_loss(self, features: torch.Tensor, rankings: torch.Tensor,
                                   method: str = 'soft_topk', k: int = 100,
                                   temperature: float = 0.1) -> torch.Tensor:
        """
        Compute a differentiable ranking loss that can be backpropagated.
        
        Args:
            features: Input features [N, D]
            rankings: Target rankings [N, k] (optional, for supervised ranking)
            method: Ranking method to use
            k: Number of neighbors
            temperature: Temperature parameter
            
        Returns:
            loss: Scalar tensor that can be backpropagated
        """
        if method == 'soft_topk':
            soft_weights, top_k_weights = self.soft_topk_ranking(features, k, temperature)
            # Loss: encourage sparsity while maintaining differentiability
            loss = -torch.sum(top_k_weights * torch.log(top_k_weights + 1e-10))  # Entropy loss
            
        elif method == 'attention':
            attention_weights, aggregated_features = self.attention_based_ranking(features, k, temperature)
            # Loss: reconstruction loss using attention-weighted features
            loss = F.mse_loss(aggregated_features, features)
            
        elif method == 'gumbel':
            gumbel_weights = self.gumbel_topk_ranking(features, k, temperature)
            # Loss: sparsity regularization
            loss = torch.sum(gumbel_weights * torch.log(gumbel_weights + 1e-10))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return loss
    
    def create_differentiable_graph_edges(self, features: torch.Tensor, k: int = 100,
                                         method: str = 'soft_topk', 
                                         temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph edges using differentiable ranking for GCN.
        
        Args:
            features: Input features [N, D]
            k: Number of neighbors per node
            method: Ranking method
            temperature: Temperature parameter
            
        Returns:
            edge_indices: [2, E] edge indices for PyTorch Geometric
            edge_weights: [E] edge weights (differentiable)
        """
        if method == 'soft_topk':
            soft_weights, top_k_weights = self.soft_topk_ranking(features, k, temperature)
            
            # Get top-k indices (for graph structure)
            scores = torch.mm(F.normalize(features, p=2, dim=1), 
                            F.normalize(features, p=2, dim=1).t())
            _, top_indices = torch.topk(scores, k=k, dim=1, largest=True)
            
            # Create edge list
            source_nodes = torch.arange(features.shape[0], device=self.device).unsqueeze(1).repeat(1, k)
            edge_indices = torch.stack([source_nodes.flatten(), top_indices.flatten()], dim=0)
            
            # Get corresponding soft weights as edge weights
            edge_weights = top_k_weights.flatten()
            
        elif method == 'attention':
            attention_weights, _ = self.attention_based_ranking(features, k, temperature)
            
            # Get top-k attention weights
            _, top_indices = torch.topk(attention_weights, k=k, dim=1, largest=True)
            
            source_nodes = torch.arange(features.shape[0], device=self.device).unsqueeze(1).repeat(1, k)
            edge_indices = torch.stack([source_nodes.flatten(), top_indices.flatten()], dim=0)
            
            edge_weights = torch.gather(attention_weights, 1, top_indices).flatten()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return edge_indices, edge_weights


def test_differentiable_ranking():
    """Test all differentiable ranking methods"""
    print("=" * 60)
    print("TESTING DIFFERENTIABLE RANKING METHODS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranker = DifferentiableRanking(device=device)
    
    # Create test features
    n_samples, n_features = 100, 384
    features = torch.randn(n_samples, n_features, device=device, requires_grad=True)
    
    print(f"Test features shape: {features.shape}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Test 1: Soft Top-K
    print("1. Testing Soft Top-K Ranking...")
    try:
        soft_weights, top_k_weights = ranker.soft_topk_ranking(features, k=10, temperature=0.1)
        loss = ranker.differentiable_ranking_loss(features, None, method='soft_topk', k=10)
        loss.backward()
        print(f"   ✓ Soft Top-K successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Gradient norm: {features.grad.norm().item():.4f}")
        features.grad = None  # Reset gradients
    except Exception as e:
        print(f"   ✗ Soft Top-K failed: {e}")
    
    # Test 2: Attention-based
    print("2. Testing Attention-based Ranking...")
    try:
        attention_weights, aggregated_features = ranker.attention_based_ranking(features, k=10)
        loss = ranker.differentiable_ranking_loss(features, None, method='attention', k=10)
        loss.backward()
        print(f"   ✓ Attention-based successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Gradient norm: {features.grad.norm().item():.4f}")
        features.grad = None
    except Exception as e:
        print(f"   ✗ Attention-based failed: {e}")
    
    # Test 3: Gumbel-Softmax
    print("3. Testing Gumbel-Softmax Ranking...")
    try:
        gumbel_weights = ranker.gumbel_topk_ranking(features, k=10, temperature=1.0)
        loss = ranker.differentiable_ranking_loss(features, None, method='gumbel', k=10)
        loss.backward()
        print(f"   ✓ Gumbel-Softmax successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Gradient norm: {features.grad.norm().item():.4f}")
        features.grad = None
    except Exception as e:
        print(f"   ✗ Gumbel-Softmax failed: {e}")
    
    # Test 4: Graph edge creation
    print("4. Testing Differentiable Graph Edge Creation...")
    try:
        edge_indices, edge_weights = ranker.create_differentiable_graph_edges(
            features, k=10, method='soft_topk'
        )
        print(f"   ✓ Graph edge creation successful!")
        print(f"   Edge indices shape: {edge_indices.shape}")
        print(f"   Edge weights shape: {edge_weights.shape}")
        print(f"   Edge weights sum: {edge_weights.sum().item():.4f}")
    except Exception as e:
        print(f"   ✗ Graph edge creation failed: {e}")
    
    print("-" * 60)
    print("RECOMMENDATIONS:")
    print("- Soft Top-K: Best for preserving ranking semantics")
    print("- Attention: Best for learning adaptive similarities")
    print("- Gumbel: Best for discrete approximations")
    print("- Use temperature to control sharpness (lower = sharper)")
    print("=" * 60)


if __name__ == "__main__":
    test_differentiable_ranking()

