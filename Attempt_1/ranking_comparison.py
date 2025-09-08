import torch
import numpy as np
from sklearn.neighbors import BallTree
import torch.nn.functional as F
import time


class RankingMethods:
    """
    Comparison of different ranking methods for GCN:
    1. BallTree (non-differentiable, euclidean distance)
    2. Cosine Similarity + torch.topk (differentiable)
    3. Euclidean Distance + torch.topk (differentiable)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def balltree_ranking(self, features, k=100):
        """
        Original BallTree implementation (non-differentiable)
        Uses euclidean distance - considered the "gold standard"
        """
        if not isinstance(features, np.ndarray):
            features = features.detach().cpu().numpy()
        
        tree = BallTree(features)
        _, rankings = tree.query(features, k=k)
        return rankings
    
    def cosine_similarity_ranking(self, features, k=100):
        """
        Cosine similarity ranking using torch.topk (differentiable)
        This can be backpropagated through!
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # Get top-k similar items (highest similarity)
        _, rankings = torch.topk(similarity_matrix, k=k, dim=1, largest=True)
        
        return rankings
    
    def euclidean_topk_ranking(self, features, k=100):
        """
        Euclidean distance ranking using torch.topk (differentiable)
        Similar to BallTree but differentiable
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Compute squared euclidean distance matrix
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
        x_norm = (features**2).sum(1).view(-1, 1)
        y_norm = (features**2).sum(1).view(1, -1)
        dist_matrix = x_norm + y_norm - 2.0 * torch.mm(features, features.transpose(0, 1))
        
        # Get top-k closest items (smallest distance)
        _, rankings = torch.topk(dist_matrix, k=k, dim=1, largest=False)
        
        return rankings
    
    def compare_rankings(self, features, k=100):
        """
        Compare all three ranking methods
        """
        print("Comparing ranking methods...")
        print(f"Features shape: {features.shape}")
        print(f"K neighbors: {k}")
        print("-" * 50)
        
        results = {}
        
        # 1. BallTree (baseline)
        print("1. Computing BallTree rankings...")
        balltree_rankings = self.balltree_ranking(features, k)
        results['balltree'] = balltree_rankings
        print(f"   BallTree rankings shape: {balltree_rankings.shape}")
        
        # 2. Cosine similarity
        print("2. Computing Cosine Similarity rankings...")
        cosine_rankings = self.cosine_similarity_ranking(features, k)
        results['cosine'] = cosine_rankings
        print(f"   Cosine rankings shape: {cosine_rankings.shape}")
        
        # 3. Euclidean with topk
        print("3. Computing Euclidean topk rankings...")
        euclidean_rankings = self.euclidean_topk_ranking(features, k)
        results['euclidean_topk'] = euclidean_rankings
        print(f"   Euclidean topk rankings shape: {euclidean_rankings.shape}")
        
        return results
    
    def test_backpropagation(self, features, k=10):
        """
        Test if ranking methods are differentiable
        """
        print("\nTesting backpropagation capabilities...")
        print("-" * 50)
        
        # Use only a small subset for faster testing
        if isinstance(features, np.ndarray):
            features_subset = features[:100]  # Use first 100 samples
            features = torch.tensor(features_subset, device=self.device, dtype=torch.float32, requires_grad=True)
        else:
            features = features[:100].clone().detach().requires_grad_(True)
        
        # Test cosine similarity backprop
        print("1. Testing Cosine Similarity backprop...")
        try:
            cosine_rankings = self.cosine_similarity_ranking(features, k)
            # Create a dummy loss (sum of rankings)
            loss = cosine_rankings.float().sum()
            loss.backward()
            print(f"   ✓ Cosine similarity is differentiable!")
            print(f"   Gradient shape: {features.grad.shape}")
        except Exception as e:
            print(f"   ✗ Cosine similarity backprop failed: {e}")
        
        # Reset gradients
        features.grad = None
        
        # Test euclidean topk backprop
        print("2. Testing Euclidean topk backprop...")
        try:
            euclidean_rankings = self.euclidean_topk_ranking(features, k)
            # Create a dummy loss
            loss = euclidean_rankings.float().sum()
            loss.backward()
            print(f"   ✓ Euclidean topk is differentiable!")
            print(f"   Gradient shape: {features.grad.shape}")
        except Exception as e:
            print(f"   ✗ Euclidean topk backprop failed: {e}")
        
        # Test BallTree (should not be differentiable)
        print("3. Testing BallTree backprop...")
        try:
            # BallTree works with numpy, so it won't be differentiable
            balltree_rankings = self.balltree_ranking(features.detach(), k)
            print(f"   ✗ BallTree is not differentiable (as expected)")
        except Exception as e:
            print(f"   ✗ BallTree test failed: {e}")
    
    def ranking_overlap_analysis(self, rankings_dict, top_n=10):
        """
        Analyze overlap between different ranking methods
        """
        print(f"\nAnalyzing ranking overlap (top-{top_n})...")
        print("-" * 50)
        
        methods = list(rankings_dict.keys())
        n_samples = len(rankings_dict[methods[0]])
        
        # Convert torch tensors to numpy for easier comparison
        rankings_np = {}
        for method, rankings in rankings_dict.items():
            if isinstance(rankings, torch.Tensor):
                rankings_np[method] = rankings.detach().cpu().numpy()
            else:
                rankings_np[method] = rankings
        
        # Calculate overlap between methods
        overlaps = {}
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                overlap_scores = []
                for sample_idx in range(n_samples):
                    set1 = set(rankings_np[method1][sample_idx][:top_n])
                    set2 = set(rankings_np[method2][sample_idx][:top_n])
                    overlap = len(set1.intersection(set2)) / top_n
                    overlap_scores.append(overlap)
                
                avg_overlap = np.mean(overlap_scores)
                overlaps[f"{method1}_vs_{method2}"] = avg_overlap
                print(f"   {method1} vs {method2}: {avg_overlap:.3f} average overlap")
        
        return overlaps


def test_ranking_comparison():
    """
    Main function to test ranking comparison
    """
    print("=" * 60)
    print("RANKING METHODS COMPARISON FOR GCN BACKPROPAGATION")
    print("=" * 60)
    
    # Load features
    try:
        features = np.load('/home/ian/Documents/Repos/GCN_BackProp/Attempt_1/dinov2_vitl14_emb.npy')
        print(f"Loaded features: {features.shape}")
    except FileNotFoundError:
        print("Creating dummy features for testing...")
        features = np.random.randn(1000, 384).astype(np.float32)  # Typical DINOv2 features
    
    # Initialize ranking methods
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranker = RankingMethods(device=device)
    
    # Compare ranking methods
    k_neighbors = 50
    results = ranker.compare_rankings(features, k=k_neighbors)
    
    # Test backpropagation
    ranker.test_backpropagation(features[:100], k=10)  # Use subset for faster testing
    
    # Analyze ranking overlaps
    ranker.ranking_overlap_analysis(results, top_n=10)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- BallTree: Non-differentiable, euclidean distance (baseline)")
    print("- Cosine Similarity: Differentiable, semantic similarity")
    print("- Euclidean topk: Differentiable, spatial similarity")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    test_ranking_comparison()
