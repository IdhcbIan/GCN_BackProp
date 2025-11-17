import torch
import numpy as np
import torch.nn.functional as F
from ranking_comparison import RankingMethods
from Main import GCNClassifier, ARMA


class GCNRankingComparison:
    """
    Test different ranking methods with GCN to compare accuracy
    """
    
    def __init__(self, features, labels, device='cuda'):
        self.features = features
        self.labels = labels
        self.device = device
        self.ranker = RankingMethods(device=device)
        
        # Convert to torch tensors if needed
        if isinstance(self.features, np.ndarray):
            self.features_tensor = torch.tensor(self.features, device=device, dtype=torch.float32)
        else:
            self.features_tensor = self.features
            
        self.labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    
    def create_simple_train_test_split(self, train_ratio=0.7):
        """
        Create a simple train/test split for testing
        """
        n_samples = len(self.features)
        n_train = int(n_samples * train_ratio)
        
        # Random split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        return train_indices, test_indices
    
    def test_ranking_method_accuracy(self, ranking_method, k_neighbors=10, n_trials=3):
        """
        Test a specific ranking method with GCN and return average accuracy
        """
        accuracies = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            # Create train/test split
            train_indices, test_indices = self.create_simple_train_test_split()
            
            # Get rankings using the specified method
            if ranking_method == 'balltree':
                rankings = self.ranker.balltree_ranking(self.features, k=k_neighbors)
            elif ranking_method == 'cosine':
                rankings = self.ranker.cosine_similarity_ranking(self.features_tensor, k=k_neighbors)
                if isinstance(rankings, torch.Tensor):
                    rankings = rankings.detach().cpu().numpy()
            elif ranking_method == 'euclidean_topk':
                rankings = self.ranker.euclidean_topk_ranking(self.features_tensor, k=k_neighbors)
                if isinstance(rankings, torch.Tensor):
                    rankings = rankings.detach().cpu().numpy()
            else:
                raise ValueError(f"Unknown ranking method: {ranking_method}")
            
            # Test with GCN
            accuracy = self.test_gcn_with_rankings(rankings, train_indices, test_indices, k_neighbors)
            accuracies.append(accuracy)
            
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        return avg_accuracy, std_accuracy, accuracies
    
    def test_gcn_with_rankings(self, rankings, train_indices, test_indices, k_neighbors):
        """
        Test GCN with given rankings and return accuracy
        """
        try:
            # Initialize GCN classifier
            pN = len(self.features)
            pNNeurons = 32
            
            gcn = GCNClassifier(
                gcn_type="gcn_arma",
                rks=rankings,
                pN=pN,
                k=k_neighbors,
                pNNeurons=pNNeurons,
                graph_type="knn"
            )
            
            # Fit the model
            gcn.fit(test_indices, train_indices, self.features, self.labels)
            
            # Get predictions
            predictions = gcn.predict()
            
            # Calculate accuracy
            true_labels = [self.labels[i] for i in test_indices]
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
            
            return accuracy
            
        except Exception as e:
            print(f"    Error in GCN testing: {e}")
            return 0.0
    
    def compare_all_methods(self, k_neighbors=10, n_trials=3):
        """
        Compare all ranking methods with GCN accuracy
        """
        print("=" * 70)
        print("GCN ACCURACY COMPARISON WITH DIFFERENT RANKING METHODS")
        print("=" * 70)
        print(f"Features shape: {self.features.shape}")
        print(f"Number of classes: {len(np.unique(self.labels))}")
        print(f"K neighbors: {k_neighbors}")
        print(f"Trials per method: {n_trials}")
        print("-" * 70)
        
        methods = ['balltree', 'cosine', 'euclidean_topk']
        results = {}
        
        for method in methods:
            print(f"\nTesting {method.upper()} rankings:")
            
            avg_acc, std_acc, trial_accs = self.test_ranking_method_accuracy(
                method, k_neighbors, n_trials
            )
            
            results[method] = {
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'trial_accuracies': trial_accs
            }
            
            print(f"  Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"  Individual trials: {[f'{acc:.4f}' for acc in trial_accs]}")
        
        # Print summary comparison
        print("\n" + "=" * 70)
        print("RANKING METHOD ACCURACY SUMMARY:")
        print("=" * 70)
        
        # Sort by average accuracy
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)
        
        for i, (method, result) in enumerate(sorted_methods):
            rank = i + 1
            avg_acc = result['avg_accuracy']
            std_acc = result['std_accuracy']
            
            print(f"{rank}. {method.upper():<15}: {avg_acc:.4f} ± {std_acc:.4f}")
            
            # Compare to BallTree (baseline)
            if method != 'balltree' and 'balltree' in results:
                diff = avg_acc - results['balltree']['avg_accuracy']
                if diff > 0:
                    print(f"   (+{diff:.4f} vs BallTree)")
                else:
                    print(f"   ({diff:.4f} vs BallTree)")
        
        # Test backpropagation capability
        print("\n" + "=" * 70)
        print("BACKPROPAGATION CAPABILITY TEST:")
        print("=" * 70)
        
        self.test_end_to_end_backprop(k_neighbors)
        
        return results
    
    def test_end_to_end_backprop(self, k_neighbors=10):
        """
        Test end-to-end backpropagation through differentiable ranking methods
        """
        print("Testing end-to-end backpropagation...")
        
        # Use a small subset for testing
        subset_size = 100
        features_subset = self.features_tensor[:subset_size].clone().detach().requires_grad_(True)
        
        # Test cosine similarity backprop
        print("\n1. Cosine Similarity end-to-end backprop test:")
        try:
            # Get rankings
            rankings = self.ranker.cosine_similarity_ranking(features_subset, k=k_neighbors)
            
            # Create a simple loss based on rankings
            # This simulates what would happen in a differentiable GCN
            ranking_loss = rankings.float().mean()
            
            # Backpropagate
            ranking_loss.backward()
            
            if features_subset.grad is not None:
                grad_norm = features_subset.grad.norm().item()
                print(f"   ✓ Backpropagation successful!")
                print(f"   Gradient norm: {grad_norm:.6f}")
            else:
                print(f"   ✗ No gradients computed")
                
        except Exception as e:
            print(f"   ✗ Cosine similarity backprop failed: {e}")
        
        # Reset gradients
        features_subset.grad = None
        
        # Test euclidean topk backprop
        print("\n2. Euclidean topk end-to-end backprop test:")
        try:
            # Get rankings
            rankings = self.ranker.euclidean_topk_ranking(features_subset, k=k_neighbors)
            
            # Create a simple loss
            ranking_loss = rankings.float().mean()
            
            # Backpropagate
            ranking_loss.backward()
            
            if features_subset.grad is not None:
                grad_norm = features_subset.grad.norm().item()
                print(f"   ✓ Backpropagation successful!")
                print(f"   Gradient norm: {grad_norm:.6f}")
            else:
                print(f"   ✗ No gradients computed")
                
        except Exception as e:
            print(f"   ✗ Euclidean topk backprop failed: {e}")
        
        print("\n3. BallTree backprop test:")
        print("   ✗ BallTree is not differentiable (as expected)")


def main():
    """
    Main function to run the GCN ranking accuracy comparison
    """
    print("Loading data...")
    
    # Try to load real data, fall back to dummy data
    try:
        features = np.load('/home/ian/Documents/Repos/GCN_BackProp/Attempt_1/dinov2_vitl14_emb.npy')
        print(f"Loaded real features: {features.shape}")
        
        # Create dummy labels for testing (you should replace this with real labels)
        n_classes = 10
        labels = np.random.randint(0, n_classes, size=len(features))
        print(f"Created dummy labels with {n_classes} classes")
        
    except FileNotFoundError:
        print("Real data not found, creating dummy data...")
        n_samples = 500
        n_features = 384  # DINOv2 feature size
        n_classes = 10
        
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        labels = np.random.randint(0, n_classes, size=n_samples)
        
        print(f"Created dummy data: {features.shape}, {n_classes} classes")
    
    # Run comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comparison = GCNRankingComparison(features, labels, device=device)
    
    # Run the comparison
    results = comparison.compare_all_methods(k_neighbors=20, n_trials=2)
    
    return results


if __name__ == "__main__":
    main()


