"""
End-to-end training: Finetune DINOv2 through differentiable ranking into GCN.
"""

import torch
import torch.nn.functional as F
import config
from Utils import fold_split, compute_accuracy, create_train_val_test_masks
from model import ARMANet
from Diferentiable_Rks import soft_topk_ranking
from torch_geometric.data import Data


def build_graph_from_features(features, labels, train_indices, test_indices):
    """
    Build graph WITH gradient flow from features to edges.
    """
    n_samples = features.shape[0]

    # Create differentiable rankings
    rankings, soft_weights = soft_topk_ranking(
        features,
        k=config.K_NEIGHBORS,
        temperature=config.RANKING_TEMPERATURE,
        metric=config.RANKING_METRIC
    )

    # Build edge list from rankings
    edge_index = []
    for node_idx in range(n_samples):
        for neighbor_pos in range(config.K_NEIGHBORS):
            neighbor_idx = int(rankings[node_idx][neighbor_pos].item())
            edge_index.append([node_idx, neighbor_idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        n_samples, train_indices, test_indices
    )

    # Create PyG Data object
    data = Data(
        x=features.float(),  # Keep gradients!
        edge_index=edge_index.to(config.DEVICE),
        y=torch.tensor(labels, dtype=torch.long).to(config.DEVICE),
        train_mask=train_mask.to(config.DEVICE),
        val_mask=val_mask.to(config.DEVICE),
        test_mask=test_mask.to(config.DEVICE)
    )

    return data, soft_weights


def train_gcn_with_feature_finetuning(dinov2, images, labels, train_indices, test_indices, fold_idx):
    """
    Train GCN while FINETUNING the feature extractor.
    """
    if config.VERBOSE:
        print(f"\n  Fold {fold_idx + 1} - End-to-End Training:")

    # Initialize GCN
    num_features = config.FEATURE_DIM
    num_classes = max(labels) + 1
    gcn_model = ARMANet(num_features, num_classes).to(config.DEVICE)

    # IMPORTANT: Optimizer includes BOTH GCN and DINOv2 parameters!
    optimizer = torch.optim.Adam(
        list(gcn_model.parameters()) + list(dinov2.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Training loop
    dinov2.train()  # Enable training mode for finetuning
    gcn_model.train()

    for epoch in range(config.N_EPOCHS):
        optimizer.zero_grad()

        # Extract features (with gradients flowing!)
        features = dinov2(images)

        # Build graph from features (differentiable!)
        data, soft_weights = build_graph_from_features(
            features, labels, train_indices, test_indices
        )

        # Forward pass through GCN
        out = gcn_model(data)

        # Compute loss
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()

        # Update BOTH GCN and DINOv2
        optimizer.step()

        if config.VERBOSE and (epoch + 1) % config.PRINT_EVERY_N_EPOCHS == 0:
            print(f"    Epoch {epoch + 1}/{config.N_EPOCHS}, Loss: {loss.item():.4f}")

    # Evaluation
    dinov2.eval()
    gcn_model.eval()

    with torch.no_grad():
        features = dinov2(images)
        data, _ = build_graph_from_features(features, labels, train_indices, test_indices)
        out = gcn_model(data)
        _, pred = out.max(dim=1)
        predictions = pred[data.test_mask].cpu().tolist()

    test_labels = [labels[i] for i in test_indices]
    accuracy = compute_accuracy(predictions, test_labels)

    if config.VERBOSE:
        print(f"    Accuracy: {accuracy:.4f}")

    return predictions, test_labels, accuracy


def run_e2e_experiment(dinov2, images, labels):
    """
    Run end-to-end experiment with DINOv2 finetuning.
    """
    print("=" * 70)
    print("END-TO-END TRAINING: Finetuning DINOv2 through GCN")

    # Create folds
    folds = fold_split(images.cpu().numpy(), labels, n_folds=config.N_FOLDS)

    all_results = []

    for fold_idx, (test_indices, train_indices) in enumerate(folds):
        result = train_gcn_with_feature_finetuning(
            dinov2, images, labels, train_indices, test_indices, fold_idx
        )
        all_results.append(result)

    # Compute statistics
    accuracies = [result[2] for result in all_results]
    overall_accuracy = sum(accuracies) / len(accuracies)

    print("\n" + "=" * 70)
    print("END-TO-END RESULTS")
    print("=" * 70)
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Accuracies per fold: {[f'{acc:.4f}' for acc in accuracies]}")
    print("=" * 70)

    return all_results, overall_accuracy
