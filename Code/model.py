"""
GCN model architectures and graph construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv
import config
from Utils import create_train_val_test_masks, compute_accuracy


class ARMANet(nn.Module):
    """
    ARMA (AutoRegressive Moving Average) Graph Convolutional Network
    """

    def __init__(self, num_features, num_classes, hidden_dim=None,
                 num_stacks=None, num_layers=None, dropout=None):
        super(ARMANet, self).__init__()

        # Use config defaults if not specified
        hidden_dim = hidden_dim or config.GCN_HIDDEN_DIM
        num_stacks = num_stacks or config.GCN_NUM_STACKS
        num_layers = num_layers or config.GCN_NUM_LAYERS
        dropout = dropout or config.GCN_DROPOUT

        # First ARMA layer
        self.conv1 = ARMAConv(
            num_features, hidden_dim,
            num_stacks=num_stacks,
            num_layers=num_layers,
            shared_weights=True,
            dropout=dropout
        )

        # Second ARMA layer
        self.conv2 = ARMAConv(
            hidden_dim, num_classes,
            num_stacks=num_stacks,
            num_layers=num_layers,
            shared_weights=True,
            dropout=dropout,
            act=lambda x: x  # No activation in final layer
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))

        # Second layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCNClassifier:
    """
    GCN Classifier with differentiable graph construction.
    """

    def __init__(self, rankings, k=None, device=None):
        """
        Initialize GCN Classifier.
        """
        self.device = device or config.DEVICE
        self.k = k or config.K_NEIGHBORS
        self.rankings = rankings

        # Will be set during fit()
        self.model = None
        self.num_classes = None
        self.num_features = None

    def build_graph(self, features, labels, train_indices, test_indices):
        """
        Build PyG Data object from features, labels, and rankings.
        """
        n_samples = features.shape[0]

        # Create masks
        train_mask, val_mask, test_mask = create_train_val_test_masks(
            n_samples, train_indices, test_indices
        )

        # Build edge list from rankings
        edge_index = []
        for node_idx in range(n_samples):
            for neighbor_pos in range(min(self.k, self.rankings.shape[1])):
                neighbor_idx = int(self.rankings[node_idx][neighbor_pos].item())
                edge_index.append([node_idx, neighbor_idx])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create PyG Data object
        data = Data(
            x=features.float().to(self.device),
            edge_index=edge_index.to(self.device),
            y=torch.tensor(labels, dtype=torch.long).to(self.device),
            train_mask=train_mask.to(self.device),
            val_mask=val_mask.to(self.device),
            test_mask=test_mask.to(self.device)
        )

        return data

    def fit(self, features, labels, train_indices, test_indices):
        """
        Train the GCN model.
        """
        # Build graph
        data = self.build_graph(features, labels, train_indices, test_indices)

        # Initialize model
        self.num_features = features.shape[1]
        self.num_classes = max(labels) + 1

        self.model = ARMANet(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Training loop
        self.model.train()
        for epoch in range(config.N_EPOCHS):
            optimizer.zero_grad()
            out = self.model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if config.VERBOSE and (epoch + 1) % config.PRINT_EVERY_N_EPOCHS == 0:
                print(f"    Epoch {epoch + 1}/{config.N_EPOCHS}, Loss: {loss.item():.4f}")

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            _, pred = out.max(dim=1)
            predictions = pred[data.test_mask].cpu().tolist()

        return predictions

    def predict(self, features, labels, train_indices, test_indices):
        """
        Convenience method that calls fit() and returns predictions.
        """
        predictions = self.fit(features, labels, train_indices, test_indices)
        test_labels = [labels[i] for i in test_indices]
        accuracy = compute_accuracy(predictions, test_labels)

        return predictions, accuracy
