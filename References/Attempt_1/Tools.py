import numpy as np
import evaluate
import torch

# Removed circular import: from Main import GCNClassifier


def run(features, labels, folds, rks, gcn_type, pNNeurons, GCNClassifier, k=10, graph_type="knn"):

    results = []
    count = 0
    for test_index, train_index in folds:
        count += 1

        # Handle PyTorch tensor properly
        if torch.is_tensor(features):
            # Use PyTorch indexing for tensors
            train_features = features[train_index]
            test_features = features[test_index]
        else:
            # Use numpy indexing for numpy arrays
            train_features = np.array([features[i] for i in train_index], dtype=np.float32)
            test_features = np.array([features[i] for i in test_index], dtype=np.float32)
            
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        # Create GCNClassifier instance
        # GCNClassifier(gcn_type, rks, pN, k, pNNeurons, graph_type="knn")
        clf = GCNClassifier(gcn_type, rks, len(features), k, pNNeurons, graph_type=graph_type)
        clf.fit(test_index, train_index, features, labels)
        pred = clf.predict()

        # Append current result
        results.append([pred, test_labels, evaluation(pred, test_labels)])

    return results


def evaluation(pred, labels):
    acc = 0
    n = len(pred)
    for i in range(n):
        if pred[i] == labels[i]:
            acc += 1
    return acc / n
