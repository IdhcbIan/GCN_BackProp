import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import config


def load_classes(classes_file=config.CLASSES_FILE):
    """
    Load image filenames and their corresponding class labels.
    """
    image_names = []
    labels = []

    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                img_name, label = line.split(':')
                image_names.append(img_name)
                labels.append(int(label))

    return image_names, labels


def get_image_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMG_NORMALIZE_MEAN,
            std=config.IMG_NORMALIZE_STD
        )
    ])
    return transform


def load_images(image_names, data_dir=config.DATA_DIR):
    """
    Load images from disk and apply preprocessing.
    """
    transform = get_image_transform()
    images = []

    for img_name in image_names:
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)

    images_tensor = torch.stack(images)
    return images_tensor


def fold_split(features, labels, n_folds=config.N_FOLDS):
    """
    Split data into stratified k-folds for cross-validation.
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    folds = list(kf.split(features, labels))
    return folds


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    """
    correct = sum(pred == label for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)
    return accuracy


def create_train_val_test_masks(n_samples, train_indices, test_indices, val_indices=None):
    """
    Create boolean masks for train/val/test splits.
    """
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    if val_indices is not None:
        val_mask[val_indices] = True

    return train_mask, val_mask, test_mask


def print_experiment_config():
    """Print experiment configuration for reproducibility."""
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"DINOv2 model: {config.DINO_MODEL}")
    print(f"Feature dimension: {config.FEATURE_DIM}")
    print("-" * 70)
    print(f"K neighbors: {config.K_NEIGHBORS}")
    print(f"Ranking method: Soft Top-K (differentiable)")
    print(f"Ranking temperature: {config.RANKING_TEMPERATURE}")
    print(f"Ranking metric: {config.RANKING_METRIC}")
    print("-" * 70)
    print(f"GCN type: {config.GCN_TYPE}")
    print(f"Hidden dim: {config.GCN_HIDDEN_DIM}")
    print(f"Dropout: {config.GCN_DROPOUT}")
    print("-" * 70)
    print(f"Number of folds: {config.N_FOLDS}")
    print(f"Number of executions: {config.N_EXECUTIONS}")
    print(f"Epochs: {config.N_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print("=" * 70)
    print()
