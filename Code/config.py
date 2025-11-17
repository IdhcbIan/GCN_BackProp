"""
Configuration file for GCN Backpropagation experiments.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_DIR = "../Data"
CLASSES_FILE = "../Data/Classes.txt"

# DINOv2 model configuration
DINO_MODEL = "dinov2_vitb14"  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14
FEATURE_DIM = 768

# Image preprocessing
IMG_SIZE = 224
IMG_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMG_NORMALIZE_STD = [0.229, 0.224, 0.225]

# Graph construction - Soft Top-K Differentiable Ranking
K_NEIGHBORS = 10 
RANKING_TEMPERATURE = 1.0 
RANKING_METRIC = 'cosine'

# GCN model configuration
GCN_TYPE = "arma" 
GCN_HIDDEN_DIM = 16
GCN_OUTPUT_DIM = 32
GCN_NUM_STACKS = 3
GCN_NUM_LAYERS = 2
GCN_DROPOUT = 0.25

# Training configuration
N_FOLDS = 5 
N_EXECUTIONS = 1
N_EPOCHS = 200
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4

# Logging
VERBOSE = True
PRINT_EVERY_N_EPOCHS = 10
