import modal
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange
import os
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


# Build Modal Image including local Python source code
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "torchvision", 
        "tqdm", 
        "timm==0.9.12", 
        "einops==0.7.0", 
        "pillow", 
        "numpy",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "seaborn",
        "evaluate"

    )
    .pip_install(
        "torch-geometric",
    )
    .add_local_file("Tools.py", "/root/Tools.py")
    .add_local_file("utils.py", "/root/utils.py")
    .add_local_file("differentiable_ranking.py", "/root/differentiable_ranking.py")
)

# Define Modal App with dataset volume
app = modal.App(
    "BackProp Though Rakings!! Attempt 1!",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("main")}
)



@app.function(
    gpu="L4:1",  # Single GPU for evaluation
    timeout=2400,  # 40 minutes timeout
    volumes={"/mnt/data": modal.Volume.from_name("main")}
)
def main():
    # Import all needed modules
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import ARMAConv
    import sys
    sys.path.append('/root')
    import Tools
    import utils
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # DinoV2 model setup
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14.eval()
    dinov2_vits14 = dinov2_vits14.to(device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Load images from the uploaded imgs directory
    images_dir = "/mnt/data/imgs"
    image_paths = [
        os.path.join(images_dir, fname)
        for fname in os.listdir(images_dir)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ]
    
    # Load and process images
    input_batch = []
    for path in image_paths:
        if os.path.exists(path):
            image = Image.open(path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            input_tensor = transform(image)
            input_batch.append(input_tensor)
        else:
            print(f"Warning: {path} not found")
    
    # Create batch and limit to 20 images for testing
    input_batch = torch.stack(input_batch)
    input_batch = input_batch[:20]
    input_batch.requires_grad_(True)
    input_batch = input_batch.to(device)
    
    # Extract features
    print(f"Processing batch of {len(input_batch)} images...")
    features = dinov2_vits14(input_batch)
    print(f"Features shape: {features.shape}")
    
    # Differentiable ranking function (Gumbel-based)
    def gumbel_topk_ranking(features, k, temperature=1.0):
        # Calculate pairwise distances
        dist_matrix = torch.cdist(features, features, p=2)
        
        # Convert distances to similarities (negative distances)
        similarities = -dist_matrix
        
        # Add Gumbel noise for differentiability
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarities) + 1e-8) + 1e-8)
        gumbel_logits = (similarities + temperature * gumbel_noise) / temperature
        
        # Use Gumbel-Softmax to get soft top-k approximation
        # This returns a differentiable soft ranking matrix
        soft_rankings = F.gumbel_softmax(gumbel_logits, tau=temperature, hard=False, dim=-1)
        
        # Get the k highest probability indices for each row (for graph construction)
        _, hard_indices = torch.topk(soft_rankings, k=k, dim=1, largest=True)
        
        # Return both soft rankings (for backprop) and hard indices (for graph construction)
        return soft_rankings, hard_indices
    
    # Test differentiable ranking
    k = 2
    soft_rankings, hard_indices = gumbel_topk_ranking(features, k)
    
    # Test backward pass on ranking (using soft rankings which are differentiable)
    try:
        soft_rankings.sum().backward()
        print("✅ Backprop through ranking successful!")
    except Exception as e:
        print(f"❌ Backprop failed: {e}")
        return {"error": str(e)}
    
    # GCN Classes (simplified ARMA)
    class ARMA(torch.nn.Module):
        def __init__(self, num_features, num_classes):
            super(ARMA, self).__init__()
            self.conv1 = ARMAConv(num_features, 32, num_stacks=1, num_layers=1)
            self.conv2 = ARMAConv(32, num_classes, num_stacks=1, num_layers=1)
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    # Load labels from uploaded files
    with open("/mnt/data/list.txt", "r") as f:
        images = [line.strip() for line in f if line.strip()]
    
    label_dict = {}
    with open("/mnt/data/Classes.txt", "r") as lf:
        for line in lf:
            line = line.strip()
            if not line or ":" not in line:
                continue
            img, label = line.split(":", 1)
            label_dict[img.strip()] = int(label.strip())
    
    # Create labels for current batch (first 20 images)
    labels = []
    for i in range(min(20, len(images))):
        img_name = images[i]
        if img_name in label_dict:
            labels.append(label_dict[img_name])
        else:
            labels.append(0)  # Default label if not found
    
    # Pad or truncate labels to match features
    while len(labels) < len(features):
        labels.append(0)
    labels = labels[:len(features)]
    
    # Create simple graph from rankings for GCN
    edge_index = []
    for img1 in range(len(features)):
        for pos in range(min(k, hard_indices.shape[1])):
            img2 = int(hard_indices[img1][pos].item())
            edge_index.append([img1, img2])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # Create GCN data object
    data = Data(
        x=features.detach(),  # Use detached features for GCN training
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long).to(device)
    )
    
    # Simple train/test split
    num_nodes = len(features)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_nodes//2] = True
    test_mask[num_nodes//2:] = True
    
    data.train_mask = train_mask.to(device)
    data.test_mask = test_mask.to(device)
    
    # Train GCN
    num_classes = len(set(labels))
    model = ARMA(features.shape[1], max(2, num_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("Training GCN...")
    model.train()
    for epoch in range(10):  # Quick training for testing
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        pred = model(data)
        pred_labels = pred.argmax(dim=1)
        
        # Calculate accuracy on test set
        test_acc = (pred_labels[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"Test accuracy: {test_acc.item():.4f}")
    
    return {
        "status": "success",
        "features_shape": features.shape,
        "soft_rankings_shape": soft_rankings.shape,
        "hard_indices_shape": hard_indices.shape,
        "test_accuracy": test_acc.item(),
        "num_images": len(input_batch),
        "backprop_successful": True
    }


if __name__ == "__main__":
    # Run the modal function
    with app.run():
        result = main.remote()
        print(f"Results: {result}")
        
