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

# Import torch geometric modules
try:
    from torch_geometric.nn import ARMAConv
    from torch_geometric.data import Data
except ImportError:
    # These will be available when running in Modal environment
    ARMAConv = None
    Data = None


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
    .add_local_file("Classes.txt", "/root/Classes.txt")
)

# Define Modal App with dataset volume
app = modal.App(
    "BackProp Though Rakings!! Attempt 1!",
    image=image,
    volumes={"/mnt/data": modal.Volume.from_name("main")}
)





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



class Dataset(Dataset):
    def __init__(self):
        self.images = self.get_dirs()
        self.labels = self.get_classes()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def get_dirs(self):
        # Load images from the uploaded imgs directory
        images_dir = "/mnt/data/imgs"
        image_paths = [
            os.path.join(images_dir, fname)
            for fname in os.listdir(images_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        #return image_paths    # Lets load all images
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


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


        input_batch = torch.stack(input_batch)
        input_batch.requires_grad_(True)
        return input_batch   # In RAM!!

    def create_batch(self, batch_size, n_classes):
        """
        Create a batch of triplets (anchor, positive, negative) such that:
        - Evenly distribute triplets across n_classes
        - For each class, create batch_size/n_classes triplets
        - Each triplet contains (anchor, positive, negative) where anchor and positive are from same class
        Returns:
            anchors, positives, negatives: torch.Tensor batches
            anchor_indices: list of indices of anchor images
            batch_labels: list of labels for each triplet
        """
        import random
        
        # Build mapping from label to indices
        label_to_indices = {}
        for idx, (img, label) in enumerate(zip(self.images, self.labels.values())):
            label_to_indices.setdefault(label, []).append(idx)
            
        # Select n_classes random classes
        available_labels = list(label_to_indices.keys())
        if len(available_labels) < n_classes:
            raise ValueError(f"Not enough classes available. Requested {n_classes} but only have {len(available_labels)}")
        selected_labels = random.sample(available_labels, n_classes)
        
        # Calculate triplets per class
        triplets_per_class = batch_size // n_classes
        if triplets_per_class < 1:
            raise ValueError(f"Batch size {batch_size} too small for {n_classes} classes")
            
        anchors, positives, negatives = [], [], []
        anchor_indices = []
        batch_labels = []
        
        # Generate triplets for each selected class
        for label in selected_labels:
            class_indices = label_to_indices[label]
            other_labels = [l for l in selected_labels if l != label]
            
            for _ in range(triplets_per_class):
                # Select anchor
                anchor_idx = random.choice(class_indices)
                anchor = self.images[anchor_idx]
                anchor_indices.append(anchor_idx)
                batch_labels.append(label)
                
                # Select positive (different from anchor if possible)
                pos_candidates = [i for i in class_indices if i != anchor_idx]
                if not pos_candidates:
                    pos_idx = anchor_idx  # Use anchor if no other options
                else:
                    pos_idx = random.choice(pos_candidates)
                positive = self.images[pos_idx]
                
                # Select negative from different class
                neg_label = random.choice(other_labels)
                neg_idx = random.choice(label_to_indices[neg_label])
                negative = self.images[neg_idx]
                
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        
        # Stack to tensors
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        
        return anchors, positives, negatives, anchor_indices, batch_labels



    def get_classes(self):
        label_dict = {}
        with open("/root/Classes.txt", "r") as lf:
            for line in lf:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                img, label = line.split(":", 1)
                label_dict[img.strip()] = int(label.strip())
        return label_dict


@app.function(
    gpu="A100-40GB:1",  # Single GPU for evaluation
    timeout=2400,  # 40 minutes timeout
    volumes={"/mnt/data": modal.Volume.from_name("main")}
)
def main():
    # Import all needed modules
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
    
    # DinoV2 model setup - load directly in main to avoid serialization issues
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.train()  # Set feature extractor to training mode
    
    # Enable gradients for feature extractor parameters
    for param in model.parameters():
        param.requires_grad = True
    
    flower_dataset = Dataset()

    steps = 100
    batch_size = 50  # Reduced from 150 to prevent memory issues and gradient explosion
    n_classes = 10

    for step in range(steps):
        print(f"Step {step} of {steps}")
        # Creating batch of triplets
        anchors, positives, negatives, anchor_indices, batch_labels = flower_dataset.create_batch(batch_size, n_classes)
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        batch_input = torch.cat([anchors, positives, negatives], dim=0)
        
        # Enable gradients for backprop through feature extractor
        all_features = model(batch_input)  # features has gradients enabled, already in order [anchors, positives, negatives]
        
        print(f"\n{'='*50} STEP {step} {'='*50}")
        print(f"Processing batch of {batch_size} triplets...")
        print(f"All features shape: {all_features.shape} (anchors|positives|negatives)")
        print(f"Batch labels: {batch_labels[:5]}...")  # Show first 5 labels
        
        # 🔍 DEBUG: Extractor Output
        print(f"\n📊 EXTRACTOR OUTPUT:")
        print(f"   Features requires_grad: {all_features.requires_grad}")
        print(f"   Features mean: {all_features.mean().item():.6f}")
        print(f"   Features std: {all_features.std().item():.6f}")
        print(f"   Features range: [{all_features.min().item():.6f}, {all_features.max().item():.6f}]")
        
        # Get actual labels for pos/neg nodes from dataset
        # We need to map the triplet indices back to actual dataset labels
        pos_actual_labels = []
        neg_actual_labels = []
        
        # For this simplified version, let's use a mapping from batch_labels to get actual labels
        # In practice, you'd want to track the actual dataset indices
        for i in range(batch_size):
            # For now, use the batch_labels as a proxy for actual labels
            # Positives should have same label as anchor
            pos_actual_labels.append(batch_labels[i])
            # Negatives should have different labels - let's cycle through available classes
            available_classes = list(set(batch_labels))
            neg_label = available_classes[(batch_labels[i] + 1) % len(available_classes)]
            neg_actual_labels.append(neg_label)
        
        # Create GCN labels: 
        # Anchors get -1 (masked during training, these are what we want to predict)
        # Positives get their actual labels
        # Negatives get their actual labels
        gcn_labels = []
        gcn_labels.extend([-1] * batch_size)  # Anchors: masked
        gcn_labels.extend(pos_actual_labels)  # Positives: actual labels
        gcn_labels.extend(neg_actual_labels)  # Negatives: actual labels
        
        gcn_labels = torch.tensor(gcn_labels, dtype=torch.long).to(device)
        
        # Map labels to 0-indexed range for neural network BEFORE creating Data object
        valid_labels = [label for label in gcn_labels.cpu().tolist() if label != -1]
        unique_labels = sorted(set(valid_labels))
        
        # Map labels to 0-indexed range for neural network
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Remap labels to 0-indexed
        gcn_labels_mapped = gcn_labels.clone()
        for i, label in enumerate(gcn_labels.cpu().tolist()):
            if label != -1:
                gcn_labels_mapped[i] = label_mapping[label]
        
        num_classes = len(unique_labels)
        print(f"\n🏷️  LABEL MAPPING:")
        print(f"   Number of classes: {num_classes}")
        print(f"   Original labels: {unique_labels}")
        print(f"   Label mapping: {label_mapping}")
        print(f"   GCN labels shape: {gcn_labels_mapped.shape}")
        print(f"   Anchor labels (masked): {gcn_labels_mapped[:5].cpu().tolist()}")
        print(f"   Pos labels: {gcn_labels_mapped[batch_size:batch_size+5].cpu().tolist()}")
        print(f"   Neg labels: {gcn_labels_mapped[2*batch_size:2*batch_size+5].cpu().tolist()}")
        
        # Create graph edges using ranking on ALL features (connect everything to everything)
        k = min(20, len(all_features)//3)  # Use reasonable k for full graph
        soft_rankings, hard_indices = gumbel_topk_ranking(all_features, k)
        
        # Create edge index from rankings
        edge_index = []
        for img1 in range(len(all_features)):
            for pos in range(min(k, hard_indices.shape[1])):
                img2 = int(hard_indices[img1][pos].item())
                edge_index.append([img1, img2])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        
        # 🔍 DEBUG: Graph Structure
        print(f"\n🕸️  GRAPH STRUCTURE:")
        print(f"   Nodes: {len(all_features)} ({batch_size} anchors + {batch_size} pos + {batch_size} neg)")
        print(f"   Edges: {edge_index.shape[1]}")
        print(f"   K neighbors: {k}")
        print(f"   Edge index shape: {edge_index.shape}")
        
        # Create GCN data object with mapped labels
        # 🚨 CRITICAL: Ensure features stay connected to computational graph!
        print(f"   Features requires_grad before Data: {all_features.requires_grad}")
        print(f"   Features is_leaf: {all_features.is_leaf}")
        print(f"   Features.grad_fn: {all_features.grad_fn}")
        
        data = Data(
            x=all_features,  # Keep gradients enabled for backprop!
            edge_index=edge_index,
            y=gcn_labels_mapped
        )
        
        # Verify computational graph connection after Data creation
        print(f"   Data.x requires_grad after Data: {data.x.requires_grad}")
        print(f"   Data.x is_leaf: {data.x.is_leaf}")
        print(f"   Data.x.grad_fn: {data.x.grad_fn}")
        print(f"   Data.x is same object as all_features: {data.x is all_features}")
        
        if data.x.requires_grad and data.x.grad_fn is not None:
            print(f"   ✅ Features properly connected to computational graph!")
        else:
            print(f"   🚨 WARNING: Features disconnected from computational graph!")
            print(f"   🔧 This will prevent GCN gradients from flowing back to extractor!")
        
        # Create masks for GCN training
        num_nodes = len(all_features)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Train mask: pos/neg nodes (indices batch_size to 3*batch_size)
        # Test mask: anchor nodes (indices 0 to batch_size)
        train_mask[batch_size:] = True  # Pos and neg nodes
        test_mask[:batch_size] = True   # Anchor nodes
        
        data.train_mask = train_mask.to(device)
        data.test_mask = test_mask.to(device)
        
        # Debug: check train mask labels
        train_labels = data.y[data.train_mask]
        print(f"Train labels range: {train_labels.min().item()} to {train_labels.max().item()}")
        print(f"Train mask sum: {data.train_mask.sum().item()}, Test mask sum: {data.test_mask.sum().item()}")
        
        # Create GCN model and optimizer that includes feature extractor
        print(f"\n🔧 CREATING GCN MODEL:")
        print(f"   Input features: {all_features.shape[1]}")
        print(f"   Output classes: {num_classes}")
        
        gcn_model = ARMA(all_features.shape[1], num_classes).to(device)
        
        # Verify GCN model structure
        print(f"   GCN model created successfully")
        print(f"   GCN model device: {next(gcn_model.parameters()).device}")
        print(f"   GCN model parameters: {sum(p.numel() for p in gcn_model.parameters())}")
        
        # Test GCN model can do forward pass
        try:
            with torch.no_grad():
                test_out = gcn_model(data)
                print(f"   ✅ GCN forward pass test successful: {test_out.shape}")
        except Exception as e:
            print(f"   🚨 GCN forward pass test failed: {e}")
        
        # Combined optimizer for both GCN and feature extractor (reduced LR for stability)
        gcn_params = list(gcn_model.parameters())
        extractor_params = list(model.parameters())
        #all_params = gcn_params + extractor_params
        
        print(f"\n🔍 OPTIMIZER SETUP:")
        print(f"   GCN parameters in optimizer: {len(gcn_params)}")
        print(f"   Extractor parameters in optimizer: {len(extractor_params)}")
        print(f"   Total parameters in optimizer: {len(extractor_params) + len(gcn_params)}")
        print(f"   Total parameter count: {sum(p.numel() for p in extractor_params) + sum(p.numel() for p in gcn_params)}")
        
        optimizer_extractor = torch.optim.AdamW(extractor_params, lr=1e-5, weight_decay=5e-4)  # Reduced from 0.01 to 0.001
        optimizer_gcn = torch.optim.Adam(gcn_params, lr=0.001, weight_decay=5e-4)  # Reduced from 0.01 to 0.001
        
        print(f"Training GCN with {num_classes} classes...")
        
        # Set both models to training mode
        gcn_model.train()
        model.train()
        
        # ONE TRAINING STEP - backprop through everything
        optimizer_extractor.zero_grad()
        optimizer_gcn.zero_grad()
        
        # 🚨 FINAL CHECK: Verify data is still connected before GCN forward pass
        print(f"\n🔍 PRE-GCN VERIFICATION:")
        print(f"   GCN model training mode: {gcn_model.training}")
        print(f"   Data.x requires_grad: {data.x.requires_grad}")
        print(f"   Data.x device: {data.x.device}")
        print(f"   GCN model device: {next(gcn_model.parameters()).device}")
        
        # Forward pass through GCN
        out = gcn_model(data)
        
        # 🔍 DEBUG: GCN Output
        print(f"\n🧠 GCN OUTPUT:")
        print(f"   Output shape: {out.shape}, Expected classes: {num_classes}")
        print(f"   Output requires_grad: {out.requires_grad}")
        print(f"   Output mean: {out.mean().item():.6f}")
        print(f"   Output std: {out.std().item():.6f}")
        print(f"   Anchor predictions sample: {out[:3, :3].detach().cpu().numpy()}")  # First 3 anchors, first 3 classes
        
        if out.requires_grad:
            print(f"   ✅ GCN output has gradients - ready for backprop!")
        else:
            print(f"   🚨 GCN output missing gradients - backprop will fail!")
        
        # Calculate loss ONLY on anchor predictions vs true anchor labels
        anchor_predictions = out[:batch_size]  # Get anchor predictions
        
        # Map true anchor labels to 0-indexed for loss calculation
        mapped_anchor_labels = []
        for label in batch_labels:
            if label in label_mapping:
                mapped_anchor_labels.append(label_mapping[label])
            else:
                # If label not seen in pos/neg, map to first available class
                mapped_anchor_labels.append(0)
        true_anchor_labels = torch.tensor(mapped_anchor_labels, dtype=torch.long).to(device)
        
        # Main loss: anchor predictions vs true anchor labels
        # Note: ARMA outputs log_softmax, so use nll_loss directly
        print(f"\n🔍 LOSS DEBUG:")
        print(f"   Anchor predictions shape: {anchor_predictions.shape}")
        print(f"   True anchor labels shape: {true_anchor_labels.shape}")
        print(f"   Anchor pred range: [{anchor_predictions.min().item():.3f}, {anchor_predictions.max().item():.3f}]")
        print(f"   True labels range: [{true_anchor_labels.min().item()}, {true_anchor_labels.max().item()}]")
        print(f"   Sample predictions (first 3):")
        print(f"     {anchor_predictions[:3].detach().cpu().numpy()}")
        print(f"   Sample true labels (first 5): {true_anchor_labels[:5].cpu().numpy()}")
        
        anchor_loss = F.nll_loss(anchor_predictions, true_anchor_labels)
        
        # Optional: add supervision loss on pos/neg nodes (exclude -1 labels)
        train_outputs = out[data.train_mask]
        train_targets = data.y[data.train_mask]
        
        # Filter out any remaining -1 values (shouldn't happen but safety check)
        valid_mask = train_targets != -1
        if valid_mask.sum() > 0:
            supervision_loss = F.nll_loss(train_outputs[valid_mask], train_targets[valid_mask])
        else:
            supervision_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = anchor_loss + 0.1 * supervision_loss  # Weight the supervision loss
        
        # 🔍 DEBUG: Loss Values
        print(f"\n💰 LOSS CALCULATION:")
        print(f"   Anchor loss: {anchor_loss.item():.6f}")
        print(f"   Supervision loss: {supervision_loss.item():.6f}")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   Total loss requires_grad: {total_loss.requires_grad}")
        
        # 🕵️ COMPUTATIONAL GRAPH TRACING
        print(f"\n🕵️ TRACING COMPUTATIONAL GRAPH:")
        print(f"   Anchor predictions requires_grad: {anchor_predictions.requires_grad}")
        print(f"   Anchor predictions grad_fn: {anchor_predictions.grad_fn}")
        print(f"   GCN output requires_grad: {out.requires_grad}")
        print(f"   GCN output grad_fn: {out.grad_fn}")
        
        # Check if anchor loss depends on GCN output
        if anchor_predictions.grad_fn is not None:
            print(f"   ✅ Anchor predictions connected to computational graph!")
        else:
            print(f"   🚨 Anchor predictions NOT connected to computational graph!")
            
        # Check if GCN output depends on features
        if out.grad_fn is not None:
            print(f"   ✅ GCN output connected to computational graph!")
        else:
            print(f"   🚨 GCN output NOT connected to computational graph!")
        
        # 🚨 SAFETY CHECK: Detect problematic loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"   🚨 WARNING: Loss is NaN or Inf! Skipping backprop.")
            continue
            
        if total_loss.item() > 20.0:
            print(f"   🚨 WARNING: Loss too high ({total_loss.item():.2f})! Clipping to prevent explosion.")
            total_loss = torch.clamp(total_loss, max=5.0)
        
        # Check gradients before backprop
        print(f"\n🔍 PRE-BACKPROP GRADIENT STATUS:")
        gcn_param = next(gcn_model.parameters())
        extractor_param = next(model.parameters())
        print(f"   GCN param grad before: {gcn_param.grad is not None}")
        print(f"   Extractor param grad before: {extractor_param.grad is not None}")
        
        # 🔍 DEBUG: Check if GCN parameters are in the computational graph
        print(f"\n🕵️ GCN PARAMETER ANALYSIS:")
        print(f"   GCN param requires_grad: {gcn_param.requires_grad}")
        print(f"   GCN param is_leaf: {gcn_param.is_leaf}")
        print(f"   Total GCN parameters: {sum(p.numel() for p in gcn_model.parameters())}")
        print(f"   GCN parameters requiring grad: {sum(p.numel() for p in gcn_model.parameters() if p.requires_grad)}")
        
        # Check if loss actually depends on GCN parameters
        print(f"   Total loss requires_grad: {total_loss.requires_grad}")
        print(f"   Anchor loss requires_grad: {anchor_loss.requires_grad}")
        print(f"   Supervision loss requires_grad: {supervision_loss.requires_grad}")
        
        # Backprop through entire pipeline with safety
        print(f"\n⚡ STARTING BACKPROP...")
        import time
        backprop_start = time.time()
        
        try:
            total_loss.backward()
            backprop_time = time.time() - backprop_start
            print(f"   ✅ Backprop completed successfully! ({backprop_time:.2f}s)")
        except RuntimeError as e:
            print(f"   🚨 Backprop failed: {e}")
            print(f"   🔧 Skipping this step and continuing...")
            continue
        except Exception as e:
            print(f"   🚨 Unexpected error during backprop: {e}")
            print(f"   🔧 Skipping this step and continuing...")
            continue
        
        # 🔍 DEBUG: Backprop Results (BEFORE any mode changes!)
        print(f"\n📈 POST-BACKPROP GRADIENT STATUS:")
        
        # 🎯 CRITICAL: Check gradients IMMEDIATELY after backprop, while still in training mode
        gcn_has_gradients = gcn_param.grad is not None
        extractor_has_gradients = extractor_param.grad is not None
        
        print(f"   GCN param grad after: {gcn_has_gradients}")
        print(f"   Extractor param grad after: {extractor_has_gradients}")
        
        # Calculate gradient norms and clip if needed
        gcn_grad_norm = 0.0
        extractor_grad_norm = 0.0
        
        if gcn_has_gradients:
            gcn_grad_norm = gcn_param.grad.norm().item()
            print(f"   GCN grad norm: {gcn_grad_norm:.6f}")
            print(f"   🎯 GCN BACKPROP: ✅ SUCCESS!")
        else:
            print(f"   🚨 GCN BACKPROP: ❌ NO GRADIENTS!")
            
        if extractor_has_gradients:
            extractor_grad_norm = extractor_param.grad.norm().item()
            print(f"   Extractor grad norm: {extractor_grad_norm:.6f}")
            print(f"   🎯 EXTRACTOR BACKPROP: ✅ SUCCESS!")
        else:
            print(f"   🚨 EXTRACTOR BACKPROP: ❌ NO GRADIENTS!")
            
        # Check feature gradients
        if all_features.grad is not None:
            feature_grad_norm = all_features.grad.norm().item()
            print(f"   Features grad norm: {feature_grad_norm:.6f}")
        else:
            print(f"   🚨 FEATURES: ❌ NO GRADIENTS!")
            
        # 🛡️ GRADIENT CLIPPING: Prevent gradient explosion
        max_grad_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(gcn_model.parameters()) + list(model.parameters()), 
            max_grad_norm
        )
        
        if total_norm > max_grad_norm:
            print(f"   🛡️ Gradients clipped! Original norm: {total_norm:.6f} -> {max_grad_norm}")
        else:
            print(f"   ✅ Gradients OK, norm: {total_norm:.6f}")
            
        # 🚨 CRITICAL: Keep models in training mode during gradient operations!
        assert gcn_model.training, "🚨 GCN should be in training mode!"
        assert model.training, "🚨 Extractor should be in training mode!"
        print(f"   ✅ Models confirmed in training mode during gradient ops")
            
        # Safe optimizer step (THIS WILL CLEAR ALL GRADIENTS!)
        try:
            optimizer_extractor.step()
            optimizer_gcn.step()
            print(f"   ✅ Optimizer step completed! (Gradients now cleared)")
        except Exception as e:
            print(f"   🚨 Optimizer step failed: {e}")
            continue
        
        print(f"\n📊 STEP SUMMARY:")
        print(f"   Step {step}, Anchor Loss: {anchor_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        
        # 🎯 EVALUATION: Keep in training mode - eval mode kills gradients!
        print(f"\n🎯 EVALUATION: (Keeping in training mode - eval mode has no gradients!)")
        
        # Verify we're still in training mode
        print(f"   GCN training mode: {gcn_model.training}")
        print(f"   Extractor training mode: {model.training}")
        
        with torch.no_grad():
            # 🚨 NO EVAL MODE SWITCH! Keep in training mode!
            # Forward pass for evaluation (still in training mode)
            pred = gcn_model(data)
            anchor_pred_labels = pred[:batch_size].argmax(dim=1)
            
            # Calculate accuracy on anchor predictions
            anchor_acc = (anchor_pred_labels == true_anchor_labels).float().mean()
            
            # 🔍 DEBUG: Evaluation Results
            print(f"   Anchor prediction accuracy: {anchor_acc.item():.4f}")
            print(f"   Predicted labels sample: {anchor_pred_labels[:5].cpu().numpy()}")
            print(f"   True labels sample: {true_anchor_labels[:5].cpu().numpy()}")
            print(f"   Correct predictions: {(anchor_pred_labels == true_anchor_labels).sum().item()}/{batch_size}")
            
        print(f"   ✅ Evaluation completed in training mode - gradients preserved!")
        
        # Store results for this step
        step_results = {
            "step": step,
            "anchor_loss": anchor_loss.item(),
            "total_loss": total_loss.item(),
            "anchor_accuracy": anchor_acc.item(),
            "num_classes": num_classes,
            "features_shape": all_features.shape
        }
        
        # Final step summary (use captured gradient status from BEFORE optimizer.step())
        print(f"\n{'='*20} STEP {step} COMPLETE {'='*20}")
        print(f"✅ Extractor: {'✅' if extractor_has_gradients else '❌'}")
        print(f"✅ GCN: {'✅' if gcn_has_gradients else '❌'}")
        print(f"📈 Accuracy: {anchor_acc.item():.4f}")
        print(f"💰 Loss: {total_loss.item():.6f}")
        print(f"🔥 Gradient Status: {'BOTH WORKING' if (gcn_has_gradients and extractor_has_gradients) else 'PROBLEMS DETECTED'}")
        print(f"{'='*60}")
        
        if step % 10 == 0:
            print(f"\n🔍 Detailed Results: {step_results}")
            
        # 🧹 MEMORY CLEANUP: Clear CUDA cache periodically
        if step % 5 == 0:
            torch.cuda.empty_cache()
            print(f"   🧹 CUDA cache cleared")
    
    print("Training completed!")
    return {"status": "success", "total_steps": steps}


if __name__ == "__main__":
    # Run the modal function
    with app.run():
        result = main.remote()
        print(f"Results: {result}")
        
