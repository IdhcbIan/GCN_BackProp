import torch
import numpy as np
from PIL import Image
from torchvision import transforms

"""
██████╗░██╗███╗░░██╗░█████╗░██╗░░░██╗██████╗░
██╔══██╗██║████╗░██║██╔══██╗██║░░░██║╚════██╗
██║░░██║██║██╔██╗██║██║░░██║╚██╗░██╔╝░░███╔═╝
██║░░██║██║██║╚████║██║░░██║░╚████╔╝░██╔══╝░░
██████╔╝██║██║░╚███║╚█████╔╝░░╚██╔╝░░███████╗
╚═════╝░╚═╝╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚══════╝
// Ian Bezerra - 2025 //
// Adapted for DINOv3 batch feature extraction (inference) //
"""

def dinov3_inference(model_name, image_paths, input_size=224):
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv3 model from the official repo
    dinov3_model = torch.hub.load('facebookresearch/dinov3:main', model_name)
    dinov3_model.to(device)
    dinov3_model.eval()
    
    # Transforms similar to DINOv2, but adjustable for DINOv3 variants (e.g., patch size 14 or 16 often uses 224)
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # Slight upscale before crop, common for ViTs
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Process images in smaller batches
    batch_size = len(image_paths) // 10  # Adjust based on your GPU memory
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Process images in the current batch
        input_tensors = []
        for img_path in batch_paths:
            # Load and transform each image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            input_tensors.append(img_tensor)
        
        # Stack the tensors for the current batch
        input_batch = torch.stack(input_tensors)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            # For DINOv3, extract the CLS token features (similar to DINOv2)
            features = dinov3_model(input_batch)
        
        # Move results to CPU immediately to free GPU memory
        features = features.cpu().numpy()
        all_features.append(features)
        
        # Clear some memory
        del input_batch
        torch.cuda.empty_cache()
    
    features_batch = np.vstack(all_features)
    return features_batch

