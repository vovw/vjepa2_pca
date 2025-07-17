#!/usr/bin/env python3

import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IMG_DIR = "images"
IMG_SIZE = 224
VJEPA_SIZE = 256

def load_models():
    """Load DINO and V-JEPA models"""
    print("Loading DINO-v2...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_model = dino_model.cuda().eval()
    
    print("Loading V-JEPA-2...")
    try:
        # Use HuggingFace transformers instead of torch.hub
        from transformers import AutoModel, AutoVideoProcessor
        
        vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        vjepa_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        
        vjepa_model = vjepa_model.cuda().eval()
        return dino_model, vjepa_model, vjepa_processor
    except Exception as e:
        print(f"V-JEPA loading failed: {e}")
        return dino_model, None, None

def get_dino_features(model, image):
    """Extract DINO features"""
    with torch.no_grad():
        features = model.forward_features(image)["x_norm_patchtokens"]
    return features[0].cpu().numpy()

def get_vjepa_features(model, processor, image_pil):
    """Extract V-JEPA features using HuggingFace processor"""
    # Resize image
    img_resized = image_pil.resize((VJEPA_SIZE, VJEPA_SIZE))
    img_array = np.array(img_resized)
    
    # Create video with 64 frames
    frames = 64
    video_np = np.stack([img_array] * frames, axis=0)  # (64, H, W, 3)
    
    # Use HuggingFace processor
    inputs = processor(video_np, return_tensors="pt")
    
    # Debug: Check what keys are available
    print(f"Processor output keys: {list(inputs.keys())}")
    
    # Try different possible keys
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values'].cuda()
    elif 'pixel_values_videos' in inputs:
        pixel_values = inputs['pixel_values_videos'].cuda()
    else:
        # If no expected key, try the first tensor
        key = list(inputs.keys())[0]
        pixel_values = inputs[key].cuda()
        print(f"Using key: {key}")
    
    print(f"V-JEPA input shape: {pixel_values.shape}")
    
    with torch.no_grad():
        outputs = model(pixel_values)
        features = outputs.last_hidden_state
    
    # Get features and handle shape
    features_np = features[0].cpu().numpy()
    print(f"V-JEPA features shape: {features_np.shape}")
    
    # V-JEPA-2 outputs spatiotemporal patches
    # For 64 frames with tubelet_size=2: 32 temporal patches
    # For 256x256 with patch_size=16: 16x16 = 256 spatial patches per frame
    # Total: 32 * 256 = 8192 patches
    
    if features_np.shape[0] == 8192:
        # Reshape to (temporal, spatial, features)
        features_3d = features_np.reshape(32, 256, features_np.shape[1])
        # Average over temporal dimension to get spatial features
        features_spatial = np.mean(features_3d, axis=0)
        print(f"Averaged temporal patches: {features_3d.shape} -> {features_spatial.shape}")
        return features_spatial
    else:
        # Handle other cases
        print(f"Using features as-is: {features_np.shape}")
        return features_np

def create_pca_viz(features, title):
    """Create PCA visualization"""
    # PCA with 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # Normalize to [0,1]
    pca_img = pca_features[:, :3].copy()
    for i in range(3):
        ch = pca_img[:, i]
        pca_img[:, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    
    # Reshape to grid
    n_patches = pca_img.shape[0]
    grid_size = int(np.sqrt(n_patches))
    
    # Pad if needed
    if grid_size * grid_size != n_patches:
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        pad_size = grid_size * grid_size - n_patches
        if pad_size > 0:
            padding = np.zeros((pad_size, 3))
            pca_img = np.vstack([pca_img, padding])
    
    pca_img = pca_img.reshape(grid_size, grid_size, 3)
    
    print(f"{title} - PCA variance: {pca.explained_variance_ratio_[:3]}")
    return pca_img

def process_image(img_path, dino_model, vjepa_model, vjepa_processor):
    """Process single image"""
    img = Image.open(img_path).convert('RGB')
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    print(f"\nProcessing: {img_name}")
    
    # DINO transform and features
    dino_transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dino_input = dino_transform(img).unsqueeze(0).cuda()
    dino_features = get_dino_features(dino_model, dino_input)
    dino_viz = create_pca_viz(dino_features, "DINO")
    
    # Create visualization
    if vjepa_model is not None:
        vjepa_features = get_vjepa_features(vjepa_model, vjepa_processor, img)
        vjepa_viz = create_pca_viz(vjepa_features, "V-JEPA")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(dino_viz)
        axes[1].set_title('DINO-v2')
        axes[1].axis('off')
        
        axes[2].imshow(vjepa_viz)
        axes[2].set_title('V-JEPA-2')
        axes[2].axis('off')
        
        filename = f"pca_comparison_{img_name}.jpg"
    else:
        # DINO only
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(dino_viz)
        axes[1].set_title('DINO-v2')
        axes[1].axis('off')
        
        filename = f"pca_dino_{img_name}.jpg"
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def main():
    # Find images
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) 
                   if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if not image_paths:
        print(f"No images found in {IMG_DIR}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load models
    dino_model, vjepa_model, vjepa_processor = load_models()
    
    # Process images
    for img_path in image_paths:
        try:
            process_image(img_path, dino_model, vjepa_model, vjepa_processor)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()


