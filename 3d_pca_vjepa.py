#!/usr/bin/env python3
# V-JEPA 3D PCA Visualization Script

import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import plotly.express as px  # for interactive 3-D plots
from sklearn.decomposition import PCA
import argparse, random

IMG_DIR = None  # will be set from args
VJEPA_SIZE = 256


def load_vjepa():
    print("Loading V-JEPA-2...")
    from transformers import AutoModel, AutoVideoProcessor
    vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    vjepa_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    vjepa_model = vjepa_model.cuda().eval()
    return vjepa_model, vjepa_processor


def parse_args():
    parser = argparse.ArgumentParser(description="V-JEPA 3D PCA visualization with optional masking")
    parser.add_argument("--img_dir", type=str, default="images", help="Directory with images")
    parser.add_argument("--mask_ratio", type=float, default=0.0, help="Ratio of patches to mask (0-1). 0 disables masking")
    parser.add_argument("--dense3d", action="store_true", help="Keep all temporal patches (higher point density)")
    return parser.parse_args()


def get_vjepa_features(model, processor, image_pil, mask_ratio=0.0, avg_temporal=True):
    img_resized = image_pil.resize((VJEPA_SIZE, VJEPA_SIZE))
    img_array = np.array(img_resized)
    img_array = apply_patch_mask(img_array, mask_ratio)
    frames = 64
    video_np = np.stack([img_array] * frames, axis=0)  # (64, H, W, 3)
    inputs = processor(video_np, return_tensors="pt")
    pixel_values = inputs['pixel_values_videos'].cuda()
    print(f"V-JEPA input shape: {pixel_values.shape}")
    with torch.no_grad():
        outputs = model(pixel_values)
        features = outputs.last_hidden_state
    features_np = features[0].cpu().numpy()
    print(f"V-JEPA features shape: {features_np.shape}")
    if features_np.shape[0] == 8192:
        features_3d = features_np.reshape(32, 256, features_np.shape[1])
        if avg_temporal:
            features_spatial = np.mean(features_3d, axis=0)
            print(f"Averaged temporal patches: {features_3d.shape} -> {features_spatial.shape}")
            return features_spatial
        else:
            features_flat = features_3d.reshape(-1, features_3d.shape[-1])
            print(f"Flattened temporal patches: {features_3d.shape} -> {features_flat.shape}")
            return features_flat
    else:
        print(f"Using features as-is: {features_np.shape}")
        return features_np





# ---------------------- NEW: 3-D PCA visualization -----------------------
def plot_pca_3d(features, title, html_filename=None):
    """Generate a 3-D scatter plot of the first three PCA components using Plotly.
    
    Now includes spatial positioning and meaningful color coding to show
    the relationship between feature space and image space.
    """
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(features)
    
    # Determine spatial grid - assume square grid from features
    n_patches = features.shape[0]
    if n_patches == 256:  # 16x16 grid
        grid_size = 16
    elif n_patches == 8192:  # 32x256 temporal patches
        grid_size = int(np.sqrt(256))  # Use spatial dimension
    else:
        grid_size = int(np.sqrt(n_patches))
    
    # Create spatial coordinates for each patch
    spatial_coords = []
    colors_by_position = []
    
    if n_patches == 8192:  # Temporal patches - reshape to get spatial info
        for i in range(n_patches):
            temporal_idx = i // 256
            spatial_idx = i % 256
            row = spatial_idx // grid_size
            col = spatial_idx % grid_size
            spatial_coords.append((row, col, temporal_idx))
            # Color by spatial position (distance from center)
            center_row, center_col = grid_size // 2, grid_size // 2
            dist_from_center = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            colors_by_position.append(dist_from_center)
    else:  # Spatial patches only
        for i in range(n_patches):
            row = i // grid_size
            col = i % grid_size
            spatial_coords.append((row, col, 0))
            # Color by spatial position
            center_row, center_col = grid_size // 2, grid_size // 2
            dist_from_center = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            colors_by_position.append(dist_from_center)
    
    spatial_coords = np.array(spatial_coords)
    colors_by_position = np.array(colors_by_position)
    
    # Create hover text with spatial information
    hover_text = []
    for i, (row, col, temporal) in enumerate(spatial_coords):
        if n_patches == 8192:
            hover_text.append(f"Patch ({row}, {col}), Frame {temporal}<br>PC1: {pcs[i,0]:.3f}<br>PC2: {pcs[i,1]:.3f}<br>PC3: {pcs[i,2]:.3f}")
        else:
            hover_text.append(f"Patch ({row}, {col})<br>PC1: {pcs[i,0]:.3f}<br>PC2: {pcs[i,1]:.3f}<br>PC3: {pcs[i,2]:.3f}")
    
    # Create the 3D plot with spatial meaning
    fig = px.scatter_3d(
        x=pcs[:, 0], y=pcs[:, 1], z=pcs[:, 2],
        color=colors_by_position,
        color_continuous_scale="Viridis",
        size_max=10,
        opacity=0.8,
        title=f"{title} – 3-D PCA (Colored by Distance from Center)",
        labels={
            'x': 'PC1 (Feature Space)',
            'y': 'PC2 (Feature Space)', 
            'z': 'PC3 (Feature Space)',
            'color': 'Distance from Center'
        },
        hover_name=[f"Patch {i}" for i in range(len(pcs))]
    )
    
    # Update hover template with spatial information
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "PC1: %{x:.3f}<br>" +
                      "PC2: %{y:.3f}<br>" +
                      "PC3: %{z:.3f}<br>" +
                      "Distance from Center: %{marker.color:.2f}<br>" +
                      "<extra></extra>",
        hovertext=hover_text
    )
    
    # Add explained variance to the plot
    explained_var = pca.explained_variance_ratio_
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Explained Variance:<br>PC1: {explained_var[0]:.3f}<br>PC2: {explained_var[1]:.3f}<br>PC3: {explained_var[2]:.3f}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    if html_filename:
        fig.write_html(html_filename)
        print(f"Saved 3-D plot: {html_filename}")
    else:
        fig.show()


def plot_spatial_feature_mapping(features, title, html_filename=None):
    """Create a visualization showing how spatial domains map to feature clusters."""
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(features)
    
    n_patches = features.shape[0]
    if n_patches == 256:
        grid_size = 16
    elif n_patches == 8192:
        grid_size = int(np.sqrt(256))
    else:
        grid_size = int(np.sqrt(n_patches))
    
    # Create spatial grid visualization
    spatial_grid = np.zeros((grid_size, grid_size, 3))
    
    if n_patches == 8192:  # Temporal patches
        # Average over temporal dimension for spatial visualization
        pcs_spatial = pcs.reshape(32, 256, 3).mean(axis=0)
    else:
        pcs_spatial = pcs.copy()
    
    # Normalize PCA components to [0,1] for RGB visualization
    for i in range(3):
        pc_norm = (pcs_spatial[:, i] - pcs_spatial[:, i].min()) / (pcs_spatial[:, i].max() - pcs_spatial[:, i].min() + 1e-8)
        spatial_grid[:, :, i] = pc_norm.reshape(grid_size, grid_size)
    
    # Create subplot with spatial and feature views
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Spatial Domain (RGB = PC1,PC2,PC3)", "Feature Space Clusters"],
        specs=[[{"type": "xy"}, {"type": "scene"}]]
    )
    
    # Add spatial grid as image
    fig.add_trace(
        px.imshow(spatial_grid, title="Spatial Domain").data[0],
        row=1, col=1
    )
    
    # Add 3D scatter
    scatter_3d = px.scatter_3d(
        x=pcs[:, 0], y=pcs[:, 1], z=pcs[:, 2],
        color=np.sqrt(np.sum(pcs**2, axis=1)),  # Distance from origin
        color_continuous_scale="Plasma"
    ).data[0]
    
    fig.add_trace(scatter_3d, row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"{title} – Spatial-Feature Mapping",
        height=600,
        width=1200
    )
    
    if html_filename:
        spatial_html = html_filename.replace('.html', '_spatial_mapping.html')
        fig.write_html(spatial_html)
        print(f"Saved spatial mapping: {spatial_html}")
    else:
        fig.show()


def plot_feature_manifold_topology(features, title, html_filename=None):
    """Create a topology-preserving visualization of feature manifold structure."""
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(features)
    
    # Use t-SNE for better manifold preservation
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)
    
    # Cluster features to identify domains
    n_clusters = 8  # Adjust based on image complexity
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    n_patches = features.shape[0]
    if n_patches == 256:
        grid_size = 16
    elif n_patches == 8192:
        grid_size = int(np.sqrt(256))
    else:
        grid_size = int(np.sqrt(n_patches))
    
    # Create spatial coordinates
    spatial_coords = []
    for i in range(min(n_patches, 256)):  # Limit to spatial patches
        row = i // grid_size
        col = i % grid_size
        spatial_coords.append((row, col))
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Feature Manifold (t-SNE)",
            "Spatial Clusters",
            "3D PCA Feature Space", 
            "Cluster Centroids"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scene"}, {"type": "xy"}]
        ]
    )
    
    # t-SNE plot
    fig.add_trace(
        px.scatter(
            x=tsne_features[:, 0], y=tsne_features[:, 1],
            color=cluster_labels[:len(tsne_features)],
            color_continuous_scale="rainbow"
        ).data[0],
        row=1, col=1
    )
    
    # Spatial cluster visualization
    cluster_grid = np.zeros((grid_size, grid_size))
    for i, (row, col) in enumerate(spatial_coords):
        if i < len(cluster_labels):
            cluster_grid[row, col] = cluster_labels[i]
    
    fig.add_trace(
        px.imshow(cluster_grid, color_continuous_scale="rainbow").data[0],
        row=1, col=2
    )
    
    # 3D PCA scatter
    fig.add_trace(
        px.scatter_3d(
            x=pcs[:, 0], y=pcs[:, 1], z=pcs[:, 2],
            color=cluster_labels,
            color_continuous_scale="rainbow"
        ).data[0],
        row=2, col=1
    )
    
    # Cluster centroids in PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    fig.add_trace(
        px.scatter(
            x=centroids_pca[:, 0], y=centroids_pca[:, 1],
            size=[10]*len(centroids_pca),
            color=list(range(n_clusters)),
            color_continuous_scale="rainbow"
        ).data[0],
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"{title} – Feature Manifold Topology",
        height=800,
        width=1200,
        showlegend=False
    )
    
    if html_filename:
        topology_html = html_filename.replace('.html', '_topology.html')
        fig.write_html(topology_html)
        print(f"Saved topology plot: {topology_html}")
    else:
        fig.show()


def apply_patch_mask(img_array, mask_ratio=0.0, patch_size=16):
    if mask_ratio <= 0:
        return img_array
    h, w, c = img_array.shape
    gh, gw = h // patch_size, w // patch_size
    total_patches = gh * gw
    num_mask = int(mask_ratio * total_patches)
    indices = list(range(total_patches))
    random.shuffle(indices)
    mask_idx = indices[:num_mask]
    out = img_array.copy()
    for idx in mask_idx:
        r = idx // gw
        c_idx = idx % gw
        y0, y1 = r * patch_size, (r + 1) * patch_size
        x0, x1 = c_idx * patch_size, (c_idx + 1) * patch_size
        out[y0:y1, x0:x1, :] = 127  # gray mask
    return out


def process_image(img_path, vjepa_model, vjepa_processor, mask_ratio=0.0, dense3d=False):
    img = Image.open(img_path).convert('RGB')
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nProcessing: {img_name}")
    
    # Always generate 3D visualizations
    if dense3d:
        # Keep all temporal patches for higher density
        vjepa_features_all = get_vjepa_features(
            vjepa_model, vjepa_processor, img, mask_ratio=0.0, avg_temporal=False
        )
        feats_for_3d = vjepa_features_all
    else:
        # Use spatial average for cleaner visualization
        vjepa_features_spatial = get_vjepa_features(
            vjepa_model, vjepa_processor, img, mask_ratio=0.0, avg_temporal=True
        )
        feats_for_3d = vjepa_features_spatial

    # Generate 3D visualizations for original image
    html_name = f"pca_vjepa_{img_name}_3d.html"
    plot_pca_3d(feats_for_3d, f"V-JEPA {img_name}", html_filename=html_name)
    plot_spatial_feature_mapping(feats_for_3d, f"V-JEPA {img_name}", html_filename=html_name)
    plot_feature_manifold_topology(feats_for_3d, f"V-JEPA {img_name}", html_filename=html_name)
    
    # Process masked version if requested
    if mask_ratio > 0:
        masked_arr = apply_patch_mask(np.array(img.resize((VJEPA_SIZE, VJEPA_SIZE))), mask_ratio)
        masked_pil = Image.fromarray(masked_arr)
        
        if dense3d:
            vjepa_features_mask_all = get_vjepa_features(
                vjepa_model, vjepa_processor, masked_pil, mask_ratio=0.0, avg_temporal=False
            )
            feats_for_3d_mask = vjepa_features_mask_all
        else:
            vjepa_features_mask_spatial = get_vjepa_features(
                vjepa_model, vjepa_processor, masked_pil, mask_ratio=0.0, avg_temporal=True
            )
            feats_for_3d_mask = vjepa_features_mask_spatial

        # Generate 3D visualizations for masked image
        html_name_mask = f"pca_vjepa_mask{int(mask_ratio*100)}_{img_name}_3d.html"
        plot_pca_3d(feats_for_3d_mask, f"V-JEPA Masked {img_name}", html_filename=html_name_mask)
        plot_spatial_feature_mapping(feats_for_3d_mask, f"V-JEPA Masked {img_name}", html_filename=html_name_mask)
        plot_feature_manifold_topology(feats_for_3d_mask, f"V-JEPA Masked {img_name}", html_filename=html_name_mask)
    
    print(f"Generated 3D visualizations for: {img_name}")


def main():
    args = parse_args()
    global IMG_DIR
    IMG_DIR = args.img_dir
    image_paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(('jpg','jpeg','png'))]
    if not image_paths:
        print(f"No images found in {IMG_DIR}")
        return
    print(f"Found {len(image_paths)} images")
    vjepa_model, vjepa_processor = load_vjepa()
    for img_path in image_paths:
        try:
            process_image(img_path, vjepa_model, vjepa_processor,
                          mask_ratio=args.mask_ratio, dense3d=args.dense3d)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print("Done!")


if __name__ == "__main__":
    main()


