#!/usr/bin/env python3
"""
V-JEPA-2 block-masking PCA visualiser
------------------------------------
Creates side-by-side panels:
1. Original image
2. Block-masked (tubelet) image
3. PCA of original
4. PCA of masked

Mask = one contiguous 3-D block spanning a random temporal depth and spatial
rectangle (following the rough strategy used in the official V-JEPA code).

Usage
-----
python vjepa2_masking_pca.py --img_dir images --spatial_scale 0.2 0.8 --aspect 0.3 3.0 --npred 1
"""
import os, math, random, argparse
import numpy as np
from PIL import Image
import torch, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

VJEPA_SIZE = 256               # resize to square
PATCH_SIZE = 16                # spatial patch size (16×16)
TUBELET   = 2                  # frames per temporal token
FRAMES    = 64                 # total frames in fake clip
TOKENS_T  = FRAMES // TUBELET  # 32 temporal tokens
TOKENS_H  = TOKENS_W = VJEPA_SIZE // PATCH_SIZE  # 16×16 spatial tokens

######################################################################
# Block-mask generator (simplified subset of Meta’s _MaskGenerator)
######################################################################
class BlockMaskGenerator:
    def __init__(self, spatial_scale=(0.2,0.8), temporal_scale=(1.0,1.0),
                 aspect=(0.3,3.0), npred=1, center=False):
        self.min_s, self.max_s = spatial_scale
        self.min_t, self.max_t = temporal_scale
        self.min_ar, self.max_ar = aspect
        self.npred = npred
        self.center = center

    def _sample_block_size(self):
        # temporal
        t_scale = random.uniform(self.min_t, self.max_t)
        t = max(1, int(TOKENS_T * t_scale))
        # spatial area
        s_scale = random.uniform(self.min_s, self.max_s)
        target_tokens = int(TOKENS_H * TOKENS_W * s_scale)
        # aspect ratio
        ar = random.uniform(self.min_ar, self.max_ar)
        h = int(round(math.sqrt(target_tokens * ar)))
        w = int(round(math.sqrt(target_tokens / ar)))
        h = min(max(1, h), TOKENS_H)
        w = min(max(1, w), TOKENS_W)
        return t, h, w

    def _single_mask(self):
        t, h, w = self._sample_block_size()
        if self.center:
            top  = (TOKENS_H - h) // 2
            left = (TOKENS_W - w) // 2
            start = (TOKENS_T - t)//2
        else:
            top  = random.randint(0, TOKENS_H - h)
            left = random.randint(0, TOKENS_W - w)
            start = random.randint(0, TOKENS_T - t)
        mask = np.ones((TOKENS_T, TOKENS_H, TOKENS_W), dtype=np.bool_)
        mask[start:start+t, top:top+h, left:left+w] = False  # False = predict / masked
        return mask

    def __call__(self):
        mask = np.ones((TOKENS_T, TOKENS_H, TOKENS_W), dtype=np.bool_)
        for _ in range(self.npred):
            mask &= self._single_mask()
        return mask  # True = keep / context, False = mask / predict

######################################################################
# V-JEPA model loader (HF)
######################################################################

def load_vjepa():
    from transformers import AutoModel, AutoVideoProcessor
    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").cuda().eval()
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    return model, processor

######################################################################
# PCA helper
######################################################################

def pca_image(feats):
    pca = PCA(n_components=3)
    X = pca.fit_transform(feats)
    X = X[:, :3]
    for i in range(3):
        ch = X[:, i]
        X[:, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    n = X.shape[0]
    g = int(math.sqrt(n))
    if g * g != n:
        pad = g * g - n
        X = np.vstack([X, np.zeros((pad,3))])
    return X.reshape(g, g, 3)

######################################################################
# Feature extraction (temporal avg to 256 spatial tokens)
######################################################################

def vjepa_features(model, processor, img_arr):
    frames = np.stack([img_arr] * FRAMES, axis=0)
    inputs = processor(frames, return_tensors="pt")
    pixel_videos = inputs['pixel_values_videos'].cuda()
    with torch.no_grad():
        feats = model(pixel_videos).last_hidden_state[0].cpu().numpy()
    if feats.shape[0] == 8192:  # 32 × 256
        feats = feats.reshape(32, 256, -1).mean(0)
    return feats  # (256, C)

######################################################################
# Mask application in pixel space (grey 127)
######################################################################

def apply_pixel_mask(img_arr, mask_bool):
    # mask_bool shape (T,H,W) tokens; expand to pixel grid
    mask_spatial = mask_bool[0]  # we mask all frames identically for display
    out = img_arr.copy()
    for hi in range(TOKENS_H):
        for wi in range(TOKENS_W):
            if not mask_spatial[hi,wi]:
                y0, y1 = hi*PATCH_SIZE, (hi+1)*PATCH_SIZE
                x0, x1 = wi*PATCH_SIZE, (wi+1)*PATCH_SIZE
                out[y0:y1, x0:x1] = 127
    return out

######################################################################
# Main runner
######################################################################

def run_on_image(path, gen, model, processor, out_dir):
    img = Image.open(path).convert('RGB').resize((VJEPA_SIZE, VJEPA_SIZE))
    arr = np.array(img)
    mask_bool = gen()
    masked_arr = apply_pixel_mask(arr, ~mask_bool)  # invert: False tokens are masked

    feats_orig = vjepa_features(model, processor, arr)
    feats_mask = vjepa_features(model, processor, masked_arr)

    viz_orig = pca_image(feats_orig)
    viz_mask = pca_image(feats_mask)

    # plot
    fig, ax = plt.subplots(1,4, figsize=(16,4))
    ax[0].imshow(arr);          ax[0].set_title('Original');      ax[0].axis('off')
    ax[1].imshow(masked_arr);   ax[1].set_title('Masked');        ax[1].axis('off')
    ax[2].imshow(viz_orig);     ax[2].set_title('PCA');           ax[2].axis('off')
    ax[3].imshow(viz_mask);     ax[3].set_title('PCA Masked');    ax[3].axis('off')
    fname = os.path.basename(path)
    out_path = os.path.join(out_dir, f"blockmask_{fname.replace('.', '_')}.jpg")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print('Saved', out_path)

######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', default='images', help='Folder with images')
    ap.add_argument('--npred', type=int, default=1)
    ap.add_argument('--spatial_scale', nargs=2, type=float, default=[0.2,0.8])
    ap.add_argument('--temporal_scale', nargs=2, type=float, default=[1.0,1.0])
    ap.add_argument('--aspect', nargs=2, type=float, default=[0.3,3.0])
    ap.add_argument('--out_dir', default='mask_results')
    ap.add_argument('--center', action='store_true', help='Mask centre block instead of random placement')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gen = BlockMaskGenerator(tuple(args.spatial_scale), tuple(args.temporal_scale),
                              tuple(args.aspect), args.npred, center=args.center)
    model, processor = load_vjepa()
    paths = [os.path.join(args.img_dir,f) for f in os.listdir(args.img_dir)
             if f.lower().endswith(('jpg','jpeg','png'))]
    for p in paths:
        run_on_image(p, gen, model, processor, args.out_dir)

if __name__ == '__main__':
    main()
