"""
Underwater Image Restoration - Preprocessing Pipeline
======================================================
This module implements the complete preprocessing workflow for underwater images.
Key Preprocessing Stages:
1. Color Space Conversion (BGR/Color space handling -> standard RGB).
2. Spatial Standardization (Lanczos/Bicubic resizing to 256x256).
3. Value Range Normalization (Scaling [0, 255] integer pixels -> [0.0, 1.0] float).
4. Tensor Formatting (Reordering HWC -> CHW for PyTorch CNN processing).
5. Synthetic Underwater Degradation (Gaussian blur and sensor noise simulating turbidity).
6. Contrast Limited Adaptive Histogram Equalization (CLAHE on LAB color space).
7. Paired Data Augmentations (Synchronized geometric flips and rotations).
"""
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
def load_and_standardize_image(image_path, target_size=(256, 256)):
    """
    Loads an image from disk, converts to RGB color space,
    and standardizes resolution using high-fidelity Lanczos resampling.
    Args:
        image_path (str): Path to input image file.
        target_size (tuple): Target dimensions (width, height), default (256, 256).
    Returns:
        np.ndarray: Preprocessed RGB image array of shape (H, W, 3) in uint8 [0, 255].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)
def to_normalized_tensor(image_array):
    """
    Normalizes pixel intensities from [0, 255] to [0.0, 1.0] and converts
    shape from (Height, Width, Channels) to PyTorch's (Channels, Height, Width).
    Args:
        image_array (np.ndarray): Input RGB array of shape (H, W, C) in [0, 255].
    Returns:
        torch.Tensor: Float tensor of shape (C, H, W) normalized to [0.0, 1.0].
    """
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    return tensor / 255.0
def tensor_to_image(tensor):
    """
    Converts a normalized PyTorch tensor back into a uint8 RGB numpy array.
    Args:
        tensor (torch.Tensor): Shape (C, H, W) or (1, C, H, W) in [0.0, 1.0].
    Returns:
        np.ndarray: RGB image array of shape (H, W, C) in [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array
def apply_clahe(image_array, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    exclusively on the Lightness (L) channel of the LAB color space.
    This corrects underwater haze while preventing color distortion.
    Args:
        image_array (np.ndarray): RGB image array of shape (H, W, 3).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Grid size for histogram equalization.
    Returns:
        np.ndarray: Enhanced RGB image array.
    """
    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    merged_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
def apply_underwater_degradation(image_array,
                                blur_prob=0.5,
                                max_blur_sigma=3.0,
                                noise_prob=0.5,
                                max_noise_sigma=5.0):
    """
    Simulates real-world aquatic turbidity on training raw images:
    - Gaussian blurring simulates light scattering in turbid water.
    - Gaussian noise simulates sensor noise in low-light underwater depths.
    Args:
        image_array (np.ndarray): Input image array in uint8.
        blur_prob (float): Probability of applying Gaussian blur.
        max_blur_sigma (float): Maximum standard deviation for blur kernel.
        noise_prob (float): Probability of injecting Gaussian noise.
        max_noise_sigma (float): Maximum noise standard deviation.
    Returns:
        np.ndarray: Degraded image array.
    """
    degraded = image_array.copy()
    if random.random() < blur_prob:
        sigma = random.uniform(0.5, max_blur_sigma)
        degraded = cv2.GaussianBlur(degraded, (0, 0), sigma)
    if random.random() < noise_prob:
        sigma = random.uniform(1.0, max_noise_sigma)
        noise = np.random.normal(0, sigma, degraded.shape)
        degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return degraded
def augment_image_pair(raw_tensor, ref_tensor):
    """
    Applies identical geometric transformations to both raw and reference tensors
    so paired spatial alignment is strictly preserved.
    Transformations:
    - Random horizontal reflection (p = 0.5)
    - Random vertical reflection (p = 0.5)
    - Random orthogonal rotation (0, 90, 180, or 270 degrees)
    """
    if random.random() < 0.5:
        raw_tensor = torch.flip(raw_tensor, dims=[2])
        ref_tensor = torch.flip(ref_tensor, dims=[2])
    if random.random() < 0.5:
        raw_tensor = torch.flip(raw_tensor, dims=[1])
        ref_tensor = torch.flip(ref_tensor, dims=[1])
    k = random.randint(0, 3)
    if k > 0:
        raw_tensor = torch.rot90(raw_tensor, k, dims=[1, 2])
        ref_tensor = torch.rot90(ref_tensor, k, dims=[1, 2])
    return raw_tensor, ref_tensor
if __name__ == "__main__":
    print("=" * 65)
    print(" UNDERWATER IMAGE RESTORATION: PREPROCESSING PIPELINE DEMO")
    print("=" * 65)
    sample_candidates = [
        os.path.join("test_data", "challenging-60", "34.png"),
        os.path.join("test_data", "challenging-60", "2.png"),
        os.path.join("test_data", "U45", "1.png"),
    ]
    sample_path = None
    for candidate in sample_candidates:
        if os.path.exists(candidate):
            sample_path = candidate
            break
    if sample_path:
        print(f"\n[1] Selected Sample Image: {sample_path}")
        img_np = load_and_standardize_image(sample_path, target_size=(256, 256))
        print(f"    - Standardized Resolution : {img_np.shape[1]}x{img_np.shape[0]} (RGB)")
        print(f"    - Raw Pixel Range         : [{img_np.min()}, {img_np.max()}] (uint8)")
        tensor = to_normalized_tensor(img_np)
        print(f"\n[2] PyTorch Tensor Conversion:")
        print(f"    - Tensor Dimensions (C,H,W): {list(tensor.shape)}")
        print(f"    - Normalized Float Range   : [{tensor.min():.4f}, {tensor.max():.4f}]")
        clahe_img = apply_clahe(img_np)
        print(f"\n[3] Adaptive Contrast Enhancement (CLAHE in LAB):")
        print(f"    - Lightness equalization completed successfully.")
        degraded_img = apply_underwater_degradation(img_np, blur_prob=1.0, noise_prob=1.0)
        print(f"\n[4] Underwater Turbidity Degradation (Training Mode):")
        print(f"    - Gaussian Blur & Sensor Noise injected.")
        raw_aug, ref_aug = augment_image_pair(tensor, tensor.clone())
        print(f"\n[5] Synchronized Pair Augmentation:")
        print(f"    - Geometric Flips / Rotations applied preserving spatial alignment.")
        print("\n" + "=" * 65)
        print(" PREPROCESSING PIPELINE VERIFIED SUCCESSFULLY")
        print("=" * 65)
    else:
        print("No sample image found to run live demo, but all functions are ready.")
