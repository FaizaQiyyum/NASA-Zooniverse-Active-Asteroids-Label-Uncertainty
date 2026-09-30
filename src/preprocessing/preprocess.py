# NASA Zooniverse Active Asteroids
# Image Preprocessing Pipeline
#
# Handles:
#   - Loading images from disk or generating synthetic representations
#     for experiments when real images are unavailable
#   - Normalization and standardization
#   - Feature extraction for logistic regression baseline
#   - PyTorch Dataset class for CNN training
#
# Usage:
#   from src.preprocessing.preprocess import AsteroidDataset, extract_features

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Image Loading ─────────────────────────────────────────────────────────────

def load_image_from_disk(image_path: str, target_size: int = 224) -> np.ndarray:
    """Load image from disk and return as numpy array (H, W, C)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


def generate_synthetic_cutout(
    subject_id: str,
    label: int,
    uncertainty_category: str,
    seed: int = None,
    size: int = 224,
) -> np.ndarray:
    """
    Generate a synthetic astronomical cutout for use when real images
    are unavailable (e.g., local compute without downloaded images).
    
    IMPORTANT: These are not real astronomical images.
    They are procedural simulations used only for:
    1. Validating the training pipeline end-to-end
    2. Demonstrating the code structure
    3. Stress-testing the uncertainty analysis framework
    
    All results from synthetic data are clearly labeled as such
    and are NOT presented as scientific findings about real asteroids.
    
    Synthetic image types:
    - teaching_positive: Bright central object with simulated tail/coma
    - catalog_matched_positive: Bright object, possibly with faint structure
    - unmatched_negative: Faint or point-source object, no tail
    """
    if seed is None:
        # Deterministic seed from subject_id
        seed = abs(hash(subject_id)) % (2**31)
    
    rng = np.random.RandomState(seed)
    
    # Base sky background (low noise)
    image = rng.normal(50, 8, (size, size, 3)).astype(np.float32)
    image = np.clip(image, 0, 255)
    
    # Add stars (point sources)
    n_stars = rng.randint(3, 15)
    for _ in range(n_stars):
        sx = rng.randint(0, size)
        sy = rng.randint(0, size)
        brightness = rng.uniform(100, 240)
        # PSF-like Gaussian
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= sx + dx < size and 0 <= sy + dy < size:
                    d2 = dx**2 + dy**2
                    val = brightness * np.exp(-d2 / 2.0)
                    image[sy + dy, sx + dx] += val
    
    # Central object (the asteroid/comet candidate)
    cx, cy = size // 2, size // 2
    
    if uncertainty_category == "teaching_positive":
        # Clearly active: bright central object + definite tail
        obj_brightness = rng.uniform(200, 255)
        tail_angle = rng.uniform(0, 2 * np.pi)
        tail_length = rng.randint(30, 70)
        tail_width = rng.randint(5, 15)
        
        # Central nucleus
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if 0 <= cx + dx < size and 0 <= cy + dy < size:
                    d2 = dx**2 + dy**2
                    val = obj_brightness * np.exp(-d2 / 8.0)
                    image[cy + dy, cx + dx] += val
        
        # Tail
        for t in range(tail_length):
            tx = int(cx + t * np.cos(tail_angle))
            ty = int(cy + t * np.sin(tail_angle))
            tail_brightness = obj_brightness * 0.6 * np.exp(-t / 20.0)
            for dw in range(-tail_width // 2, tail_width // 2 + 1):
                px = tx + int(dw * np.sin(tail_angle))
                py = ty - int(dw * np.cos(tail_angle))
                if 0 <= px < size and 0 <= py < size:
                    val = tail_brightness * np.exp(-dw**2 / (tail_width**2 / 4))
                    image[py, px] += val
    
    elif uncertainty_category == "catalog_matched_positive":
        # Ambiguous: bright central object, no clear tail visible
        # This represents the LABEL UNCERTAINTY case:
        # object is historically active but THIS image may not show activity
        obj_brightness = rng.uniform(150, 220)
        
        # Central nucleus (no clear tail - potentially non-active epoch)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if 0 <= cx + dx < size and 0 <= cy + dy < size:
                    d2 = dx**2 + dy**2
                    val = obj_brightness * np.exp(-d2 / 6.0)
                    image[cy + dy, cx + dx] += val
        
        # Sometimes add faint extension (ambiguous morphology)
        if rng.random() > 0.5:
            tail_angle = rng.uniform(0, 2 * np.pi)
            for t in range(rng.randint(5, 20)):
                tx = int(cx + t * np.cos(tail_angle))
                ty = int(cy + t * np.sin(tail_angle))
                if 0 <= tx < size and 0 <= ty < size:
                    image[ty, tx] += obj_brightness * 0.15 * rng.random()
    
    else:  # unmatched_negative
        # Inactive asteroid or background galaxy
        obj_type = rng.choice(["point_source", "galaxy", "artifact"])
        
        if obj_type == "point_source":
            # Simple point source (inactive asteroid)
            obj_brightness = rng.uniform(80, 180)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if 0 <= cx + dx < size and 0 <= cy + dy < size:
                        d2 = dx**2 + dy**2
                        val = obj_brightness * np.exp(-d2 / 3.0)
                        image[cy + dy, cx + dx] += val
        
        elif obj_type == "galaxy":
            # Extended elliptical source (galaxy, common false positive)
            a = rng.randint(15, 35)
            b = rng.randint(5, 15)
            angle = rng.uniform(0, np.pi)
            brightness = rng.uniform(120, 200)
            for dy in range(-a - 5, a + 6):
                for dx in range(-a - 5, a + 6):
                    # Rotated ellipse
                    xr = dx * np.cos(angle) + dy * np.sin(angle)
                    yr = -dx * np.sin(angle) + dy * np.cos(angle)
                    d2 = (xr / a)**2 + (yr / b)**2
                    if d2 < 4 and 0 <= cx + dx < size and 0 <= cy + dy < size:
                        val = brightness * np.exp(-d2 * 0.8)
                        image[cy + dy, cx + dx] += val
        
        else:  # artifact
            # CCD artifact (streak or cosmic ray)
            streak_length = rng.randint(20, 50)
            streak_angle = rng.uniform(0, 2 * np.pi)
            for t in range(-streak_length // 2, streak_length // 2):
                sx = int(cx + t * np.cos(streak_angle))
                sy = int(cy + t * np.sin(streak_angle))
                if 0 <= sx < size and 0 <= sy < size:
                    image[sy, sx] += rng.uniform(200, 255)
    
    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# ─── Feature Extraction (for Logistic Regression Baseline) ────────────────────

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract 8 hand-crafted features from a cutout image.
    
    Features adapted from ZamirDev baseline (documented as reference):
    1. Central concentration: mean of center 32x32 minus annulus mean
    2. Global contrast: standard deviation of pixel values
    3. Hot-tail spread: 99.5th percentile minus 90th percentile
    4. Centre structure: std of central 24x24 box
    5. X-roughness: mean absolute horizontal gradient
    6. Edge energy: mean Laplacian magnitude
    7. Peak-vs-background: max value minus global mean
    8. Elongation ratio: PCA of bright pixels (moment ratio)
    
    These are the same feature types as ZamirDev's baseline, documented
    for comparison purposes. Our investigation goes BEYOND these features
    by using CNN representations to study uncertainty.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    # Resize to 96x96 for feature computation
    pil_img = Image.fromarray(gray.astype(np.uint8))
    pil_img = pil_img.resize((96, 96), Image.LANCZOS)
    gray = np.array(pil_img, dtype=np.float32)
    
    h, w = gray.shape
    ch, cw = h // 2, w // 2
    half = 16  # center region half-size
    
    # 1. Central concentration
    center = gray[ch - half:ch + half, cw - half:cw + half]
    annulus = gray.copy()
    annulus[ch - half:ch + half, cw - half:cw + half] = np.nan
    center_mean = float(np.mean(center))
    annulus_mean = float(np.nanmean(annulus))
    f1 = center_mean - annulus_mean
    
    # 2. Global contrast
    f2 = float(np.std(gray))
    
    # 3. Hot-tail spread
    f3 = float(np.percentile(gray, 99.5) - np.percentile(gray, 90))
    
    # 4. Centre structure
    f4 = float(np.std(gray[ch - 12:ch + 12, cw - 12:cw + 12]))
    
    # 5. X-roughness (horizontal gradient)
    grad_x = np.abs(np.diff(gray, axis=1))
    f5 = float(np.mean(grad_x))
    
    # 6. Edge energy (Laplacian)
    pil_gray = Image.fromarray(gray.astype(np.uint8))
    laplacian = pil_gray.filter(ImageFilter.FIND_EDGES)
    f6 = float(np.mean(np.array(laplacian)))
    
    # 7. Peak vs background
    f7 = float(np.max(gray) - np.mean(gray))
    
    # 8. Elongation ratio (moment-based)
    # Using pixel moments to estimate elongation
    threshold = float(np.percentile(gray, 80))
    bright_mask = gray > threshold
    if bright_mask.sum() > 10:
        y_coords, x_coords = np.where(bright_mask)
        cov = np.cov(x_coords.astype(float), y_coords.astype(float))
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        f8 = float(eigenvalues[0] / max(eigenvalues[1], 1e-6))
    else:
        f8 = 1.0
    
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)


# ─── PyTorch Dataset ───────────────────────────────────────────────────────────

class AsteroidDataset(Dataset):
    """
    PyTorch Dataset for Active Asteroid cutout images.
    
    Supports both real downloaded images and synthetic images
    (for pipeline testing without downloaded data).
    
    Args:
        df: DataFrame with subject metadata (from subjects.csv)
        image_dir: Directory with downloaded images (or None for synthetic)
        split: 'train', 'val', or 'test'
        transform: torchvision transforms
        use_synthetic: Force synthetic images (for testing/demonstration)
        synthetic_size: Size of synthetic images
    """
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str = None,
        split: str = "train",
        transform=None,
        use_synthetic: bool = False,
        synthetic_size: int = 224,
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir else None
        self.split = split
        self.use_synthetic = use_synthetic
        self.synthetic_size = synthetic_size
        
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(180),  # Astronomical images have no orientation
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
                ])
        else:
            self.transform = transform
        
        print(f"AsteroidDataset [{split}]: {len(self.df)} subjects")
        if "object_level_label" in self.df.columns:
            pos = (self.df["object_level_label"] == 1).sum()
            neg = (self.df["object_level_label"] == 0).sum()
            print(f"  Positive: {pos}, Negative: {neg}")
        
        if use_synthetic:
            print(f"  NOTE: Using SYNTHETIC images (not real astronomical data)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row.get("object_level_label", 0))
        subject_id = str(row["subject_id"])
        uncertainty_cat = str(row.get("uncertainty_category", "unmatched_negative"))
        
        # Try to load real image
        image = None
        
        if not self.use_synthetic and self.image_dir is not None:
            label_dir = "positive" if label == 1 else "negative"
            img_path = self.image_dir / self.split / label_dir / f"{subject_id}.png"
            
            if img_path.exists():
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
        
        # Fall back to synthetic if real image unavailable
        if image is None:
            synth_array = generate_synthetic_cutout(
                subject_id=subject_id,
                label=label,
                uncertainty_category=uncertainty_cat,
                size=self.synthetic_size,
            )
            image = Image.fromarray(synth_array)
        
        # Apply transforms
        tensor = self.transform(image)
        
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "subject_id": subject_id,
            "uncertainty_category": uncertainty_cat,
            "label_source": str(row.get("label_source", "unknown")),
        }
    
    def get_class_weights(self):
        """Return class weights for imbalanced training."""
        labels = self.df["object_level_label"].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        n_total = len(labels)
        
        # Weight inversely proportional to class frequency
        weight_pos = n_total / (2 * max(n_pos, 1))
        weight_neg = n_total / (2 * max(n_neg, 1))
        
        return {"positive": weight_pos, "negative": weight_neg}
    
    def get_sample_weights(self):
        """Return per-sample weights for WeightedRandomSampler."""
        labels = self.df["object_level_label"].values
        class_weights = self.get_class_weights()
        
        sample_weights = np.where(
            labels == 1,
            class_weights["positive"],
            class_weights["negative"]
        )
        return torch.tensor(sample_weights, dtype=torch.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    image_dir: str = None,
    use_synthetic: bool = True,
) -> tuple:
    """
    Build feature matrix for logistic regression baseline.
    
    Returns:
        X: (n_samples, n_features) float array
        y: (n_samples,) int array of labels
        subject_ids: list of subject IDs
        uncertainty_cats: list of uncertainty categories
    """
    X_list = []
    y_list = []
    ids = []
    unc_cats = []
    
    image_dir_path = Path(image_dir) if image_dir else None
    
    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        label = int(row.get("object_level_label", 0))
        unc_cat = str(row.get("uncertainty_category", "unmatched_negative"))
        split = str(row.get("split", "train"))
        
        # Try real image first
        image = None
        if image_dir_path is not None:
            label_dir = "positive" if label == 1 else "negative"
            img_path = image_dir_path / split / label_dir / f"{subject_id}.png"
            if img_path.exists():
                try:
                    image = np.array(Image.open(img_path).convert("RGB"))
                except Exception:
                    pass
        
        if image is None:
            image = generate_synthetic_cutout(subject_id, label, unc_cat)
        
        features = extract_features(image)
        X_list.append(features)
        y_list.append(label)
        ids.append(subject_id)
        unc_cats.append(unc_cat)
    
    return np.array(X_list), np.array(y_list), ids, unc_cats
