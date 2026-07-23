"""
BreakHis Dataset Loader
Block 1: Data Augmentation & Class Rebalancing
Block 2: Image Restructuring (patch tokenization handled by ViT internally)

BreakHis folder structure expected:
    BreaKHis_v1/
    └── histology_slides/
        └── breast/
            ├── benign/
            │   └── SOB/
            │       ├── adenosis/         -> label: 'A'(binary: benign)
            │       ├── fibroadenoma/     -> label: 'F'
            │       ├── phyllodes_tumor/  -> label: 'PT'
            │       └── tubular_adenoma/  -> label: 'TA'
            └── malignant/
                └── SOB/
                    ├── ductal_carcinoma/     -> label: 'DC' (binary: malignant)
                    ├── lobular_carcinoma/    -> label: 'LC'
                    ├── mucinous_carcinoma/   -> label: 'MC'
                    └── papillary_carcinoma/  -> label: 'PC'
"""

import os
import random
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ── Label mappings ────────────────────────────────────────────────────────────
SUBTYPE_TO_IDX = {
    "adenosis": 0,
    "fibroadenoma": 1,
    "phyllodes_tumor": 2,
    "tubular_adenoma": 3,
    "ductal_carcinoma": 4,
    "lobular_carcinoma": 5,
    "mucinous_carcinoma": 6,
    "papillary_carcinoma": 7,
}
IDX_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_IDX.items()}

BENIGN_SUBTYPES = {"adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"}
MALIGNANT_SUBTYPES = {"ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"}

BINARY_TO_IDX = {"benign": 0, "malignant": 1}
IDX_TO_BINARY = {0: "benign", 1: "malignant"}

VIT_IMAGE_SIZE = 384


def get_transforms(split: str, image_size: int = VIT_IMAGE_SIZE):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # 1. Convert to Tensor FIRST so that shape operations work
            transforms.ToTensor(),
            # 2. RandomErasing can now safely run on the Tensor
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            # 3. Apply normalization last
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class BreakHisDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        magnification: str = "40X",
        mode: str = "binary",
        split: str = "train",
        split_ratio: Tuple[float, float] = (0.70, 0.15),
        seed: int = 42,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.magnification = magnification.upper() if magnification != "all" else "all"
        
        self.mode = mode.lower().strip()
        self.split = split.lower().strip()
        
        self.transform = transform or get_transforms(self.split)
        self.seed = seed

        assert self.mode in ("binary", "multiclass"), f"Invalid mode '{mode}'. Must be 'binary' or 'multiclass'"
        assert self.split in ("train", "val", "test"), f"Invalid split '{split}'. Must be 'train', 'val', or 'test'"
        assert 0.0 <= split_ratio[0] + split_ratio[1] <= 1.0, "Split ratios must sum to <= 1.0"

        self.samples = []
        # Fixed reference to the internal function
        self._load_samples(split_ratio, self.seed)

    def _load_samples(self, split_ratio, seed):
        """Walk directory tree and collect (path, label) pairs."""
        all_samples = []

        for category in ["benign", "malignant"]:
            cat_dir = self.root_dir / category / "SOB"
            if not cat_dir.exists():
                print(f"[WARNING] Directory not found: {cat_dir}")
                continue

            for subtype_dir in sorted(cat_dir.iterdir()):
                if not subtype_dir.is_dir():
                    continue
                subtype = subtype_dir.name.lower()
                if subtype not in SUBTYPE_TO_IDX:
                    continue

                img_paths = []
                if self.magnification == "all":
                    img_paths = list(subtype_dir.rglob("*.png"))
                else:
                    mag_dirs = list(subtype_dir.glob(f"**/{self.magnification}"))
                    for mag_dir in mag_dirs:
                        img_paths.extend(mag_dir.glob("*.png"))

                if self.mode == "binary":
                    label = BINARY_TO_IDX["benign" if subtype in BENIGN_SUBTYPES else "malignant"]
                else:
                    label = SUBTYPE_TO_IDX[subtype]

                for p in img_paths:
                    all_samples.append((str(p), label))

        if not all_samples:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                "Check that the path points to the 'breast' folder inside BreaKHis_v1."
            )

        random.seed(seed)
        random.shuffle(all_samples)
        n = len(all_samples)
        t_end = int(n * split_ratio[0])
        v_end = t_end + int(n * split_ratio[1])

        if self.split == "train":
            self.samples = all_samples[:t_end]
        elif self.split == "val":
            self.samples = all_samples[t_end:v_end]
        else:
            self.samples = all_samples[v_end:]

        print(f"[{self.split.upper()}] {len(self.samples)} images | mode={self.mode} | mag={self.magnification}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self):
        labels = [s[1] for s in self.samples]
        return Counter(labels)

    @property
    def num_classes(self):
        return 2 if self.mode == "binary" else 8

    @property
    def class_names(self):
        if self.mode == "binary":
            return list(IDX_TO_BINARY.values())
        return list(IDX_TO_SUBTYPE.values())


def make_weighted_sampler(dataset: BreakHisDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts
    total = sum(counts.values())
    class_weights = {cls: total / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(root_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mag   = config.get("magnification", "40X")
    mode  = config.get("mode", "binary")
    bs    = config.get("batch_size", 32)
    nw    = config.get("num_workers", 4)
    isz   = config.get("image_size", VIT_IMAGE_SIZE)
    seed  = config.get("seed", 42)

    train_ds = BreakHisDataset(root_dir, mag, mode, "train", seed=seed, transform=get_transforms("train", isz))
    val_ds   = BreakHisDataset(root_dir, mag, mode, "val",   seed=seed, transform=get_transforms("val",   isz))
    test_ds  = BreakHisDataset(root_dir, mag, mode, "test",  seed=seed, transform=get_transforms("test",  isz))

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    DATA_DIR = DATA_DIR = "/home/user/Proj-Ploy/vit_breast_cancer/data/BreaKHis_v1/histology_slides/breast/"
    MODE = "binary"  # Fixed tuple configuration issue here
    MAG = "all"   # Use "all" to load all magnifiations, or specify "40X", "100X", "200X", "400X" for specific magnifications
    print("=" * 55)
    print("BreakHis Dataset")
    print("=" * 55)

    # Fixed list initialization to empty dict
    splits = {}
    target_splits = ["train", "val", "test"]
    
    # Ensure standard output directories exist
    os.makedirs("/home/user/Proj-Ploy/vit_breast_cancer/outputs", exist_ok=True)

    for split in target_splits:
        try:
            ds = BreakHisDataset(DATA_DIR, MAG, MODE, split)
            splits[split] = ds
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            print("\nSolve dataset loading issue by ensuring the path points to the 'breast' folder inside BreakHis_V1 and that the expected folder structure is intact.")
            sys.exit(1)

    print(f"\n{'Split':<8} {'Total': >7} Class Distribution")
    print("-" * 50)
    for split_name, ds in splits.items():
        counts = ds.class_counts
        dist = " ".join(f"{ds.class_names[k]}: {v}" for k, v in sorted(counts.items()))
        print(f"{split_name:<8} {len(ds): >7} {dist}")

    train_ds = splits["train"]

    print(f"\nShow a sample image from the training set with its labels:")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    indices = random.sample(range(len(train_ds)), k=8)
    for i, idx in enumerate(indices):
        img, label = train_ds[idx]
        img_disp = img.permute(1, 2, 0) * std + mean
        img_disp = np.clip(img_disp.numpy(), 0, 1)
        class_name = train_ds.class_names[label]
        axes[i].imshow(img_disp)
        # Fixed typo fontwight -> fontweight, replaced missing turtle 'color' with string
        axes[i].set_title(f"[{class_name}]", color="darkblue", fontsize=12, fontweight="bold")
        axes[i].axis("off")

    plt.suptitle(f"Training Dataset Sample Images | mode={MODE} | mag={MAG}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/home/user/Proj-Ploy/vit_breast_cancer/outputs/dataset_sample.png")
    plt.close()

    print(f"\nVisualize class distribution in the training set:")
    # Fixed typo figsizes -> figsize
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, split_name in enumerate(target_splits):
        ds = splits[split_name]
        counts = ds.class_counts
        class_names = [ds.class_names[k] for k in sorted(counts.keys())]
        class_counts = [counts[k] for k in sorted(counts.keys())]
        ax2[i].bar(class_names, class_counts, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd", "#8c8c8c", "#e377c2"][:len(class_names)])
        ax2[i].set_title(f"{split_name.capitalize()} Set Class Distribution", fontsize=12, fontweight="bold")
        ax2[i].set_xlabel("Class")
        ax2[i].set_ylabel("Number of Samples")
        ax2[i].tick_params(axis="x", rotation=45)
        
    plt.suptitle(f"BreakHis Dataset Class Distribution | mode={MODE} | mag={MAG}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/home/user/Proj-Ploy/vit_breast_cancer/outputs/dataset_distribution.png")
    plt.close()

    print(f"\nVerify WeightedRandomSampler gives balanced sampling probabilities:")
    counts = train_ds.class_counts
    total = sum(counts.values())
    for k, v in sorted(counts.items()):
        w = total / v
        print(f"Class ' {train_ds.class_names[k]:<12} ' | Count: {v:>5} | Weight: {w:>6.2f}")

    print(f"\nThe above weights indicate that the sampler will oversample minority classes and undersample majority classes to achieve a more balanced training distribution.")

    