"""
Evaluation & Attention Map Visualization for ViTfBCD
Computes: Accuracy, Precision, Recall, F1, Specificity, AUC-ROC
Generates: Attention map overlays on histopathological images
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.model import ViTfBCD
from src.dataset import IDX_TO_BINARY, IDX_TO_SUBTYPE, VIT_IMAGE_SIZE


# ── Full evaluation on test set ───────────────────────────────────────────────
def evaluate(model: ViTfBCD, loader: DataLoader, device: torch.device,
             mode: str = "binary", output_dir: str = "/workspace/outputs"):
    """
    Run full evaluation and save metrics + confusion matrix plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    class_names = list(IDX_TO_BINARY.values()) if mode == "binary" else list(IDX_TO_SUBTYPE.values())

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Specificity per class (macro average)
    cm = confusion_matrix(all_labels, all_preds)
    specificities = []
    for i in range(len(class_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity = np.mean(specificities)

    # AUC-ROC
    try:
        if mode == "binary":
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Sensitivity : {rec:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print("="*60)
    print("\nPer-class Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {mode.capitalize()} Classification\nAccuracy: {acc:.4f}", fontsize=13)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    return {"accuracy": acc, "precision": prec, "sensitivity": rec,
            "specificity": specificity, "f1": f1, "auc": auc}


# ── Attention Map Visualization ───────────────────────────────────────────────
def visualize_attention(
    model: ViTfBCD,
    image_path: str,
    device: torch.device,
    image_size: int = VIT_IMAGE_SIZE,
    patch_size: int = 16,
    output_dir: str = "/workspace/outputs",
    head_fusion: str = "mean",   # 'mean' | 'max' | 'min'
):
    """
    Generate attention map overlay for a single histopathological image.
    Shows which regions the model focuses on when making a prediction.

    Args:
        head_fusion: how to combine attention across heads ('mean' recommended)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    orig_img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    tensor   = transform(orig_img).unsqueeze(0).to(device)

    # Extract attention weights from last encoder block
    model.eval()
    attn_weights = model._extract_attention(tensor)
    # attn_weights: [1, num_heads, num_patches+1, num_patches+1]

    if attn_weights is None:
        print("Could not extract attention weights.")
        return

    attn = attn_weights[0]  # [num_heads, seq, seq]

    # Fuse heads
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0).values
    elif head_fusion == "min":
        attn = attn.min(dim=0).values
    attn = attn.cpu().detach().numpy()

    # Attention Rollout: attention from CLS token to all patches
    cls_attn = attn[0, 1:]  # [num_patches]
    num_patches = cls_attn.shape[0]
    grid_size   = int(num_patches ** 0.5)

    attn_map = cls_attn.reshape(grid_size, grid_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to image size
    attn_up = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    attn_up = np.array(attn_up) / 255.0

    # Overlay heatmap on original image
    orig_np = np.array(orig_img) / 255.0
    heatmap  = cm.jet(attn_up)[:, :, :3]
    overlay  = 0.55 * orig_np + 0.45 * heatmap
    overlay  = np.clip(overlay, 0, 1)

    # Save figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_np);    axes[0].set_title("Original Image",   fontsize=13); axes[0].axis("off")
    axes[1].imshow(attn_up, cmap="jet"); axes[1].set_title("Attention Map", fontsize=13); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title("Overlay",           fontsize=13); axes[2].axis("off")

    plt.suptitle("ViTfBCD Attention Visualization — Diseased areas appear brighter", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / f"attention_{Path(image_path).stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Attention map saved to {out_path}")
    return out_path


# ── Batch attention visualization ─────────────────────────────────────────────
def visualize_attention_batch(
    model: ViTfBCD,
    loader: DataLoader,
    dataset,
    device: torch.device,
    n_samples: int = 8,
    output_dir: str = "/workspace/outputs",
):
    """Visualize attention maps for n_samples images from a DataLoader."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    for images, labels in loader:
        for i in range(images.size(0)):
            if images_shown >= n_samples:
                break
            img_tensor = images[i:i+1].to(device)

            attn_weights = model._extract_attention(img_tensor)
            if attn_weights is None:
                continue

            attn = attn_weights[0].mean(dim=0).cpu().detach().numpy()
            cls_attn = attn[0, 1:]
            grid_size = int(cls_attn.shape[0] ** 0.5)
            attn_map = cls_attn.reshape(grid_size, grid_size)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            h = images[i].shape[-1]
            attn_up = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((h, h), Image.BILINEAR)) / 255.0

            # Denormalize image for display
            img_disp = images[i].permute(1, 2, 0) * std + mean
            img_disp = np.clip(img_disp.numpy(), 0, 1)
            heatmap  = cm.jet(attn_up)[:, :, :3]
            overlay  = np.clip(0.55 * img_disp + 0.45 * heatmap, 0, 1)

            label_name = dataset.class_names[labels[i].item()]

            axes[images_shown, 0].imshow(img_disp);  axes[images_shown, 0].set_title(f"Original\n[{label_name}]"); axes[images_shown, 0].axis("off")
            axes[images_shown, 1].imshow(attn_up, cmap="jet"); axes[images_shown, 1].set_title("Attention"); axes[images_shown, 1].axis("off")
            axes[images_shown, 2].imshow(overlay);   axes[images_shown, 2].set_title("Overlay"); axes[images_shown, 2].axis("off")
            images_shown += 1

        if images_shown >= n_samples:
            break

    plt.suptitle("ViTfBCD — Attention Map Visualization (Batch)", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "attention_batch.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Batch attention maps saved to {out_path}")
