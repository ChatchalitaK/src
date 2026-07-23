"""
ViTfBCD - Vision Transformer for Breast Cancer Detection
Adapted from ViTfSCD (Yang et al., 2023) for histopathological images.
Supports both Binary (Benign/Malignant) and Multi-class (8 subtypes).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights


class ClassificationHead(nn.Module):
    """
    Custom classification head replacing ViT's default MLP head.
    Block 4: Flatten -> BatchNorm -> Dense(GeLU) -> BatchNorm -> Softmax
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ViTfBCD(nn.Module):
    """
    ViT for Breast Cancer Detection (ViTfBCD)

    Args:
        model_size : 'base' (86M params, 12 layers) or 'large' (307M params, 24 layers)
        num_classes: 2 for binary, 8 for multi-class subtypes
        pretrained  : use ImageNet-21k pretrained weights
        dropout     : dropout rate in classification head
    """
    def __init__(
        self,
        model_size: str = "base",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes

        if model_size == "base":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            hidden_dim = self.vit.heads.head.in_features  # 768
        elif model_size == "large":
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_l_16(weights=weights)
            hidden_dim = self.vit.heads.head.in_features  # 1024
        else:
            raise ValueError(f"model_size must be 'base' or 'large', got '{model_size}'")

        # Replace default head with custom classification block
        self.vit.heads = ClassificationHead(hidden_dim, num_classes, dropout)

    def forward(self, x):
        return self.vit(x)

    def get_attention_maps(self, x):
        """
        Extract attention maps from the last transformer encoder block.
        Returns attention weights of shape [B, num_heads, num_patches+1, num_patches+1]
        """
        attn_maps = []

        def hook_fn(module, input, output):
            # output shape: [B, num_patches+1, hidden_dim]
            # We access the attention weights stored before projection
            attn_maps.append(module.attn_weights if hasattr(module, 'attn_weights') else None)

        # Patch: override forward of last encoder block to capture attention
        encoder_blocks = self.vit.encoder.layers
        last_block = encoder_blocks[-1]

        attention_weights = []

        def attn_hook(module, input, output):
            # MultiheadAttention returns (output, attn_weights)
            pass

        # Use manual forward pass to extract attention
        return self._extract_attention(x)

    def _extract_attention(self, x):
        """Manual forward pass capturing attention weights from last block."""
        with torch.no_grad():
            # Process through patch embedding
            x = self.vit._process_input(x)
            n = x.shape[0]

            # Prepend class token
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Process through encoder layers, capture last layer attention
            attn_weights = None
            for i, block in enumerate(self.vit.encoder.layers):
                # Layer norm before attention
                normed = block.ln_1(x)
                # Get attention weights from MultiheadAttention
                _, weights = block.self_attention(normed, normed, normed, need_weights=True, average_attn_weights=False)
                if i == len(self.vit.encoder.layers) - 1:
                    attn_weights = weights  # [B, heads, seq, seq]
                x = block(x)

            return attn_weights  # [B, num_heads, num_patches+1, num_patches+1]

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_size": self.model_size,
            "num_classes": self.num_classes,
            "total_params": f"{total_params / 1e6:.1f}M",
            "trainable_params": f"{trainable_params / 1e6:.1f}M",
        }


def build_model(config: dict) -> ViTfBCD:
    """Build model from config dictionary."""
    return ViTfBCD(
        model_size=config.get("model_size", "base"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
    )
