"""
Training loop for ViTfBCD
Handles: training, validation, early stopping, LR scheduling, checkpointing
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

from src.model import ViTfBCD


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Full training pipeline for ViTfBCD.

    Args:
        model       : ViTfBCD instance
        config      : training hyperparameter dict
        device      : torch device
        output_dir  : directory to save checkpoints and logs
    """

    def __init__(
        self,
        model: ViTfBCD,
        config: dict,
        device: torch.device,
        output_dir: str = "/workspace/outputs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss — use label smoothing to reduce overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))

        # Optimizer — AdamW with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Scheduler — cosine annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 30),
            eta_min=config.get("min_lr", 1e-6),
        )

        # Mixed precision scaler (GPU only)
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.use_amp = device.type == "cuda"

        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            delta=config.get("es_delta", 1e-4),
        )

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    # ── Single epoch ──────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    # Gradient clipping to stabilize transformer training
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        epochs = self.config.get("epochs", 30)
        print(f"\n{'='*60}")
        print(f"  Training ViTfBCD-{self.config.get('model_size','base').capitalize()}")
        print(f"  Mode: {self.config.get('mode','binary')} | Epochs: {epochs}")
        print(f"  Device: {self.device} | AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   training=False)

            self.scheduler.step()
            elapsed = time.time() - t0

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best_model.pt")
                print(f"  ✓ New best val acc: {val_acc:.4f} — checkpoint saved")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        # Save final history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. Best val acc: {self.best_val_acc:.4f}")
        print(f"History saved to {history_path}")

        return self.history

    def _save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }, path)

    def load_best(self):
        """Load the best checkpoint back into the model."""
        ckpt = torch.load(self.output_dir / "best_model.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (val_acc={ckpt['best_val_acc']:.4f})")
