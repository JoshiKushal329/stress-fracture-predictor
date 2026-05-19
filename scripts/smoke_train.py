#!/usr/bin/env python3
"""Run a small training smoke test on the generated pilot dataset.

This uses the lightweight U-Net settings in scripts/train.py and trains for a
small number of epochs so the pipeline can be validated quickly on a laptop GPU.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from preprocess import StressDataset
from train import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, ModelTrainer, UNet


def main():
    data_dir = Path("data/training_data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = StressDataset(str(data_dir))
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples for a smoke train")

    split = max(1, int(len(dataset) * 0.8))
    train_ds = Subset(dataset, list(range(split)))
    val_ds = Subset(dataset, list(range(split, len(dataset))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = min(DEFAULT_BATCH_SIZE, len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = UNet(in_channels=3, out_channels=2, base_channels=16)
    trainer = ModelTrainer(
        model,
        device=device,
        mixed_precision=(device == "cuda"),
        batch_size=batch_size,
    )

    epochs = min(2, DEFAULT_EPOCHS)
    print(f"Smoke training on {len(dataset)} samples | device={device} | batch_size={batch_size} | epochs={epochs}")
    trainer.train(train_loader, val_loader, epochs=epochs, checkpoint_dir="models/smoke", early_stopping_patience=2)


if __name__ == "__main__":
    main()
