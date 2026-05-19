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
    # Parse args to allow overriding data folder
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fea_training_data", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to train")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    dataset = StressDataset(str(data_dir))
    if len(dataset) < 4:
        raise RuntimeError("Need at least 4 samples to split correctly")

    total_len = len(dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)

    print(f"Data split: {train_size} Train | {val_size} Val | {test_size} Test  (Total: {total_len})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = min(DEFAULT_BATCH_SIZE, len(train_ds))
    batch_size = min(batch_size, 4) # ensure we don't OOM local RTX card

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
    test_loader = DataLoader(
        test_ds,
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

    epochs = args.epochs
    print(f"Training on {len(dataset)} samples | device={device} | batch_size={batch_size} | epochs={epochs}")
    trainer.train(train_loader, val_loader, epochs=epochs, checkpoint_dir="models/smoke", early_stopping_patience=10)

    print("\n--- Training Complete ---")
    
    # Run accuracy test on Test Set
    print("\nStarting Test Set Evaluation...")
    best_path = Path("models/smoke/unet_best.pth")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = trainer.validate(test_loader)
    print(f"\nFinal TEST SET Loss (MSE): {test_loss:.6f}")

if __name__ == "__main__":
    main()
