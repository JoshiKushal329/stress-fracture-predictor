#!/usr/bin/env python3
"""Run a short smoke training using the synthetic pilot dataset."""
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path so we can import local modules
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))

import torch
from torch.utils.data import DataLoader, Subset
import train as train_mod
from preprocess import StressDataset


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    data_folder = Path('data/training_data')
    ds = StressDataset(str(data_folder))
    n = len(ds)
    if n == 0:
        print('No samples found in', data_folder)
        return

    # split 80/20
    split = int(n * 0.8)
    train_idx = list(range(0, split))
    val_idx = list(range(split, n))

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=8)

    model = train_mod.UNet(in_channels=3, out_channels=2)
    trainer = train_mod.ModelTrainer(model, device=device, mixed_precision=False)

    history = trainer.train(train_loader, val_loader, epochs=3, checkpoint_dir='models', early_stopping_patience=5)
    print('Training history:', history)


if __name__ == '__main__':
    main()
