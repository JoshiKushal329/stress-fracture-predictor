#!/usr/bin/env python3
"""Preprocessing and PyTorch Dataset for stress prediction samples."""
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import List


class StressDataset(Dataset):
    def __init__(self, folder: str, files: List[str] = None):
        self.folder = Path(folder)
        self.files = sorted(self.folder.glob("sample_*.npz")) if files is None else files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        geom = data["geometry"].astype(np.float32)
        res = geom.shape[0]
        # load location heatmap
        y, x = np.ogrid[:res, :res]
        sigma = 30
        load_x = float(data["load_x"])
        load_y = float(data["load_y"])
        heat = np.exp(-((x - load_x)**2 + (y - load_y)**2) / (2*sigma**2)).astype(np.float32)
        mag_field = np.ones_like(geom, dtype=np.float32) * min(float(data["load_magnitude"]) / 10000.0, 1.0)

        input_tensor = np.stack([geom, heat, mag_field], axis=0)
        target = np.stack([data["stress"].astype(np.float32), data["strain"].astype(np.float32)], axis=0)

        return torch.from_numpy(input_tensor), torch.from_numpy(target)


if __name__ == '__main__':
    # quick sanity check
    ds = StressDataset('data/training_data')
    print('Samples found:', len(ds))
    if len(ds) > 0:
        inp, tgt = ds[0]
        print('Input shape:', inp.shape, 'Target shape:', tgt.shape)
