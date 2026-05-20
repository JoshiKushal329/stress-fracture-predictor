#!/usr/bin/env python3
"""Synthetic data generator for stress/strain training samples.

Generates samples in data/training_data as .npz files with fields:
  - geometry: (512,512) binary mask
  - load_x, load_y: pixel coordinates
  - load_magnitude
  - stress: (512,512) float32
  - strain: (512,512) float32

If an STL file is provided, it rasterizes the top view; otherwise creates a simple rectangular bracket mask.
"""
import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import gaussian_filter

def rasterize_top_from_stl(stl_path, res=512):
    try:
        import trimesh
    except Exception:
        raise RuntimeError("trimesh required to rasterize STL. Install with pip install trimesh")
    mesh = trimesh.load(stl_path)
    bounds = mesh.bounds
    w = bounds[1][0] - bounds[0][0]
    h = bounds[1][1] - bounds[0][1]
    img = np.zeros((res, res), dtype=np.uint8)
    for v in mesh.vertices:
        x = int(((v[0] - bounds[0][0]) / w) * (res - 1)) if w > 0 else 0
        y = int(((v[1] - bounds[0][1]) / h) * (res - 1)) if h > 0 else 0
        if 0 <= x < res and 0 <= y < res:
            img[y, x] = 1
    img = gaussian_filter(img.astype(float), sigma=2)
    img = (img > 0.01).astype(np.float32)
    return img

def make_procedural_mask(res=512):
    # Simple L-bracket mask: vertical + horizontal
    img = np.zeros((res, res), dtype=np.float32)
    # vertical arm
    img[50:460, 100:200] = 1.0
    # horizontal arm
    img[360:460, 100:420] = 1.0
    img = gaussian_filter(img, sigma=1)
    img = (img > 0.1).astype(np.float32)
    return img

def make_stress_map(geom_mask, load_x_px, load_y_px, load_mag, res=512):
    y, x = np.ogrid[:res, :res]
    sigma = max(8, int(res * 0.05))
    g = np.exp(-((x - load_x_px)**2 + (y - load_y_px)**2) / (2*sigma*sigma))
    stress = g * (load_mag / 10000.0)
    stress = stress * geom_mask
    # normalize to 0..1
    if stress.max() > 0:
        stress = stress / stress.max()
    strain = stress * 0.01
    return stress.astype(np.float32), strain.astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stl", default=None)
    p.add_argument("--out", default="data/training_data")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--res", type=int, default=512)
    args = p.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    if args.stl:
        geom = rasterize_top_from_stl(args.stl, res=args.res)
    else:
        geom = make_procedural_mask(res=args.res)

    res = args.res
    for i in range(args.n):
        load_x = np.random.uniform(0.15, 0.85) * res
        load_y = np.random.uniform(0.15, 0.85) * res
        load_mag = np.random.uniform(-10000, 10000)
        stress, strain = make_stress_map(geom, int(load_x), int(load_y), abs(load_mag), res)
        sample_file = outdir / f"sample_{i:04d}.npz"
        np.savez(sample_file,
                 geometry=geom.astype(np.float32),
                 load_x=float(load_x), load_y=float(load_y),
                 load_magnitude=float(load_mag),
                 stress=stress,
                 strain=strain)
        if (i+1) % 10 == 0:
            print(f"Saved {sample_file}")
    print("Done. Samples:", args.n)

if __name__ == '__main__':
    main()
