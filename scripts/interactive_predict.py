#!/usr/bin/env python3
"""
Interactive console for stress/strain prediction using the trained U-Net model.
Prompts the user for STL file, load location, and load magnitude, then runs prediction.
"""

import os
import sys
import torch
from pathlib import Path
from .predict_stress import predict_stress_strain, find_fracture_points
from .fea_generator_v2 import build_geometry_mask

DEFAULT_MODEL_PATH = "models/smoke/unet_best.pth"


def main():
    print("\n=== Stress/Strain Prediction Console ===\n")

    stl_path = input("Enter path to STL file: ").strip()
    if not os.path.exists(stl_path):
        print(f"File not found: {stl_path}")
        sys.exit(1)

    model_path = input(f"Enter path to model .pth file [default: {DEFAULT_MODEL_PATH}]: ").strip()
    if not model_path:
        model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    # Get bounding box for guidance
    import meshio
    mesh = meshio.read(stl_path)
    points = mesh.points
    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)
    print(f"Part bounding box: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}], Z=[{min_z:.1f}, {max_z:.1f}]")

    # Ask for load location (X, Y in mm)
    load_x = float(input(f"Enter X location for load (in mm, {min_x:.1f} to {max_x:.1f}): ").strip())
    load_y = float(input(f"Enter Y location for load (in mm, {min_y:.1f} to {max_y:.1f}): ").strip())
    load_z = float(input("Enter vertical load magnitude (negative for downward, e.g. -5000): ").strip())

    # Run prediction
    print("\nRunning prediction...")
    try:
        stress, strain = predict_stress_strain(
            geometry_stl=stl_path,
            load_x=load_x,
            load_y=load_y,
            load_z=load_z,
            model_path=model_path
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    print(f"\nPrediction complete. Stress range: {stress.min():.2f} - {stress.max():.2f} MPa")
    print(f"Strain range: {strain.min():.6f} - {strain.max():.6f}")

    # Find and print fracture points
    hotspots = find_fracture_points(stress, threshold=250, top_n=10)
    print("\nTop 10 fracture-prone regions (stress > 250 MPa):")
    for i, (x, y, s) in enumerate(hotspots, 1):
        print(f"  {i}. Pixel ({x}, {y}): {s:.1f} MPa")

    # Optionally save results
    save = input("Save results to .npy/.json files and heatmap? [y/N]: ").strip().lower()
    if save == 'y':
        out_dir = Path("prediction_results")
        out_dir.mkdir(exist_ok=True)
        import numpy as np, json
        np.save(out_dir / "predicted_stress.npy", stress)
        np.save(out_dir / "predicted_strain.npy", strain)
        with open(out_dir / "hotspots.json", "w") as f:
            # Convert numpy types to Python native types for JSON serialization
            hotspots_serializable = [(int(x), int(y), float(s)) for x, y, s in hotspots]
            json.dump({"fracture_points": hotspots_serializable}, f, indent=2)
        
        # Create and save heatmap visualization
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Stress heatmap
            im1 = ax1.imshow(stress, cmap='hot', origin='lower')
            ax1.set_title(f'Stress Distribution (Max: {stress.max():.1f} MPa)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Stress (MPa)')
            
            # Strain heatmap
            im2 = ax2.imshow(strain, cmap='viridis', origin='lower')
            ax2.set_title(f'Strain Distribution (Max: {strain.max():.6f})', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=ax2, label='Strain')
            
            # Mark hotspots on stress heatmap
            for i, (x, y, s) in enumerate(hotspots[:5], 1):
                ax1.plot(x, y, 'co', markersize=8, markerfacecolor='none', markeredgewidth=2)
                ax1.text(x, y-20, f'{i}', color='cyan', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            heatmap_path = out_dir / "stress_strain_heatmap.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved to {heatmap_path}")
            plt.close()
        except Exception as e:
            print(f"Could not create heatmap: {e}")
        
        print(f"Results saved to {out_dir}/")

if __name__ == "__main__":
    main()
