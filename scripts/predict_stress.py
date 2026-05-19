#!/usr/bin/env python3
"""
Inference script: predict stress/strain on GI geometry given a load.
Usage: python scripts/predict_stress.py --geometry data/geometry/vertical_support.stl --load-x 0 --load-y 0 --load-z -500
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from scripts.train import UNet
from scripts.fea_generator_v2 import build_geometry_mask
from scripts.config import (
    denormalize_stress, denormalize_strain,
    STRESS_SCALE_FACTOR, STRAIN_SCALE_FACTOR,
    HOTSPOT_THRESHOLD, HOTSPOT_COUNT
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_stress_strain(
    geometry_stl: str,
    load_x: float,
    load_y: float,
    load_z: float,
    model_path: str = "models/smoke/unet_best.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """
    Predict stress and strain fields for a GI part under a given load.
    
    Args:
        geometry_stl: Path to STL file of your part
        load_x, load_y, load_z: Load components in Newtons
        model_path: Path to trained UNet model
        device: Compute device ('cuda' or 'cpu')
    
    Returns:
        (predicted_stress, predicted_strain): 512x512 numpy arrays (MPa, unitless)
    """
    # Resolve model path relative to script location if it's relative
    model_path = Path(model_path)
    if not model_path.is_absolute():
        script_dir = Path(__file__).parent.parent  # Go up from scripts/ to project root
        model_path = script_dir / model_path
    
    logger.info(f"Loading model from {model_path}")
    model = UNet(in_channels=3, out_channels=2, base_channels=16)
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=device))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    model = model.to(device)
    model.eval()
    

    # Build geometry mask from STL
    logger.info(f"Building geometry mask from {geometry_stl}")
    import meshio
    mesh = meshio.read(geometry_stl)
    points = mesh.points  # (N, 3)
    node_coords_by_id = {i + 1: list(pt) for i, pt in enumerate(points)}
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    bounds = np.array([min_xyz, max_xyz])
    geometry_mask = build_geometry_mask(node_coords_by_id, bounds, resolution=512)  # 512x512
    
    # Create load location heatmap (Gaussian around point for concentrated load)
    load_mag = np.sqrt(load_x**2 + load_y**2 + load_z**2)
    load_mag = max(load_mag, 1e-8)
    load_x_norm = load_x / load_mag
    load_y_norm = load_y / load_mag
    
    # Map load location to pixel coordinates
    min_xy = bounds[0, :2]
    max_xy = bounds[1, :2]
    width = max(max_xy[0] - min_xy[0], 1e-8)
    height = max(max_xy[1] - min_xy[1], 1e-8)
    load_px = ((load_x - min_xy[0]) / width) * 511
    load_py = ((load_y - min_xy[1]) / height) * 511
    
    # Create concentrated Gaussian heatmap around load point (tighter sigma for point concentration)
    y_coords, x_coords = np.ogrid[:512, :512]
    sigma = 8.0  # Tighter Gaussian for concentrated load
    heat = np.exp(-((x_coords - load_px)**2 + (y_coords - load_py)**2) / (2 * sigma**2)).astype(np.float32)
    
    # Normalize and apply load direction
    if heat.max() > 0:
        heat = heat / heat.max()
    
    # Create direction maps
    load_x_map = heat * load_x_norm
    load_y_map = heat * load_y_norm
    logger.info(f"Applying CONCENTRATED LOAD at ({load_x:.1f}, {load_y:.1f}) with sigma={sigma} pixels")
    
    # Stack into 3-channel input
    input_tensor = torch.stack([
        torch.from_numpy(geometry_mask).float(),
        torch.from_numpy(load_x_map).float(),
        torch.from_numpy(load_y_map).float()
    ], dim=0).unsqueeze(0).to(device)  # (1, 3, 512, 512)
    
    logger.info(f"Predicting stress/strain for load: ({load_x:.1f}, {load_y:.1f}, {load_z:.1f}) N")
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # output: (1, 2, 512, 512)
        # channel 0 = stress, channel 1 = strain
        predicted_stress_normalized = output[0, 0].cpu().numpy()
        predicted_strain_normalized = output[0, 1].cpu().numpy()
    
    # ========== DENORMALIZATION ==========
    # Convert from normalized (0-1) to actual MPa and strain values
    logger.info(f"Denormalizing outputs (scale factor: {STRESS_SCALE_FACTOR:.1f} MPa)")
    predicted_stress = denormalize_stress(predicted_stress_normalized)
    predicted_strain = denormalize_strain(predicted_strain_normalized)
    
    logger.info(f"✅ Denormalization complete:")
    logger.info(f"   Stress: normalized [0-1] → actual MPa [0-{STRESS_SCALE_FACTOR:.1f}]")
    logger.info(f"   Strain: normalized [0-1] → actual [0-{STRAIN_SCALE_FACTOR:.6f}]")
    
    return predicted_stress, predicted_strain


def find_fracture_points(
    stress_field: np.ndarray,
    threshold: float = None,
    top_n: int = None
) -> list:
    """
    Find high-stress regions in the predicted stress field.
    
    Args:
        stress_field: 512x512 stress map (MPa)
        threshold: Stress threshold (MPa); GI yield ~300 MPa
        top_n: Return top N hotspots
    
    Returns:
        List of [(x, y, stress_value), ...] sorted by stress descending
    """
    if threshold is None:
        threshold = HOTSPOT_THRESHOLD
    if top_n is None:
        top_n = HOTSPOT_COUNT
    
    # Find pixels above threshold
    hotspots = np.argwhere(stress_field > threshold)
    
    if len(hotspots) == 0:
        logger.warning(f"No stress points above {threshold} MPa found")
        # Return top N absolute peaks instead
        flat_idx = np.argsort(stress_field.flatten())[-top_n:]
        y_coords, x_coords = np.unravel_index(flat_idx, stress_field.shape)
        hotspots = list(zip(x_coords, y_coords))
    else:
        # Sort by stress value
        stresses = [stress_field[y, x] for y, x in hotspots]
        sorted_pairs = sorted(zip(hotspots, stresses), key=lambda p: p[1], reverse=True)
        hotspots = [p[0] for p in sorted_pairs[:top_n]]
    
    result = [(x, y, stress_field[y, x]) for y, x in hotspots]
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict stress/strain on GI part")
    parser.add_argument("--geometry", type=str, required=True, help="STL file path")
    parser.add_argument("--load-x", type=float, default=0.0, help="Load X component (N)")
    parser.add_argument("--load-y", type=float, default=0.0, help="Load Y component (N)")
    parser.add_argument("--load-z", type=float, default=-500.0, help="Load Z component (N)")
    parser.add_argument("--model", type=str, default="models/smoke/unet_best.pth", help="Model path")
    parser.add_argument("--threshold", type=float, default=None, help="Hotspot stress threshold (MPa)")
    parser.add_argument("--output", type=str, default="prediction_output", help="Output dir")
    
    args = parser.parse_args()
    
    # Run prediction
    predicted_stress, predicted_strain = predict_stress_strain(
        geometry_stl=args.geometry,
        load_x=args.load_x,
        load_y=args.load_y,
        load_z=args.load_z,
        model_path=args.model
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PREDICTION RESULTS (ACTUAL VALUES IN MPa)")
    logger.info(f"{'='*70}")
    logger.info(f"Stress range: {predicted_stress.min():.2f} - {predicted_stress.max():.2f} MPa")
    logger.info(f"Strain range: {predicted_strain.min():.8f} - {predicted_strain.max():.8f}")
    logger.info(f"{'='*70}\n")
    
    # Find hotspots
    hotspots = find_fracture_points(predicted_stress, threshold=args.threshold, top_n=HOTSPOT_COUNT)
    logger.info(f"Top {HOTSPOT_COUNT} fracture-prone regions (GI yield ~300 MPa):")
    for i, (x, y, stress) in enumerate(hotspots, 1):
        logger.info(f"  {i}. Position ({x}, {y}): {stress:.2f} MPa")
    
    # Save outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "predicted_stress.npy", predicted_stress)
    np.save(out_dir / "predicted_strain.npy", predicted_strain)
    
    # Save hotspots to JSON
    import json
    hotspot_data = {
        "material": "Galvanized Iron (GI)",
        "young_modulus_mpa": 200000,
        "yield_strength_mpa": 300,
        "load": {"x": args.load_x, "y": args.load_y, "z": args.load_z},
        "max_stress_mpa": float(predicted_stress.max()),
        "stress_scale_factor_mpa": float(STRESS_SCALE_FACTOR),
        "strain_scale_factor": float(STRAIN_SCALE_FACTOR),
        "note": "All stress values are in actual MPa (denormalized)",
        "fracture_points": [
            {"x": int(x), "y": int(y), "stress_mpa": float(s)}
            for x, y, s in hotspots
        ]
    }
    with open(out_dir / "hotspots.json", "w") as f:
        json.dump(hotspot_data, f, indent=2)
    
    # Create and save heatmap visualization
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Stress heatmap (in MPa)
        im1 = ax1.imshow(predicted_stress, cmap='hot', origin='lower')
        ax1.set_title(f'Stress Distribution (Max: {predicted_stress.max():.1f} MPa)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Stress (MPa)')
        
        # Strain heatmap
        im2 = ax2.imshow(predicted_strain, cmap='viridis', origin='lower')
        ax2.set_title(f'Strain Distribution (Max: {predicted_strain.max():.8f})', fontsize=12, fontweight='bold')
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
        logger.info(f"Heatmap saved to {heatmap_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create heatmap: {e}")
    
    logger.info(f"\nOutputs saved to {out_dir}/")
    logger.info("  - predicted_stress.npy (512x512 stress map in MPa)")
    logger.info("  - predicted_strain.npy (512x512 strain map)")
    logger.info("  - hotspots.json (fracture point locations with scale factors)")
    logger.info(f"\n✅ Scale factors used:")
    logger.info(f"   Stress: {STRESS_SCALE_FACTOR:.1f} MPa")
    logger.info(f"   Strain: {STRAIN_SCALE_FACTOR:.6f}")


if __name__ == "__main__":
    main()
