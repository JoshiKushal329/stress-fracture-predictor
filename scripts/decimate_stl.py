#!/usr/bin/env python3
"""STL decimation utility.

Attempts a quality decimation using trimesh's optional quadratic decimation
when available. If not available, falls back to a simple random-face sampling
decimation which reduces facets cheaply.

Usage:
  python3 scripts/decimate_stl.py input.stl --faces 2000 --out output.stl
"""
import argparse
import sys

def try_quadratic(mesh, target_faces):
    # Use trimesh's quadratic simplification if present
    if hasattr(mesh, 'simplify_quadratic_decimation'):
        print(f"Using quadratic decimation to ~{target_faces} faces")
        decim = mesh.simplify_quadratic_decimation(target_faces)
        return decim
    return None

def fallback_random_sample(mesh, target_faces):
    import numpy as np
    print("Using random-face sampling fallback decimation")
    n_faces = len(mesh.faces)
    if n_faces <= target_faces:
        return mesh.copy()

    rng = np.random.default_rng(0)
    chosen = rng.choice(n_faces, size=target_faces, replace=False)
    chosen.sort()
    faces = mesh.faces[chosen]
    verts = mesh.vertices
    unique_vids, inverse = np.unique(faces.flatten(), return_inverse=True)
    new_verts = verts[unique_vids]
    new_faces = inverse.reshape(faces.shape)
    import trimesh
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    return new_mesh

def decimate_file(input_path: str, output_path: str, target_faces: int):
    try:
        import trimesh
    except Exception as e:
        print("ERROR: trimesh not available:", e, file=sys.stderr)
        raise

    mesh = trimesh.load(input_path, force='mesh')
    if mesh.is_empty or len(mesh.faces) == 0:
        raise RuntimeError(f"Loaded mesh is empty or has no faces: {input_path}")

    print(f"Original faces: {len(mesh.faces)}")
    if target_faces <= 0:
        raise ValueError("target faces must be > 0")

    # Try best-effort quadratic decimation
    decim = try_quadratic(mesh, target_faces)
    if decim is None:
        decim = fallback_random_sample(mesh, target_faces)

    print(f"Resulting faces: {len(decim.faces)}")
    decim.export(output_path)
    print(f"Wrote decimated STL to {output_path}")

def main():
    p = argparse.ArgumentParser(description='Decimate an STL file')
    p.add_argument('input', help='Input STL path')
    p.add_argument('--faces', type=int, default=2000, help='Target face count')
    p.add_argument('--out', default=None, help='Output STL path')
    args = p.parse_args()
    out = args.out or args.input.replace('.stl', '_decim.stl')
    try:
        decimate_file(args.input, out, args.faces)
    except Exception as e:
        print(f"Decimation failed: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
