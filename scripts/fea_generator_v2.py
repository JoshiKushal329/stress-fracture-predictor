#!/usr/bin/env python3
"""Working FEA data generation with Gmsh meshing and CalculiX analysis."""
import numpy as np
from pathlib import Path
import subprocess
import logging
import json
import re
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLOAT_RE = re.compile(r'[+-]?(?:\d+\.\d+|\d+)(?:[Ee][+-]?\d+)?')


def compute_von_mises(components: List[float]) -> float:
    """Compute von Mises stress from a 6-component tensor."""
    if len(components) < 6:
        return float(np.linalg.norm(components))
    sxx, syy, szz, sxy, syz, szx = components[:6]
    return float(np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    ))


def compute_equivalent_strain(components: List[float]) -> float:
    """Compute a simple equivalent strain magnitude from a 6-component tensor."""
    if len(components) < 6:
        return float(np.linalg.norm(components))
    exx, eyy, ezz, exy, eyz, ezx = components[:6]
    return float(np.sqrt(exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * (exy ** 2 + eyz ** 2 + ezx ** 2)))


def parse_frd_results(frd_file: str) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """Parse CalculiX FRD text results into a block dictionary."""
    blocks: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    current_name = None
    current_labels: List[str] = []
    reading_nodes = False

    with open(frd_file, 'r', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            if line.strip().startswith('2C'):
                reading_nodes = True
                blocks.setdefault('NODES', {'labels': {}, 'data': {}})
                continue

            if line.startswith(' -4'):
                reading_nodes = False
                parts = line.split()
                if len(parts) >= 2:
                    current_name = parts[1].upper()
                    current_labels = []
                    blocks[current_name] = {'labels': {}, 'data': {}}
                continue

            if line.startswith(' -5') and current_name:
                parts = line.split()
                if len(parts) >= 2:
                    current_labels.append(parts[1].upper())
                    blocks[current_name]['labels'][len(current_labels) - 1] = parts[1].upper()
                continue

            if line.startswith(' -1') and reading_nodes and current_name is None:
                nums = FLOAT_RE.findall(line)
                if len(nums) >= 5:
                    node_id = int(float(nums[1]))
                    coords = [float(nums[2]), float(nums[3]), float(nums[4])]
                    blocks['NODES']['data'][node_id] = coords
                continue

            if line.startswith(' -1') and current_name:
                nums = FLOAT_RE.findall(line)
                if len(nums) >= 2:
                    node_id = int(float(nums[1]))
                    values = [float(v) for v in nums[2:]]
                    blocks[current_name]['data'][node_id] = values
                continue

            if line.startswith(' -3'):
                current_name = None

    return blocks


def rasterize_nodal_field(node_coords_by_id: Dict[int, List[float]], values_by_node: Dict[int, float], bounds: np.ndarray,
                          resolution: int = 512, blur_sigma: float = 2.0) -> np.ndarray:
    """Project a nodal scalar field onto a 2D image grid."""
    canvas = np.zeros((resolution, resolution), dtype=np.float32)
    min_x, min_y = bounds[0][0], bounds[0][1]
    max_x, max_y = bounds[1][0], bounds[1][1]
    width = max(max_x - min_x, 1e-8)
    height = max(max_y - min_y, 1e-8)

    for node_id, value in values_by_node.items():
        coords = node_coords_by_id.get(node_id)
        if coords is None:
            continue
        x, y = coords[0], coords[1]
        px = int(((x - min_x) / width) * (resolution - 1))
        py = int(((y - min_y) / height) * (resolution - 1))
        if 0 <= px < resolution and 0 <= py < resolution:
            if value > canvas[py, px]:
                canvas[py, px] = value

    try:
        from scipy.ndimage import gaussian_filter
        canvas = gaussian_filter(canvas, sigma=blur_sigma)
    except Exception:
        pass

    peak = float(np.max(canvas))
    if peak > 0:
        canvas /= peak
    return canvas.astype(np.float32)


def build_geometry_mask(node_coords_by_id: Dict[int, List[float]], bounds: np.ndarray, resolution: int = 512) -> np.ndarray:
    """Rasterize mesh vertices to a soft geometry mask."""
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    min_x, min_y = bounds[0][0], bounds[0][1]
    max_x, max_y = bounds[1][0], bounds[1][1]
    width = max(max_x - min_x, 1e-8)
    height = max(max_y - min_y, 1e-8)

    for coords in node_coords_by_id.values():
        x, y, _ = coords
        px = int(((x - min_x) / width) * (resolution - 1))
        py = int(((y - min_y) / height) * (resolution - 1))
        if 0 <= px < resolution and 0 <= py < resolution:
            mask[py, px] = 1.0

    try:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=2.0)
    except Exception:
        pass

    peak = float(np.max(mask))
    if peak > 0:
        mask /= peak
    return mask.astype(np.float32)


def pick_support_nodes(points: np.ndarray, band_fraction: float = 0.05) -> np.ndarray:
    """Pick nodes near the minimum Y edge to act as a fixed support."""
    ys = points[:, 1]
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    band = max((y_max - y_min) * band_fraction, 1e-6)
    return np.where(ys <= y_min + band)[0] + 1


def pick_load_nodes(points: np.ndarray, load_x: float, load_y: float, radius: float = 15.0) -> list:
    """Choose multiple nodes within a radius to distribute the load and avoid unrealistic singularities."""
    xy = points[:, :2]
    target = np.array([load_x, load_y], dtype=np.float32)
    dists = np.linalg.norm(xy - target[None, :], axis=1)
    
    # Find nodes within the specified radius
    nodes = np.where(dists <= radius)[0] + 1
    
    # If no nodes fall strictly within the radius, take the 3 closest to spread it a bit
    if len(nodes) == 0:
        nodes = np.argsort(dists)[:3] + 1
        
    return nodes.tolist()


def generate_design_recommendation(load_x: float, load_y: float, hotspot_xy: Tuple[float, float], bounds: np.ndarray, max_stress: float) -> dict:
    """Create material and dimensional recommendations from the dominant hotspot and max stress."""
    min_x, min_y = bounds[0][0], bounds[0][1]
    max_x, max_y = bounds[1][0], bounds[1][1]
    width = max(max_x - min_x, 1e-8)
    height = max(max_y - min_y, 1e-8)
    hx, hy = hotspot_xy
    rel_y = (hy - min_y) / height
    rel_x = (hx - min_x) / width
    load_dx = abs(hx - load_x) / width
    load_dy = abs(hy - load_y) / height

    # GI Yield strength is ~250 MPa
    yield_strength = 250.0 
    safety_factor = yield_strength / max(max_stress, 1e-3)
    
    # Material Recommendation
    if max_stress > yield_strength:
        if max_stress < 450:
            mat_rec = "Consider High-Strength Low-Alloy (HSLA) Steel or SS304"
        else:
            mat_rec = "Consider Advanced High-Strength Steel or Titanium Alloy"
    elif max_stress < 50:
        mat_rec = "Consider Aluminum 6061 or GFRP to reduce weight"
    else:
        mat_rec = "Galvanized Iron (Current) is sufficient"
        
    # Dimension adjustments based on load experienced
    if safety_factor < 1.0:
        thick_inc = int((1.0 / safety_factor - 1) * 100 + 20)  # Add 20% margin
        dim_rec = f"Increase local thickness/width at hotspot (X:{hx:.1f}, Y:{hy:.1f}) by ~{thick_inc}% to handle over-stress safely."
    elif safety_factor < 1.5:
        thick_inc = 15
        dim_rec = f"Increase local thickness by ~{thick_inc}% at hotspot (X:{hx:.1f}, Y:{hy:.1f}) for adequate safety margin."
    else:
        dim_rec = "Current dimensions are adequate for this load configuration."

    if rel_y < 0.2:
        struct_rec = 'Add a larger fillet / gusset where the support base meets the vertical structure.'
    elif load_dx < 0.15 and load_dy < 0.15:
        struct_rec = 'Add a localized stiffening pad at the load contact area to distribute the applied force.'
    else:
        struct_rec = 'Add a vertical rib spanning the hotspot to reduce localized bending stress.'
        
    return {
        "material_recommendation": mat_rec,
        "dimension_modification": dim_rec,
        "structural_feature": struct_rec,
        "safety_factor": round(safety_factor, 2)
    }


def mesh_stl_with_gmsh(stl_file: str, mesh_size: float = 5.0) -> str:
    """Create tetrahedral mesh from STL using Gmsh, output in CalculiX format."""
    stl_abs = str(Path(stl_file).resolve())
    output_msh = Path(stl_file).with_suffix('.msh')
    msh_abs = str(output_msh.resolve())
    
    logger.info(f"Meshing {stl_abs}...")
    
    gmsh_script = f"""
Merge "{stl_abs}";
Mesh.CharacteristicLengthMax = {mesh_size};
Mesh.CharacteristicLengthMin = {mesh_size * 0.2};
Mesh 3;
Save "{msh_abs}";
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geo', delete=False) as f:
        f.write(gmsh_script)
        script_file = f.name
    
    try:
        # Request legacy MSH2 format which is easier to parse/convert with meshio
        result = subprocess.run(['/usr/bin/gmsh', script_file, '-o', msh_abs, '-format', 'msh2', '-nopopup'],
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"Gmsh stderr: {result.stderr}")
            logger.error(f"Gmsh stdout: {result.stdout}")
    finally:
        Path(script_file).unlink()
    
    logger.info(f"✅ Mesh created: {output_msh}")
    return str(output_msh)


def convert_msh_to_ccx_inp(msh_file: str, output_inp: str = None) -> str:
    """Convert a Gmsh surface mesh to a CalculiX shell model input file."""
    if output_inp is None:
        output_inp = Path(msh_file).with_suffix('.inp')
    
    logger.info(f"Converting Gmsh MSH to CalculiX shell input (.inp)...")
    try:
        import meshio
        mesh = meshio.read(msh_file)
        points = np.asarray(mesh.points)
        triangles = np.asarray(mesh.cells_dict.get('triangle', []))
        if triangles.size == 0:
            raise ValueError('No triangle cells found in mesh')

        with open(output_inp, 'w') as f:
            f.write('*HEADING\n')
            f.write('FEA shell analysis\n')
            f.write('*NODE\n')
            for node_id, (x, y, z) in enumerate(points, start=1):
                f.write(f'{node_id}, {x}, {y}, {z}\n')
            f.write('*ELEMENT, TYPE=S3, ELSET=EALL\n')
            for elem_id, tri in enumerate(triangles, start=1):
                n1, n2, n3 = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
                f.write(f'{elem_id}, {n1}, {n2}, {n3}\n')
            f.write('*MATERIAL, NAME=GI\n')
            f.write('*ELASTIC\n')
            f.write('200000, 0.27\n')  # E=200 GPa, nu=0.27 for Galvanized Iron
            f.write('*DENSITY\n')
            f.write('7850\n')  # Density in kg/m^3 for GI
            f.write('*SHELL SECTION, ELSET=EALL, MATERIAL=GI\n')
            f.write('1.0\n')

        logger.info(f"Converted MSH -> {output_inp} as a shell mesh")
        return str(output_inp)
    except Exception as e:
        logger.error(f"Shell conversion failed: {e}")
        raise


def run_calculix(inp_file: str, show_gui: bool = False) -> str:
    """Run CalculiX solver with optional GUI."""
    inp_path = Path(inp_file)
    job_name = inp_path.stem
    work_dir = inp_path.parent
    
    logger.info(f"Running CalculiX: {inp_file}")
    
    if show_gui:
        # Use CGX (GUI) to show results
        logger.info("Opening CalculiX GUI (cgx)...")
        try:
            subprocess.Popen(['/usr/bin/cgx', '-b', str(inp_file)], cwd=str(work_dir))
        except Exception as e:
            logger.error(f"Failed to launch cgx: {e}")
    else:
        # Use CCX (solver) synchronously so we know results are ready.
        try:
            result = subprocess.run(['/usr/bin/ccx', job_name], cwd=str(work_dir), capture_output=True, text=True, timeout=300)
            logger.info(f"ccx exited with code {result.returncode}")
            if result.stdout:
                logger.info(result.stdout[:1000])
            if result.stderr:
                logger.warning(result.stderr[:1000])
        except FileNotFoundError:
            logger.error("ccx executable not found at /usr/bin/ccx")
            return None
        except subprocess.TimeoutExpired:
            logger.error("ccx timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to start ccx: {e}")
            return None
    
    # Check for results
    frd_file = work_dir / f"{job_name}.frd"
    if frd_file.exists():
        logger.info(f"✅ Results: {frd_file}")
        return str(frd_file)
    else:
        logger.info("FRD not present yet; solver may still be running or produced different outputs.")
        return None


def generate_realistic_stress_map(resolution: int = 512, load_x: float = 256, load_y: float = 256, 
                                 load_mag: float = 500) -> np.ndarray:
    """Generate a realistic stress distribution based on load location."""
    y, x = np.ogrid[:resolution, :resolution]
    # Gaussian blob at load location
    sigma = max(20, int(resolution * 0.08))
    stress = np.exp(-((x - load_x)**2 + (y - load_y)**2) / (2*sigma**2))
    # Scale by load magnitude
    stress = stress * (load_mag / 1000.0)
    # Add some spatial variation
    noise = np.random.rand(resolution, resolution) * 0.1
    stress = stress + noise
    stress = np.clip(stress, 0, 1)
    return stress.astype(np.float32)


class FEADataGenerator:
    """Generate training samples with real geometry and load-aware stress."""
    
    def __init__(self, stl_file: str, output_dir: str = 'data/fea_training_data', mesh_file: str = None):
        self.stl_file = stl_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_file = mesh_file
    
    def generate_samples(self, num_samples: int = 5, start_idx: int = 0, show_gui: bool = False):
        """Generate samples."""
        logger.info(f"\nGenerating {num_samples} FEA samples starting at {start_idx}...\n")
        
        # Step 1: Mesh or reuse an existing mesh if one is already present.
        if self.mesh_file:
            msh_file = self.mesh_file
            logger.info(f"Reusing provided mesh: {msh_file}")
        else:
            cached_msh = Path(self.stl_file).with_suffix('.msh')
            if cached_msh.exists():
                msh_file = str(cached_msh)
                logger.info(f"Reusing cached mesh: {msh_file}")
            else:
                # Fall back to meshing the STL when no cached mesh exists.
                msh_file = mesh_stl_with_gmsh(self.stl_file, mesh_size=10.0)

        import meshio
        base_mesh = meshio.read(msh_file)
        points = np.asarray(base_mesh.points)
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        geom_mask = build_geometry_mask({i + 1: [float(x), float(y), float(z)] for i, (x, y, z) in enumerate(points)}, bounds, resolution=512)
        logger.info(f"Geometry mask created from mesh: {geom_mask.mean() * 100:.1f}% average coverage")
        
        # Load existing metadata if appending
        metadata_file = self.output_dir / 'metadata.json'
        metadata = []
        if start_idx > 0 and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded {len(metadata)} existing metadata entries.")
            except Exception as e:
                logger.error(f"Could not read metadata.json: {e}")
        
        # Step 2: Generate training samples by solving one load case per sample.
        for i in range(start_idx, start_idx + num_samples):
            logger.info(f"\nGenerating sample {i+1}/{num_samples}...")
            
            # Random load parameters
            min_x, min_y = bounds[0][0], bounds[0][1]
            max_x, max_y = bounds[1][0], bounds[1][1]
            load_x = float(np.random.uniform(min_x + 0.1 * (max_x - min_x), max_x - 0.1 * (max_x - min_x)))
            
            # Since vertical load usually acts from top, bias generation towards upper half
            load_y = float(np.random.uniform(min_y + 0.5 * (max_y - min_y), max_y - 0.05 * (max_y - min_y)))
            
            # Massive vertical loads (2000N - 10000N)
            load_mag = float(np.random.uniform(2000, 10000))

            case_inp = self.output_dir / f"case_{i:04d}.inp"
            case_job = self.output_dir / f"case_{i:04d}"
            case_frd = case_job.with_suffix('.frd')

            # Build a shell-based CalculiX input for this load case.
            with open(case_inp, 'w') as f:
                f.write('*HEADING\n')
                f.write(f'FEA case {i:04d}\n')
                f.write('*NODE\n')
                for node_id, (x, y, z) in enumerate(points, start=1):
                    f.write(f'{node_id}, {x}, {y}, {z}\n')
                f.write('*ELEMENT, TYPE=S3, ELSET=EALL\n')
                triangles = np.asarray(base_mesh.cells_dict['triangle'])
                for elem_id, tri in enumerate(triangles, start=1):
                    n1, n2, n3 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                    f.write(f'{elem_id}, {n1}, {n2}, {n3}\n')
                f.write('*MATERIAL, NAME=GI\n')
                f.write('*ELASTIC\n')
                f.write('200000, 0.27\n')  # E=200 GPa, nu=0.27 for Galvanized Iron
                f.write('*DENSITY\n')
                f.write('7850\n')  # Density in kg/m^3 for GI
                f.write('*SHELL SECTION, ELSET=EALL, MATERIAL=GI\n')
                f.write('1.0\n')

                support_nodes = pick_support_nodes(points)
                load_nodes = pick_load_nodes(points, load_x, load_y, radius=15.0)

                f.write('*BOUNDARY\n')
                for node_id in support_nodes:
                    f.write(f'{node_id}, 1, 6, 0.0\n')
                f.write('*STEP\n')
                f.write('*STATIC\n')
                f.write('*CLOAD\n')
                
                # Distribute the load evenly across the selected nodes in the contact patch
                distributed_load = load_mag / len(load_nodes)
                for load_node in load_nodes:
                    f.write(f'{load_node}, 2, {-distributed_load:.6f}\n')  # 2 is Y-axis, -load acts vertically downwards
                    
                f.write('*NODE FILE\n')
                f.write('U\n')
                f.write('*EL FILE\n')
                f.write('S, E\n')
                f.write('*END STEP\n')

            # Solve the load case.
            run_calculix(str(case_inp), show_gui=show_gui)
            if not case_frd.exists():
                raise RuntimeError(f'No FRD produced for {case_inp}')

            blocks = parse_frd_results(str(case_frd))
            node_coords = blocks.get('NODES', {}).get('data', {})
            stress_block = blocks.get('STRESS', {}).get('data', {})
            strain_block = blocks.get('TOSTRAIN', {}).get('data', {})
            if not node_coords or not stress_block or not strain_block:
                raise RuntimeError(f'Missing solver fields in {case_frd}')

            node_values = np.asarray(list(node_coords.values()), dtype=np.float32)
            bounds = np.array([node_values.min(axis=0), node_values.max(axis=0)])
            geom_mask = build_geometry_mask(node_coords, bounds, resolution=512)

            stress_by_node = {node_id: compute_von_mises(values) for node_id, values in stress_block.items()}
            strain_by_node = {node_id: compute_equivalent_strain(values) for node_id, values in strain_block.items()}

            stress_map = rasterize_nodal_field(node_coords, stress_by_node, bounds)
            strain_map = rasterize_nodal_field(node_coords, strain_by_node, bounds)

            stress_scale = float(max(stress_by_node.values()) if stress_by_node else 1.0)
            strain_scale = float(max(strain_by_node.values()) if strain_by_node else 1.0)
            stress_map = np.clip(stress_map * geom_mask, 0.0, 1.0)
            strain_map = np.clip(strain_map * geom_mask, 0.0, 1.0)

            hotspot_nodes = sorted(stress_by_node.items(), key=lambda item: item[1], reverse=True)[:10]
            hotspot_coords = []
            for node_id, value in hotspot_nodes:
                coords = node_coords.get(node_id)
                if coords is not None:
                    x, y, z = coords
                    hotspot_coords.append([float(x), float(y), float(z), float(value)])
            if len(hotspot_coords) > 0:
                hotspot_xy = (hotspot_coords[0][0], hotspot_coords[0][1])
                max_stress = hotspot_coords[0][3]
            else:
                hotspot_xy = (float((bounds[0][0] + bounds[1][0]) / 2), float((bounds[0][1] + bounds[1][1]) / 2))
                max_stress = 0.0
            recommendation = generate_design_recommendation(load_x, load_y, hotspot_xy, bounds, max_stress)
            
            # Map physical load to 512x512 pixel coordinates for the dataset
            width = max((bounds[1][0] - bounds[0][0]), 1e-8)
            height = max((bounds[1][1] - bounds[0][1]), 1e-8)
            load_x_px = ((load_x - bounds[0][0]) / width) * 511.0
            load_y_px = ((load_y - bounds[0][1]) / height) * 511.0

            # Save sample
            sample_file = self.output_dir / f"sample_{i:04d}.npz"
            np.savez(sample_file,
                    geometry=geom_mask,
                    stress=stress_map,
                    strain=strain_map,
                    load_magnitude=float(load_mag),
                    load_x=float(load_x_px), # Pixel coordinates for heatmap
                    load_y=float(load_y_px),
                    load_x_pos=float(load_x), # Physical coordinates
                    load_y_pos=float(load_y),
                    stress_scale=stress_scale,
                    strain_scale=strain_scale,
                    hotspot_points=np.asarray(hotspot_coords, dtype=np.float32))
            
            part_size_mm = {
                "width_x": float(bounds[1][0] - bounds[0][0]),
                "height_y": float(bounds[1][1] - bounds[0][1]),
                "depth_z": float(bounds[1][2] - bounds[0][2])
            }
            metadata.append({
                'sample_id': i,
                'file': str(sample_file),
                'part_size_mm': part_size_mm,
                'load_magnitude': float(load_mag),
                'load_x_px': float(load_x_px),
                'load_y_px': float(load_y_px),
                'load_x_pos': float(load_x),
                'load_y_pos': float(load_y),
                'solver_frd': str(case_frd),
                'fracture_points': hotspot_coords,
                'recommendation': recommendation
            })
            
            logger.info(f"✅ Saved: {sample_file}")
        
        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Generated {len(metadata)} samples in {self.output_dir}")
        logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--stl', required=True, help='Input STL file')
    p.add_argument('--msh', default=None, help='Optional cached Gmsh mesh to reuse')
    p.add_argument('--out', default='data/fea_training_data', help='Output directory')
    p.add_argument('--n', type=int, default=5, help='Number of samples')
    p.add_argument('--start', type=int, default=0, help='Start index for sample generation (append mode)')
    p.add_argument('--gui', action='store_true', help='Show CalculiX GUI window')
    args = p.parse_args()
    
    gen = FEADataGenerator(args.stl, output_dir=args.out, mesh_file=args.msh)
    gen.generate_samples(num_samples=args.n, start_idx=args.start, show_gui=args.gui)
