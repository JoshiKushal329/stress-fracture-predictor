#!/usr/bin/env python3
"""Convert STEP/STP to STL using FreeCAD (command-line).

Usage:
  freecadcmd scripts/convert_step_to_stl.py input.step output.stl
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_step_to_stl.py input.step [output.stl]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix('.stl')

    try:
        import FreeCAD
        import Part
        import Mesh
    except Exception as e:
        print("FreeCAD python modules not available. Run this with freecadcmd:")
        print("  freecadcmd scripts/convert_step_to_stl.py input.step output.stl")
        raise

    shape = Part.Shape()
    shape.read(str(in_path))
    solids = shape.Solids if hasattr(shape, 'Solids') else []
    comp = solids[0] if solids else shape
    mesh = Mesh.Mesh()
    mesh.addShape(comp)
    mesh.write(str(out_path))
    print("Exported:", out_path)

if __name__ == '__main__':
    main()
