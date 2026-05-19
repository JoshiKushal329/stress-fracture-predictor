# Inference Guide: Predicting Stress/Strain on Galvanized Iron (GI)

## Material Specification
**Material:** Galvanized Iron (GI)  
**Young's Modulus (E):** 200,000 MPa (200 GPa)  
**Poisson's Ratio (ν):** 0.27  
**Density:** 7,850 kg/m³  
**Yield Strength:** ~250–350 MPa (varies by coating thickness)  
**Typical Use:** Corrosion-resistant structural parts, supports, brackets

---

## How to Provide Input to the Trained Model

Once the UNet model is trained, you provide:

### **1. Geometry Input**
- **Source:** Your CAD model (`.stp` or `.step` file)
- **Format:** 512×512 binary mask (1 = part, 0 = background)
- **How it's generated:**
  ```python
  from PIL import Image
  import numpy as np
  
  # Load your part geometry (STL converted from STP)
  from preprocess import build_geometry_mask
  
  geometry_mask = build_geometry_mask(stl_file, resolution=512)
  # Returns: 512×512 numpy array
  ```

### **2. Load Input**
Specify the applied force in 3 components:
```python
load_magnitude = 618.18  # Force in Newtons (N)
load_x = -19.94         # X-component of load direction (N)
load_y = -97.53         # Y-component of load direction (N)
# Internally computed: load_z = sqrt(mag^2 - lx^2 - ly^2)
```

**Example Loads:**
- Downward force (gravity): `load_z = -1000 N`, `load_x = 0`, `load_y = 0`
- Diagonal shear: `load_x = 500 N`, `load_y = 500 N`, `load_z = -500 N`
- Torsion-like: rotational loads on supported nodes

### **3. Combined Input to Model**

```python
import torch
from scripts.train import UNet
import numpy as np

# Load trained model
model = UNet(in_channels=3, out_channels=2)
model.load_state_dict(torch.load('models/unet_trained.pth'))
model.eval()

# Prepare input: 3-channel image
# Channel 0: geometry mask (512, 512)
# Channel 1: load_x direction map (512, 512)
# Channel 2: load_y direction map (512, 512)

input_tensor = torch.stack([
    torch.from_numpy(geometry_mask).float(),
    torch.from_numpy(load_x_map).float(),
    torch.from_numpy(load_y_map).float()
], dim=0).unsqueeze(0)  # Shape: (1, 3, 512, 512)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    # output shape: (1, 2, 512, 512)
    # output[0, 0] = predicted stress field
    # output[0, 1] = predicted strain field

predicted_stress = output[0, 0].numpy()  # (512, 512)
predicted_strain = output[0, 1].numpy()  # (512, 512)
```

---

## Output Interpretation

### **Stress Map** (512×512)
- **Pixel value** = von Mises stress (MPa) at that location
- **High values** = risk zones (compare to GI yield ~300 MPa)
- **Hotspots:** Pixels > 250 MPa indicate potential fracture points

### **Strain Map** (512×512)
- **Pixel value** = equivalent strain (dimensionless)
- Helps identify plastic deformation regions

### **Design Recommendations**
```python
# Extract hotspots
hotspot_threshold = 250  # MPa for GI yield
hotspots = np.argwhere(predicted_stress > hotspot_threshold)

# Recommend reinforcement at high-stress regions
for y, x in hotspots:
    print(f"Reinforce region at ({x}, {y}): stress = {predicted_stress[y, x]:.1f} MPa")
```

---

## Example Workflow

```bash
# 1. Have your STL/STP file ready (vertical_support.stp)
# 2. Define load case
load_mag = 500  # N
load_x = 0
load_y = 0
load_z = -500

# 3. Convert to geometry mask
geometry_mask = build_geometry_mask('vertical_support.stl', 512)

# 4. Run inference
predicted_stress, predicted_strain = model_inference(
    geometry_mask, load_x, load_y, load_mag
)

# 5. Find fracture points
high_stress_points = predict_fracture_points(predicted_stress, threshold=300)

# 6. Export recommendations
export_design_recommendations(
    geometry_mask, 
    predicted_stress, 
    high_stress_points, 
    material='Galvanized Iron'
)
```

---

## GI-Specific Considerations

| Property | Value | Impact |
|----------|-------|--------|
| Young's Modulus | 200 GPa | Lower stiffness than stainless (210 GPa) |
| Yield Strength | 250–350 MPa | Moderate strength; monitor stress < 250 MPa |
| Fatigue Limit | ~150 MPa | Repeated loads reduce safe working stress |
| Corrosion Resistance | Excellent | Zinc coating extends life; don't over-stress |
| Cost | Low | Economic choice; fracture = product failure risk |

**Recommendation:** Train with GI-specific data; model learns where GI fails first under your load patterns.

---

## Next Steps

1. Generate 20–50 more FEA samples with varied loads
2. Train UNet on full solver dataset
3. Export model to ONNX for deployment
4. Build UI/API for quick stress prediction on design variants
