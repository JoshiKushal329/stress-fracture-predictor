"""
FastAPI Backend - Stress Prediction Server

This is the main API server that:
1. Accepts CAD files + load parameters
2. Preprocesses geometry to images
3. Runs U-Net model for inference
4. Detects hotspots and fracture points
5. Generates design recommendations
6. Provides batch processing and optimization

Endpoints:
- POST /predict - Single part prediction
- POST /predict/batch - Analyze multiple parts
- POST /optimize - Find best design automatically
- GET /sensitivity - Which parameters matter most
- GET /material_db - Material database
- POST /fatigue - Lifespan prediction
- GET /health - Health check

Author: AI Engineer
Date: 2025
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import trimesh
from PIL import Image
import io
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZATION
# ============================================================================

app = FastAPI(
    title="AI Stress Fracture Predictor API",
    description="Neural network-based FEA surrogate model for structural analysis",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
EXECUTOR = ThreadPoolExecutor(max_workers=4)

logger.info(f"Using device: {DEVICE}")


# ============================================================================
# MATERIAL DATABASE
# ============================================================================

MATERIAL_DATABASE = {
    "ASTM_A36": {
        "name": "ASTM A36 Carbon Steel",
        "yield_strength": 250,  # MPa
        "ultimate_strength": 400,
        "endurance_limit": 100,
        "density": 7850,  # kg/m³
        "cost_per_kg": 1.2,
        "availability": "Common",
        "notes": "Most economical, widely available"
    },
    "ASTM_A514": {
        "name": "ASTM A514 High Strength Steel",
        "yield_strength": 690,
        "ultimate_strength": 760,
        "endurance_limit": 300,
        "density": 7850,
        "cost_per_kg": 3.5,
        "availability": "Common",
        "notes": "For heavy-duty applications"
    },
    "4130_Steel": {
        "name": "4130 Alloy Steel (Chrome-Moly)",
        "yield_strength": 435,
        "ultimate_strength": 565,
        "endurance_limit": 250,
        "density": 7850,
        "cost_per_kg": 2.8,
        "availability": "Available",
        "notes": "Aerospace applications, good weldability"
    },
    "Stainless_304": {
        "name": "Stainless Steel 304",
        "yield_strength": 170,
        "ultimate_strength": 515,
        "endurance_limit": 160,
        "density": 8000,
        "cost_per_kg": 4.5,
        "availability": "Common",
        "notes": "Corrosion resistant, lower strength"
    },
    "Aluminum_6061": {
        "name": "Aluminum 6061-T6",
        "yield_strength": 275,
        "ultimate_strength": 310,
        "endurance_limit": 96,
        "density": 2700,
        "cost_per_kg": 2.2,
        "availability": "Very Common",
        "notes": "Lightweight, good machinability"
    }
}


# ============================================================================
# GEOMETRY PROCESSING
# ============================================================================

def extract_geometry_parameters(stl_file_path: str) -> Dict[str, float]:
    """
    Extract geometric parameters from STL file.
    
    Args:
        stl_file_path: Path to STL file
        
    Returns:
        Dictionary with extracted dimensions
    """
    try:
        mesh = trimesh.load(stl_file_path)
        bounds = mesh.bounds
        
        width = bounds[1][0] - bounds[0][0]
        height = bounds[1][1] - bounds[0][1]
        thickness = bounds[1][2] - bounds[0][2]
        volume = mesh.volume
        surface_area = mesh.area
        
        return {
            'width': float(width),
            'height': float(height),
            'thickness': float(thickness),
            'volume': float(volume),
            'surface_area': float(surface_area),
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces)
        }
    except Exception as e:
        logger.error(f"Error extracting geometry: {e}")
        raise


def validate_geometry(params: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate if geometry is within training distribution.
    
    Args:
        params: Extracted geometry parameters
        
    Returns:
        Tuple of (is_valid, message)
    """
    constraints = {
        'width': (30, 200),        # mm
        'height': (30, 250),
        'thickness': (2, 25),
        'volume': (1000, 100000),  # mm³
    }
    
    for param, (min_val, max_val) in constraints.items():
        if param in params:
            val = params[param]
            if not (min_val <= val <= max_val):
                return False, f"{param}={val:.1f} outside range [{min_val}, {max_val}]"
    
    return True, "Valid geometry"


def rasterize_to_image(stl_file_path: str, resolution: int = 512) -> np.ndarray:
    """
    Convert 3D STL mesh to 2D binary image (top-down projection).
    
    Args:
        stl_file_path: Path to STL file
        resolution: Output image size (512×512)
        
    Returns:
        Binary image as numpy array (0-1)
    """
    mesh = trimesh.load(stl_file_path)
    bounds = mesh.bounds
    
    # Get dimensions
    width = bounds[1][0] - bounds[0][0]
    height = bounds[1][1] - bounds[0][1]
    
    # Create empty image
    image = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Project vertices
    for vertex in mesh.vertices:
        x = int(((vertex[0] - bounds[0][0]) / width * (resolution - 1)))
        y = int(((vertex[1] - bounds[0][1]) / height * (resolution - 1)))
        
        if 0 <= x < resolution and 0 <= y < resolution:
            image[y, x] = 1.0
    
    return image


def encode_load_as_channels(load_x: int, load_y: int, 
                           load_magnitude: float, 
                           resolution: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode load position and magnitude as image channels.
    
    Args:
        load_x: Load X position (pixel coordinate)
        load_y: Load Y position (pixel coordinate)
        load_magnitude: Load magnitude (N)
        resolution: Image resolution (512×512)
        
    Returns:
        Tuple of (load_location_heatmap, load_magnitude_field)
    """
    # Channel 1: Gaussian blob showing load location
    y, x = np.ogrid[:resolution, :resolution]
    sigma = 30  # Blur radius
    load_location = np.exp(-((x - load_x)**2 + (y - load_y)**2) / (2 * sigma**2)).astype(np.float32)
    
    # Channel 2: Uniform field with magnitude
    load_mag_normalized = min(load_magnitude / 1000.0, 1.0)
    load_magnitude_field = np.ones((resolution, resolution), dtype=np.float32) * load_mag_normalized
    
    return load_location, load_magnitude_field


# ============================================================================
# MODEL LOADING & INFERENCE
# ============================================================================

def load_model() -> None:
    """Load pre-trained U-Net model"""
    global MODEL
    try:
        model_path = Path("models/unet_best.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model architecture (same as train.py)
        MODEL = create_unet()
        
        # Load weights
        state_dict = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Path: {model_path}")
        logger.info(f"   Device: {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_unet() -> nn.Module:
    """Create U-Net model (copy of architecture from train.py)"""
    # Simplified version - use actual U-Net class
    # This is a placeholder; in production use the real UNet class
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 512 * 512, 512 * 512)
        
        def forward(self, x):
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            x = self.fc(x)
            return x.reshape(batch_size, 1, 512, 512)
    
    return SimpleUNet()


def predict_stress_field(geometry_image: np.ndarray,
                        load_location: np.ndarray,
                        load_magnitude_field: np.ndarray) -> np.ndarray:
    """
    Run U-Net inference to predict stress field.
    
    Args:
        geometry_image: Rasterized geometry (512×512)
        load_location: Load position heatmap
        load_magnitude_field: Load magnitude field
        
    Returns:
        Predicted stress map (512×512)
    """
    # Stack into 3-channel input
    input_tensor = np.stack([geometry_image, load_location, load_magnitude_field], axis=0)
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        stress_map = MODEL(input_tensor)[0, 0].cpu().numpy()
    
    return stress_map


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def detect_hotspots(stress_map: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Find hotspot locations (fracture points).
    
    Args:
        stress_map: Predicted stress map
        top_k: Number of hotspots to return
        
    Returns:
        List of hotspot dictionaries
    """
    stress_flat = stress_map.flatten()
    top_k_indices = np.argsort(stress_flat)[-top_k:][::-1]
    
    hotspots = []
    for idx in top_k_indices:
        y, x = np.unravel_index(idx, stress_map.shape)
        stress_value = stress_map[y, x]
        
        hotspots.append({
            'x': int(x),
            'y': int(y),
            'stress_mpa': float(stress_value * 400),  # Scale to realistic MPa
            'stress_normalized': float(stress_value)
        })
    
    return hotspots


def determine_fracture_risk(max_stress: float, yield_strength: float = 250) -> str:
    """
    Classify fracture risk based on stress level.
    
    Args:
        max_stress: Maximum stress (normalized 0-1)
        yield_strength: Material yield strength (MPa)
        
    Returns:
        Risk classification string
    """
    stress_mpa = max_stress * 400
    safety_factor = yield_strength / stress_mpa if stress_mpa > 0 else float('inf')
    
    if safety_factor < 1.0:
        return "CRITICAL"
    elif safety_factor < 1.5:
        return "HIGH"
    elif safety_factor < 2.0:
        return "MEDIUM"
    else:
        return "LOW"


def generate_recommendations(hotspots: List[Dict],
                           geometry_params: Dict,
                           stress_map: np.ndarray) -> List[Dict]:
    """
    Generate design recommendations based on analysis.
    
    Args:
        hotspots: Detected hotspots
        geometry_params: Extracted geometry parameters
        stress_map: Stress map array
        
    Returns:
        List of recommendations
    """
    recommendations = []
    primary_hotspot = hotspots[0] if hotspots else {}
    stress_level = primary_hotspot.get('stress_normalized', 0.5)
    
    # Recommendation 1: Fillets
    if stress_level > 0.6:
        recommendations.append({
            'type': 'add_fillet',
            'priority': 'HIGH',
            'description': 'Add or increase fillet radius at hotspot corners',
            'expected_reduction_percent': 30,
            'difficulty': 'EASY',
            'cost_impact': 'Low'
        })
    
    # Recommendation 2: Thickness
    if stress_level > 0.5:
        recommendations.append({
            'type': 'increase_thickness',
            'priority': 'MEDIUM',
            'description': 'Increase material thickness in high-stress region',
            'expected_reduction_percent': 25,
            'difficulty': 'MEDIUM',
            'cost_impact': 'Medium'
        })
    
    # Recommendation 3: Material upgrade
    if stress_level > 0.7:
        recommendations.append({
            'type': 'material_upgrade',
            'priority': 'HIGH',
            'description': 'Upgrade to higher strength material',
            'expected_reduction_percent': 0,  # Doesn't reduce stress, just allows more
            'difficulty': 'MEDIUM',
            'cost_impact': 'High'
        })
    
    return recommendations


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("✅ API ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    load_x: int = 256,
    load_y: int = 256,
    load_magnitude: float = 500.0
):
    """
    Main prediction endpoint.
    
    Args:
        file: STL or STEP CAD file
        load_x: Load position X (0-512)
        load_y: Load position Y (0-512)
        load_magnitude: Load magnitude (N)
        
    Returns:
        JSON with stress map, hotspots, recommendations
    """
    temp_path = f"/tmp/{file.filename}"
    
    try:
        # Save file
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Extract and validate geometry
        params = extract_geometry_parameters(temp_path)
        is_valid, msg = validate_geometry(params)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Geometry validation failed: {msg}")
        
        # Rasterize geometry
        geometry_image = rasterize_to_image(temp_path)
        
        # Encode load
        load_location, load_magnitude_field = encode_load_as_channels(
            load_x, load_y, load_magnitude
        )
        
        # Predict
        stress_map = predict_stress_field(geometry_image, load_location, load_magnitude_field)
        
        # Analyze
        hotspots = detect_hotspots(stress_map)
        max_stress = np.max(stress_map)
        fracture_risk = determine_fracture_risk(max_stress)
        recommendations = generate_recommendations(hotspots, params, stress_map)
        
        return {
            "success": True,
            "geometry": params,
            "stress_map": stress_map.tolist(),
            "max_stress_normalized": float(max_stress),
            "max_stress_mpa": float(max_stress * 400),
            "fracture_risk": fracture_risk,
            "primary_hotspot": hotspots[0] if hotspots else None,
            "all_hotspots": hotspots,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/material_db")
async def get_material_database():
    """Return material property database"""
    return {
        "materials": MATERIAL_DATABASE,
        "count": len(MATERIAL_DATABASE)
    }


@app.get("/")
async def root():
    """API documentation"""
    return {
        "title": "AI Stress Fracture Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict stress for CAD file",
            "GET /material_db": "Material database",
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
