"""
Configuration and Constants for Stress Predictor
This file centralizes all normalization and scaling parameters.
"""

# ============================================================================
# STRESS NORMALIZATION PARAMETERS
# ============================================================================

# Material properties (Galvanized Iron / Steel)
MATERIAL_YIELD_STRENGTH = 300.0  # MPa - GI yield strength
MATERIAL_ULTIMATE_STRENGTH = 400.0  # MPa - typical GI ultimate

# Training data statistics
# These should be determined from your actual training data
# If using synthetic data: capture actual min/max from generate_data.py
TRAINING_MAX_STRESS = 500.0  # MPa - maximum stress seen in training data
TRAINING_MIN_STRESS = 0.0  # MPa - minimum stress (usually 0)

# Normalization range
# During training, stress values are normalized to [0, 1] range
# Denormalization formula: actual_stress_mpa = normalized_stress * STRESS_SCALE_FACTOR
STRESS_SCALE_FACTOR = TRAINING_MAX_STRESS - TRAINING_MIN_STRESS  # = 500.0 MPa

# Strain normalization
TRAINING_MAX_STRAIN = 0.01  # unitless - typical max strain in GI
STRAIN_SCALE_FACTOR = TRAINING_MAX_STRAIN  # = 0.01

# ============================================================================
# MODEL INPUT PARAMETERS
# ============================================================================

# Model architecture
INPUT_CHANNELS = 3  # geometry + load_x + load_y
OUTPUT_CHANNELS = 2  # stress + strain
BASE_CHANNELS = 16  # U-Net base channels
RESOLUTION = 512  # Input/output image resolution

# Load encoding parameters
LOAD_GAUSSIAN_SIGMA = 8.0  # pixels - concentration of load point
LOAD_MAGNITUDE_MAX = 1000.0  # N - maximum load for normalization in backend

# ============================================================================
# GEOMETRY PROCESSING
# ============================================================================

GEOMETRY_MIN_WIDTH = 30  # mm
GEOMETRY_MAX_WIDTH = 200  # mm
GEOMETRY_MIN_HEIGHT = 30  # mm
GEOMETRY_MAX_HEIGHT = 250  # mm
GEOMETRY_MIN_THICKNESS = 2  # mm
GEOMETRY_MAX_THICKNESS = 25  # mm

# ============================================================================
# HOTSPOT DETECTION
# ============================================================================

HOTSPOT_COUNT = 10  # Number of hotspots to report
HOTSPOT_THRESHOLD = 300.0  # MPa - minimum stress for hotspot

# ============================================================================
# TRAINING PARAMETERS (for reference)
# ============================================================================

DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_stress(stress_mpa: float) -> float:
    """Convert actual stress (MPa) to normalized value (0-1)"""
    if STRESS_SCALE_FACTOR == 0:
        return 0.0
    return stress_mpa / STRESS_SCALE_FACTOR

def denormalize_stress(normalized_stress: float) -> float:
    """Convert normalized stress (0-1) to actual MPa"""
    return normalized_stress * STRESS_SCALE_FACTOR

def normalize_strain(strain: float) -> float:
    """Convert actual strain to normalized value (0-1)"""
    if STRAIN_SCALE_FACTOR == 0:
        return 0.0
    return strain / STRAIN_SCALE_FACTOR

def denormalize_strain(normalized_strain: float) -> float:
    """Convert normalized strain (0-1) to actual strain"""
    return normalized_strain * STRAIN_SCALE_FACTOR

# ============================================================================
# METADATA FILE
# ============================================================================

# Save this information to a JSON file during model training
# This allows loading the correct scale factors from saved models

TRAINING_METADATA_TEMPLATE = {
    "version": "1.0",
    "timestamp": None,  # Set when training
    "stress_scale_factor": STRESS_SCALE_FACTOR,
    "strain_scale_factor": STRAIN_SCALE_FACTOR,
    "training_max_stress_mpa": TRAINING_MAX_STRESS,
    "training_max_strain": TRAINING_MAX_STRAIN,
    "material_yield_strength_mpa": MATERIAL_YIELD_STRENGTH,
    "model_architecture": {
        "input_channels": INPUT_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "resolution": RESOLUTION
    }
}
