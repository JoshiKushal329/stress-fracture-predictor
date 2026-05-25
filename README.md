# AI Stress Fracture Predictor

A full-stack AI engineering project for fast structural risk estimation from CAD geometry and load inputs.

The system uses a neural surrogate model to estimate stress distribution, detect fracture hotspots, classify risk, and return design recommendations through a FastAPI backend, with a React frontend for interactive usage.

---

## Project Status

✅ **Working MVP**

- Backend API is implemented and runnable
- Core inference flow is implemented end-to-end
- Material database and health endpoints are available
- Frontend UI source is included (`frontend/src`)

---

## Key Capabilities

- Upload CAD files (`.stl`, `.step`, `.stp`) for analysis
- Predict 2D stress maps (512 × 512)
- Detect and rank stress hotspots
- Classify fracture risk (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- Generate engineering recommendations
- Export prediction results as JSON (frontend)

---

## System Architecture

- **Backend:** FastAPI + PyTorch (`backend/main.py`)
- **Model Runtime:** U-Net-style surrogate loaded from `models/unet_best.pth`
- **Geometry Processing:** `trimesh` projection/rasterization pipeline
- **Shared Scripts:** preprocessing, training, and utility scripts in `scripts/`
- **Frontend:** React-based app in `frontend/src/App.jsx`

---

## Repository Structure

- `backend/` — FastAPI server
- `frontend/` — frontend source files
- `models/` — trained model weights
- `scripts/` — training, preprocessing, prediction, FEA/data tools
- `data/` — datasets/sample data
- `INFERENCE_GUIDE.md` — model inference and interpretation notes

---

## Startup Guide

### 1) Prerequisites

- Python 3.10+ recommended
- `pip`
- (Optional) Node.js 18+ and npm for frontend app setup

---

### 2) Clone and enter project

```bash
git clone <your-repo-url>
cd stress-fracture-predictor
```

---

### 3) Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install fastapi uvicorn torch numpy pillow trimesh python-multipart
```

---

### 4) Add model weights

Place trained model weights at:

```text
models/unet_best.pth
```

> The backend starts even if loading fails, but inference requires the model to be loaded successfully.

---

### 5) Run backend API

```bash
python backend/main.py
```

API will be available at:

- `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

---

### 6) Verify backend health

```bash
curl http://localhost:8000/health
```

You should see `"status": "ok"` and model/device metadata.

---

### 7) Run a prediction request

```bash
curl -X POST "http://localhost:8000/predict?load_x=256&load_y=256&load_magnitude=500" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/your_part.stl"
```

Response includes:

- `stress_map`
- `max_stress_mpa`
- `fracture_risk`
- `all_hotspots`
- `recommendations`

---

## Frontend Usage

The repository currently includes frontend source code (`frontend/src`) but no package manifest (`package.json`) in the current tree.

To run the UI locally, initialize or restore the frontend project config (for example with Vite/React), then point requests to:

- `http://localhost:8000/predict`

---

## API Endpoints (Current)

- `GET /` — API summary
- `GET /health` — runtime health + model info
- `POST /predict` — CAD upload + stress prediction
- `GET /material_db` — available material properties

---

## Known Limitations

- Current U-Net architecture in backend is a simplified placeholder and must match training architecture for production-quality predictions
- Geometry rasterization is projection-based and may not capture full 3D stress behavior
- Accuracy depends on quality and coverage of training data

---

## Next Steps to Productionize

- Version and pin dependencies with `requirements.txt`
- Add automated tests for API and inference paths
- Add frontend build configuration and deployment scripts
- Introduce model/version metadata and model registry checks
- Add CI for lint/test/build and release packaging

---

## Additional Documentation

- See `INFERENCE_GUIDE.md` for model input/output interpretation details.
