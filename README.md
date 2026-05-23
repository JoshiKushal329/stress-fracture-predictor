  # Stress Fracture Predictor

An end-to-end prototype for **predicting stress hotspots / fracture risk** from a CAD part + load inputs using a neural network surrogate model (U-Net style), exposed via a **FastAPI backend** and intended to be used from a **frontend UI**.

This project is structured like a small full-stack ML app:
- **Backend (Python / FastAPI):** accepts CAD files + load parameters, runs inference, returns stress map + hotspot analysis + recommendations
- **Models:** saved model weights (expected by the backend)
- **Frontend (JavaScript):** UI code (in `frontend/`)
- **Data / scripts / prediction folders:** training/inference utilities and outputs

> For detailed inference input/output format and interpretation, see: `INFERENCE_GUIDE.md`.

---

## Repository Structure

- `backend/` – FastAPI server (`backend/main.py`)
- `frontend/` – frontend app source (currently contains `src/`)
- `models/` – model weights (backend expects a file like `models/unet_best.pth`)
- `scripts/` – shared config/training/inference utilities (imported by backend)
- `data/` – datasets / samples
- `prediction_1/`, `prediction_2/`, `prediction_output/`, `custom_predictions/`, `test_folder/` – experiments and outputs
- `INFERENCE_GUIDE.md` – step-by-step inference guide and interpretation notes

---

## What the Backend Does

The backend (`backend/main.py`) exposes a **Stress Fracture Predictor API** that:

1. Accepts a CAD file upload (STL/STEP) + load parameters
2. Extracts basic geometry parameters and validates they’re in expected ranges
3. Rasterizes the geometry into a **512×512** representation
4. Encodes the load into input channels
5. Runs model inference to predict a **stress map**
6. Detects **hotspots** (top high-stress points)
7. Computes an approximate **fracture risk** classification
8. Returns recommendations (e.g., add fillet, increase thickness, material upgrade)

---

## API Endpoints (Backend)

From the header comment and implementation in `backend/main.py`:

- `GET /health` – health check and model/device info  
- `POST /predict` – upload a part + load parameters and get stress + hotspots  
- `GET /material_db` – material property database  
- `GET /` – quick API overview (and points to `/docs`)

FastAPI interactive docs:
- `/docs`

---

## Quickstart (Backend)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 2) Install dependencies

This repo may not include a pinned `requirements.txt` yet. At minimum the backend uses:
- `fastapi`, `uvicorn`
- `torch`
- `numpy`, `Pillow`
- `trimesh`

Example install:

```bash
pip install fastapi uvicorn torch numpy pillow trimesh
```

### 3) Ensure model weights exist

Backend startup tries to load:

- `models/unet_best.pth`

Make sure that file exists, otherwise the API will fail to fully initialize for inference.

### 4) Run the backend

```bash
python backend/main.py
```

By default it runs on:
- `http://0.0.0.0:8000`
- Docs: `http://localhost:8000/docs`

---

## Example Request (Predict)

`POST /predict` accepts:
- `file`: CAD file upload
- `load_x` (default `256`)
- `load_y` (default `256`)
- `load_magnitude` (default `500.0` Newtons)

Example with `curl`:

```bash
curl -X POST "http://localhost:8000/predict?load_x=256&load_y=256&load_magnitude=500" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_part.stl"
```

Response includes:
- `stress_map` (512×512 array)
- `max_stress_mpa`
- `fracture_risk`
- `all_hotspots`
- `recommendations`

> Note: The backend is designed to return **stress in MPa** (denormalized).

---

## Frontend

Frontend code lives in `frontend/` (currently shows `frontend/src`). If you have (or plan to add) a `package.json`, typical commands will look like:

```bash
cd frontend
npm install
npm run dev
```

If you want, tell me what framework you used (React/Vite/Next/etc.) and I’ll tailor the exact frontend run instructions.

---

## Notes / Caveats

- The current `backend/main.py` includes a `create_unet()` placeholder implementation. For real results, you’ll want to ensure the architecture matches the weights saved in `models/unet_best.pth`.
- Geometry rasterization is currently a simple projection-based approach; accuracy depends heavily on consistent preprocessing/training.

---

## License

Add a license if you plan to share or reuse this project (e.g., MIT).

---

## Contributing

Issues and PRs welcome. If you open a PR, please include:
- what changed
- how to run/test it locally
- example input + expected output (screenshots or JSON snippets if possible)
