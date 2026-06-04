# 🔬 Skin Scanner — Backend API

> **AI-powered skin analysis in a single API call.** Upload a face photo, get structured skin metrics back — acne, wrinkles, redness, pores, and texture, all scored 0–1.

---

## What It Does

The Skin Scanner backend accepts a facial image, automatically detects and crops the face, runs it through an EfficientNet-based deep learning model, and returns clean JSON skin scores — ready to display in a mobile app.

**The full pipeline in one request:**

```
📱 Mobile App  →  POST /predict-skin  →  Face Detection  →  Model Inference  →  📊 JSON Scores
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI Server | Uvicorn |
| Deep Learning | PyTorch + Torchvision (EfficientNet-B0) |
| Face Detection | MediaPipe |
| Image Processing | OpenCV + NumPy |
| Validation | Pydantic |

---

## Project Structure

```
backend/
│
├── api/
│   └── routes_skin.py       # POST /predict-skin endpoint
│
├── skin_model/
│   ├── model.py             # EfficientNet-B0 architecture (5 skin outputs)
│   ├── inference.py         # Model loading, preprocessing, forward pass
│   └── weights/             # Trained model weights (empty at MVP stage)
│
├── utils/                   # Shared image preprocessing helpers
├── face3d/                  # 3D face generation (out of MVP scope)
│
├── config.py                # Upload limits, device config, global constants
├── main.py                  # App entry point — registers routes & middleware
├── requirements.txt
└── .gitignore
```

---

## API Reference

### `POST /predict-skin`

Accepts a facial image and returns structured skin scores.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG or PNG, size-limited |

**Response** — `application/json`

```json
{
  "acne":    0.23,
  "wrinkle": 0.11,
  "redness": 0.45,
  "pore":    0.67,
  "texture": 0.38
}
```

All scores are normalized between `0.0` (clear) and `1.0` (high severity).

**What happens under the hood:**
1. File type and size are validated
2. Image is decoded with OpenCV
3. MediaPipe detects and crops the largest face
4. Cropped face is preprocessed into a tensor
5. EfficientNet-B0 runs the forward pass
6. Sigmoid-activated outputs are returned as JSON

---

## File-by-File Breakdown

### `main.py`
Entry point. Initializes FastAPI, registers the `/predict-skin` route, and configures middleware (CORS, etc.). Run with:
```bash
uvicorn main:app
```

### `api/routes_skin.py`
Defines the `/predict-skin` endpoint. Handles validation, decoding, face detection, and wires everything together. Keeps route logic clean — heavy lifting is delegated to `inference.py`.

### `skin_model/model.py`
The neural network. EfficientNet-B0 with a custom 5-neuron classifier head and sigmoid activation. Outputs one score per skin metric.

### `skin_model/inference.py`
Handles the model lifecycle cleanly:
- **Singleton loading** — model loads once at startup, not per request
- **Preprocessing** — NumPy array → normalized Torch tensor
- **Forward pass** — returns a formatted output dictionary

### `skin_model/weights/`
Where trained `.pth` weights live. Currently empty — the model architecture is complete but weights haven't been fine-tuned on a dedicated skin dataset yet.

### `utils/`
Shared helpers for image preprocessing and any reusable logic across the backend.

### `config.py`
Single source of truth for configuration: max upload size, CPU/GPU device selection, and any global constants. No magic numbers scattered through the codebase.

### `face3d/`
Originally scoped for 3D face generation. Not needed for the current MVP — can be revisited if 3D output becomes a product requirement.

### `requirements.txt`
Full dependency list. Recreate the environment exactly with:
```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Clone & set up the environment

```bash
git clone <your-repo-url>
cd backend

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

### 4. Explore the API

Open Swagger UI in your browser:

```
http://127.0.0.1:8000/docs
```

---

## Current Status

| Feature | Status |
|---|---|
| API endpoint | ✅ Complete |
| Face detection (MediaPipe) | ✅ Complete |
| EfficientNet-B0 architecture | ✅ Complete |
| Model weights (fine-tuned) | ⏳ In progress |
| React Native integration | 🔗 Ready to connect |
| 3D face generation (`face3d/`) | 🔜 Post-MVP (later) |

---

## Integration

This backend is designed to pair with a **React Native** mobile app. Point the app's image upload to `POST /predict-skin`, and render the returned scores however you like — progress bars, color-coded overlays, trend charts, etc.
