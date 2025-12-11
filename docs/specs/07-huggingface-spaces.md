# Spec 07: Hugging Face Spaces Deployment

> **Phase 7 of 7** — Web deployment on Hugging Face Spaces
>
> **Goal:** Deploy MeshNet inference demo on Hugging Face Spaces WITHOUT Gradio.

---

## Overview

This spec covers:
- Hugging Face Spaces deployment options (non-Gradio)
- Static HTML + JavaScript approach with ONNX Runtime Web
- Docker-based deployment for custom backends
- Model hosting on Hugging Face Hub
- User interface design for MRI upload and visualization

---

## 1. Deployment Strategy

### 1.1 Why NOT Gradio

Per user requirements, Gradio is excluded. Alternative approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **Static HTML + ONNX.js** | Simple, fast, client-side | Limited UI, no server-side processing |
| **Docker + FastAPI** | Full control, server-side inference | More complex, requires GPU quota |
| **Streamlit** | Python-native, decent UI | Still a framework dependency |
| **Static + Pyodide** | Python in browser | Large bundle, slow startup |

**Recommended: Docker + FastAPI with vanilla HTML/JS frontend**

This gives full control, avoids Gradio, and works well with ONNX Runtime.

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HUGGING FACE SPACES                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│   Static Frontend    │     │   FastAPI Backend    │     │   HF Hub     │
│   (HTML/JS/CSS)      │ ──▶ │   (ONNX Runtime)     │ ◀── │   (Models)   │
│                      │     │                      │     │              │
│  - File upload       │     │  - /predict endpoint │     │  - ONNX      │
│  - NiiVue viewer     │     │  - Preprocessing     │     │  - Weights   │
│  - Results display   │     │  - Inference         │     │              │
└──────────────────────┘     └──────────────────────┘     └──────────────┘
```

---

## 2. Project Structure for Spaces

```
arc-meshchop-space/
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── app/
│   ├── main.py               # FastAPI application
│   ├── inference.py          # ONNX inference logic
│   └── preprocessing.py      # NIfTI preprocessing
├── static/
│   ├── index.html            # Main page
│   ├── style.css             # Styles
│   └── app.js                # Frontend logic
├── models/
│   └── .gitkeep              # Models loaded from HF Hub
└── README.md                  # Space documentation
```

---

## 3. Implementation

### 3.1 Dockerfile

**File:** `Dockerfile`

```dockerfile
# Hugging Face Spaces Docker template
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY static/ ./static/

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Expose port (Spaces uses 7860 by default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 3.2 Requirements

**File:** `requirements.txt`

```text
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
onnxruntime>=1.16.0
nibabel>=5.0.0
numpy>=1.24.0
scipy>=1.10.0
huggingface-hub>=0.32.0
```

### 3.3 FastAPI Backend

**File:** `app/main.py`

```python
"""FastAPI backend for MeshNet stroke lesion segmentation.

Provides REST API for MRI upload and segmentation inference.
Models are loaded from Hugging Face Hub.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

from app.inference import ONNXInferenceEngine
from app.preprocessing import preprocess_nifti, postprocess_segmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MeshNet Stroke Lesion Segmentation",
    description="State-of-the-art stroke lesion segmentation at 1/1000th of parameters",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global inference engine (loaded on startup)
engine: ONNXInferenceEngine | None = None


@app.on_event("startup")
async def load_model() -> None:
    """Load model on startup from Hugging Face Hub."""
    global engine

    logger.info("Loading model from Hugging Face Hub...")

    try:
        # Download model from Hub
        model_path = hf_hub_download(
            repo_id="username/arc-meshchop-models",  # Update with actual repo
            filename="meshnet_16.onnx",
            cache_dir="/tmp/models",
        )

        engine = ONNXInferenceEngine(model_path)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


@app.get("/")
async def index() -> FileResponse:
    """Serve main page."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": engine is not None,
    })


@app.get("/models")
async def list_models() -> JSONResponse:
    """List available models."""
    return JSONResponse({
        "models": [
            {
                "name": "MeshNet-5",
                "params": "5,682",
                "dice": "0.848",
                "description": "Ultra-compact model",
            },
            {
                "name": "MeshNet-16",
                "params": "56,194",
                "dice": "0.873",
                "description": "Balanced model (recommended)",
            },
            {
                "name": "MeshNet-26",
                "params": "147,474",
                "dice": "0.876",
                "description": "Best performance",
            },
        ],
        "current": "MeshNet-16",
    })


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="NIfTI file (.nii or .nii.gz)")],
) -> JSONResponse:
    """Run segmentation inference on uploaded NIfTI file.

    Args:
        file: Uploaded NIfTI file.

    Returns:
        JSON response with segmentation results and download URL.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a NIfTI file (.nii or .nii.gz)",
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        logger.info("Processing file: %s", file.filename)

        # Preprocess
        image_data, affine, original_shape = preprocess_nifti(tmp_path)

        # Run inference
        segmentation = engine.predict(image_data)

        # Postprocess
        output_path, lesion_volume = postprocess_segmentation(
            segmentation,
            affine,
            original_shape,
            output_dir=Path("/tmp/outputs"),
        )

        # Calculate statistics
        total_voxels = int(np.prod(segmentation.shape))
        lesion_voxels = int(np.sum(segmentation > 0.5))
        lesion_percentage = (lesion_voxels / total_voxels) * 100

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "results": {
                "lesion_volume_voxels": lesion_voxels,
                "lesion_volume_ml": lesion_voxels / 1000,  # Assuming 1mm³ voxels
                "lesion_percentage": round(lesion_percentage, 2),
                "total_voxels": total_voxels,
            },
            "download_url": f"/download/{output_path.name}",
        })

    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/download/{filename}")
async def download(filename: str) -> FileResponse:
    """Download segmentation result.

    Args:
        filename: Name of output file.

    Returns:
        NIfTI file for download.
    """
    file_path = Path("/tmp/outputs") / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/gzip",
    )
```

### 3.4 Inference Engine

**File:** `app/inference.py`

```python
"""ONNX Runtime inference engine for MeshNet."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for MeshNet segmentation."""

    def __init__(self, model_path: Path | str) -> None:
        """Initialize inference engine.

        Args:
            model_path: Path to ONNX model file.
        """
        self.model_path = Path(model_path)

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create inference session
        # Use CPU provider for HF Spaces (GPU requires quota)
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info("Loaded ONNX model: %s", self.model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image.

        Args:
            image: Preprocessed image array of shape (1, 1, D, H, W).

        Returns:
            Segmentation mask of shape (D, H, W).
        """
        # Ensure correct shape and type
        if image.ndim == 3:
            image = image[np.newaxis, np.newaxis, ...]  # Add batch and channel dims
        elif image.ndim == 4:
            image = image[np.newaxis, ...]  # Add batch dim

        image = image.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image},
        )

        # Get predictions (argmax over class dimension)
        logits = outputs[0]  # Shape: (1, 2, D, H, W)
        segmentation = np.argmax(logits, axis=1)[0]  # Shape: (D, H, W)

        return segmentation
```

### 3.5 Preprocessing

**File:** `app/preprocessing.py`

```python
"""Preprocessing and postprocessing for NIfTI files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def preprocess_nifti(
    file_path: Path,
    target_shape: tuple[int, int, int] = (256, 256, 256),
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Preprocess NIfTI file for inference.

    Args:
        file_path: Path to NIfTI file.
        target_shape: Target volume shape.
        target_spacing: Target voxel spacing in mm.

    Returns:
        Tuple of (preprocessed_image, affine, original_shape).
    """
    # Load NIfTI
    nii = nib.load(str(file_path))
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine
    original_shape = data.shape

    # Get current spacing
    current_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    # Resample to target spacing
    zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    resampled = ndimage.zoom(data, zoom_factors, order=1)

    # Crop or pad to target shape
    result = np.zeros(target_shape, dtype=np.float32)

    # Calculate overlap region
    src_start = [max(0, (s - t) // 2) for s, t in zip(resampled.shape, target_shape)]
    src_end = [min(s, src_start[i] + target_shape[i]) for i, s in enumerate(resampled.shape)]

    dst_start = [max(0, (t - s) // 2) for s, t in zip(resampled.shape, target_shape)]
    dst_end = [dst_start[i] + (src_end[i] - src_start[i]) for i in range(3)]

    result[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = resampled[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    # Normalize to [0, 1]
    data_min = result.min()
    data_max = result.max()
    if data_max - data_min > 1e-8:
        result = (result - data_min) / (data_max - data_min)

    return result, affine, original_shape


def postprocess_segmentation(
    segmentation: np.ndarray,
    affine: np.ndarray,
    original_shape: tuple[int, int, int],
    output_dir: Path,
) -> Tuple[Path, int]:
    """Postprocess segmentation and save to NIfTI.

    Args:
        segmentation: Segmentation mask array.
        affine: Original affine transformation.
        original_shape: Original image shape.
        output_dir: Directory for output file.

    Returns:
        Tuple of (output_path, lesion_volume_voxels).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resample back to original shape if needed
    if segmentation.shape != original_shape:
        zoom_factors = [o / s for o, s in zip(original_shape, segmentation.shape)]
        segmentation = ndimage.zoom(segmentation, zoom_factors, order=0)

    # Calculate lesion volume
    lesion_volume = int(np.sum(segmentation > 0.5))

    # Save as NIfTI
    output_nii = nib.Nifti1Image(segmentation.astype(np.uint8), affine)
    output_path = output_dir / "segmentation.nii.gz"
    nib.save(output_nii, str(output_path))

    return output_path, lesion_volume
```

### 3.6 Frontend HTML

**File:** `static/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MeshNet Stroke Lesion Segmentation</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>MeshNet Stroke Lesion Segmentation</h1>
            <p class="subtitle">State-of-the-art segmentation at 1/1000th of parameters</p>
        </header>

        <main>
            <!-- Upload Section -->
            <section class="upload-section">
                <h2>Upload MRI Scan</h2>
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="fileInput" accept=".nii,.nii.gz" hidden>
                    <div class="upload-content">
                        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17,8 12,3 7,8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                        <p>Drag & drop a NIfTI file here</p>
                        <p class="upload-hint">or click to browse (.nii, .nii.gz)</p>
                    </div>
                </div>
                <p id="fileName" class="file-name"></p>
            </section>

            <!-- Processing Section -->
            <section class="processing-section" id="processingSection" hidden>
                <div class="spinner"></div>
                <p>Processing... This may take a minute.</p>
            </section>

            <!-- Results Section -->
            <section class="results-section" id="resultsSection" hidden>
                <h2>Results</h2>
                <div class="results-grid">
                    <div class="result-card">
                        <h3>Lesion Volume</h3>
                        <p class="result-value" id="lesionVolume">--</p>
                        <p class="result-unit">voxels</p>
                    </div>
                    <div class="result-card">
                        <h3>Volume (mL)</h3>
                        <p class="result-value" id="lesionVolumeML">--</p>
                        <p class="result-unit">milliliters</p>
                    </div>
                    <div class="result-card">
                        <h3>Brain Coverage</h3>
                        <p class="result-value" id="lesionPercentage">--</p>
                        <p class="result-unit">percent</p>
                    </div>
                </div>
                <div class="download-section">
                    <a id="downloadLink" class="download-button" download>
                        Download Segmentation Mask
                    </a>
                </div>
            </section>

            <!-- Error Section -->
            <section class="error-section" id="errorSection" hidden>
                <h2>Error</h2>
                <p id="errorMessage"></p>
                <button onclick="location.reload()">Try Again</button>
            </section>
        </main>

        <footer>
            <p>Model: MeshNet-16 (56K parameters, 0.873 DICE)</p>
            <p>Based on: <a href="#">Fedorov et al. "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"</a></p>
        </footer>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

### 3.7 Frontend CSS

**File:** `static/style.css`

```css
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --success: #10b981;
    --error: #ef4444;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-600: #4b5563;
    --gray-800: #1f2937;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--gray-100);
    color: var(--gray-800);
    line-height: 1.6;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    text-align: center;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: var(--gray-600);
}

/* Upload Area */
.upload-area {
    border: 2px dashed var(--gray-200);
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: white;
}

.upload-area:hover,
.upload-area.drag-over {
    border-color: var(--primary);
    background: rgba(37, 99, 235, 0.05);
}

.upload-icon {
    width: 48px;
    height: 48px;
    margin-bottom: 1rem;
    color: var(--gray-600);
}

.upload-hint {
    color: var(--gray-600);
    font-size: 0.875rem;
}

.file-name {
    margin-top: 1rem;
    font-weight: 500;
    color: var(--primary);
}

/* Processing */
.processing-section {
    text-align: center;
    padding: 3rem;
}

.spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--gray-200);
    border-top-color: var(--primary);
    border-radius: 50%;
    margin: 0 auto 1rem;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Results */
.results-section {
    margin-top: 2rem;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.result-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.result-card h3 {
    font-size: 0.875rem;
    color: var(--gray-600);
    margin-bottom: 0.5rem;
}

.result-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.result-unit {
    font-size: 0.75rem;
    color: var(--gray-600);
}

/* Download Button */
.download-section {
    text-align: center;
    margin-top: 1.5rem;
}

.download-button {
    display: inline-block;
    background: var(--success);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 500;
    transition: background 0.2s;
}

.download-button:hover {
    background: #059669;
}

/* Error */
.error-section {
    text-align: center;
    padding: 2rem;
    background: #fef2f2;
    border-radius: 12px;
    margin-top: 2rem;
}

.error-section h2 {
    color: var(--error);
}

.error-section button {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    background: var(--error);
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
}

/* Footer */
footer {
    margin-top: 3rem;
    text-align: center;
    color: var(--gray-600);
    font-size: 0.875rem;
}

footer a {
    color: var(--primary);
}

/* Responsive */
@media (max-width: 640px) {
    .results-grid {
        grid-template-columns: 1fr;
    }
}
```

### 3.8 Frontend JavaScript

**File:** `static/app.js`

```javascript
// MeshNet Stroke Lesion Segmentation - Frontend Application

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Upload area click handler
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file selection
async function handleFile(file) {
    // Validate file type
    if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
        showError('Invalid file type. Please upload a NIfTI file (.nii or .nii.gz)');
        return;
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
        showError('File too large. Maximum size is 100MB.');
        return;
    }

    // Show file name
    fileName.textContent = `Selected: ${file.name}`;

    // Show processing
    uploadArea.hidden = true;
    processingSection.hidden = false;
    resultsSection.hidden = true;
    errorSection.hidden = true;

    try {
        const results = await uploadAndProcess(file);
        showResults(results);
    } catch (error) {
        showError(error.message);
    }
}

// Upload file and run inference
async function uploadAndProcess(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Processing failed');
    }

    return response.json();
}

// Display results
function showResults(data) {
    processingSection.hidden = true;
    resultsSection.hidden = false;

    document.getElementById('lesionVolume').textContent =
        data.results.lesion_volume_voxels.toLocaleString();
    document.getElementById('lesionVolumeML').textContent =
        data.results.lesion_volume_ml.toFixed(2);
    document.getElementById('lesionPercentage').textContent =
        data.results.lesion_percentage + '%';

    const downloadLink = document.getElementById('downloadLink');
    downloadLink.href = data.download_url;
}

// Display error
function showError(message) {
    processingSection.hidden = true;
    uploadArea.hidden = false;
    errorSection.hidden = false;
    errorMessage.textContent = message;
}
```

---

## 4. Hugging Face Spaces Configuration

### 4.1 README.md for Space

**File:** `README.md`

```markdown
---
title: MeshNet Stroke Lesion Segmentation
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# MeshNet Stroke Lesion Segmentation

State-of-the-art stroke lesion segmentation using only 56K parameters.

## Usage

1. Upload a T2-weighted MRI scan (NIfTI format: .nii or .nii.gz)
2. Wait for processing (~30-60 seconds on CPU)
3. View results and download segmentation mask

## Model Information

| Property | Value |
|----------|-------|
| Model | MeshNet-16 |
| Parameters | 56,194 |
| DICE Score | 0.873 |
| Input Size | 256×256×256 |

## Paper

> "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"
> Fedorov et al. (Emory, Georgia State, USC)

## Repository

[GitHub: arc-meshchop](https://github.com/username/arc-meshchop)
```

---

## 5. Deployment Steps

### 5.1 Create Space

```bash
# Login to Hugging Face
huggingface-cli login

# Create new Space
huggingface-cli repo create arc-meshchop-demo --type space --space_sdk docker

# Clone and setup
git clone https://huggingface.co/spaces/username/arc-meshchop-demo
cd arc-meshchop-demo
```

### 5.2 Upload Model to Hub

```bash
# Create model repository
huggingface-cli repo create arc-meshchop-models --type model

# Upload ONNX model
huggingface-cli upload arc-meshchop-models ./exports/meshnet_16.onnx
```

### 5.3 Push to Space

```bash
# Copy files
cp -r app/ static/ Dockerfile requirements.txt README.md arc-meshchop-demo/

# Push
cd arc-meshchop-demo
git add .
git commit -m "Initial deployment"
git push
```

---

## 6. Implementation Checklist

### Phase 7.1: Backend

- [ ] Create `Dockerfile`
- [ ] Create `requirements.txt`
- [ ] Implement `app/main.py` (FastAPI)
- [ ] Implement `app/inference.py`
- [ ] Implement `app/preprocessing.py`

### Phase 7.2: Frontend

- [ ] Create `static/index.html`
- [ ] Create `static/style.css`
- [ ] Create `static/app.js`

### Phase 7.3: Deployment

- [ ] Upload model to HF Hub
- [ ] Create HF Space
- [ ] Deploy and test
- [ ] Document usage

---

## 7. Verification

```bash
# Local testing
docker build -t arc-meshchop-demo .
docker run -p 7860:7860 arc-meshchop-demo

# Open browser: http://localhost:7860
```

---

## 8. Future Enhancements

1. **Add NiiVue viewer** for in-browser visualization
2. **Multiple model selection** (MeshNet-5, -16, -26)
3. **Batch processing** for multiple files
4. **Download original + overlay** composite
5. **3D rendering** of segmentation results

---

## 9. References

- Hugging Face Spaces Docker: https://huggingface.co/docs/hub/spaces-sdks-docker
- FastAPI: https://fastapi.tiangolo.com/
- ONNX Runtime: https://onnxruntime.ai/
- NiiVue: https://github.com/niivue/niivue
