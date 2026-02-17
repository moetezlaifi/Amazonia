# 🌿 Amazonia  
## AI-Powered Digital Excavation for the Amazon

🥇 **1st Place — Upside Hack Hackathon (Nerdata ENSIT Club)**

Amazonia is a complete geospatial AI web application designed to surface hidden anthropogenic patterns beneath dense Amazon rainforest canopy using satellite imagery and multimodal anomaly detection.

Built in 24 hours by a team of three, the project integrates AI analysis, asynchronous backend processing, and an interactive web interface into a fully functional end-to-end system.

---

# 🚀 What Amazonia Does

Amazonia allows users to:

- Select an Area of Interest (AOI) directly on an interactive map
- Run AI-powered satellite analysis asynchronously
- Detect vegetation anomalies using NDVI
- Detect structural irregularities using DINOv2 (Vision Transformer)
- Fuse both signals into a unified confidence heatmap
- Extract and rank candidate hotspot regions
- Explore anomaly explanations interactively

The system provides **AI-assisted decision support**, not definitive archaeological classification.

---

# 🧠 AI Pipeline

## 1️⃣ Satellite Retrieval
- Sentinel-2 L2A imagery via STAC (Microsoft Planetary Computer)
- Cloud filtering and best-scene selection
- Raster stacking with `stackstac`

## 2️⃣ Spectral Anomaly Detection
- NDVI-based vegetation anomaly scoring
- Percentile normalization
- Canopy deviation mapping

## 3️⃣ Structural Anomaly Detection
- Self-supervised Vision Transformer (`dinov2_vits14`)
- Patch token extraction
- Cosine similarity anomaly scoring

## 4️⃣ Multimodal Fusion
- Weighted spectral + structural blending
- Gaussian smoothing
- Confidence surface generation

## 5️⃣ Hotspot Extraction & Ranking
- Thresholding + connected components
- Polygon generation (GeoJSON)
- Ranked candidate export (CSV)

---

# 🌐 Web Application Architecture

### Interactive Frontend
- Leaflet map centered on the Amazon
- AOI drawing (polygon + rectangle)
- Live hotspot rendering
- Interactive anomaly exploration

### Backend (Django)
- Asynchronous job system
- Background processing
- Status polling endpoint
- Structured job folders (`media/<job_id>/`)
- GeoJSON + metadata parsing

### User Experience
- Non-blocking analysis
- Real-time status updates (queued / running / done / error)
- Zoom-to-hotspot functionality
- Responsible AI disclaimer

---

# 🛠 Tech Stack

- Python
- Django
- Sentinel-2 (STAC / Planetary Computer)
- Rasterio / Stackstac
- NumPy / SciPy
- PyTorch (DINOv2)
- GeoPandas / Shapely
- Leaflet
- Streamlit (initial prototype)

---

# Installation

```bash
pip install -r requirements.txt
```
# Run the Application
```bash
python manage.py runserver
```

Then open:
```bash
http://127.0.0.1:8000
```
