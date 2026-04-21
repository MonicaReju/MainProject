# AeroScan — Military Aircraft Detection

Flask + YOLOv8s web application for real-time military aircraft detection.

## Folder Structure

```
aeroscan/
├── app.py                  ← Flask backend (all routes + inference)
├── requirements.txt        ← Python dependencies
├── save_weights.py         ← Run in Colab to download best.pt
├── .gitignore
├── model/
│   └── best.pt             ← Place your trained weights here  ⬅
├── templates/
│   └── index.html          ← Full frontend (single file)
├── static/
│   ├── css/                ← (empty — styles are inline in index.html)
│   └── js/                 ← (empty — JS is inline in index.html)
└── uploads/                ← Temp upload folder (auto-created)
```

## Setup in VS Code

### 1. Get your model weights from Colab
Run `save_weights.py` as a cell in your Colab notebook.
It will download `best.pt` directly to your computer.
Place it at: `aeroscan/model/best.pt`

### 2. Create virtual environment
```bash
cd aeroscan
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Flask
```bash
python app.py
```

Open browser at: **http://localhost:5000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the AeroScan UI |
| `/predict` | POST | Accepts image, returns JSON detections |
| `/health` | GET | Model status check |

### /predict request
```
POST /predict
Content-Type: multipart/form-data
Body: file=<image file>
```

### /predict response
```json
{
  "demo_mode": false,
  "top": "F16",
  "top_conf": 0.873,
  "detections": [
    {"class": "F16", "confidence": 0.873, "bbox": [x1, y1, x2, y2]},
    {"class": "F18", "confidence": 0.312, "bbox": [x1, y1, x2, y2]}
  ],
  "aircraft": {
    "full": "F-16 Fighting Falcon",
    "role": "Multirole Fighter",
    "speed": "Mach 2.0",
    "range": "3,200 km",
    "ceiling": "15,240 m",
    "color": "#E63946"
  },
  "image_b64": "<base64 annotated image>",
  "image_size": [1024, 768]
}
```

## Notes

- If `model/best.pt` is missing, the app runs in **demo mode** (random results, no real inference)
- `demo_mode: true` will show an amber DEMO MODE label in the UI
- Max upload size: 16 MB
