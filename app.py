import os
import io
import base64
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from ultralytics import YOLO

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upload limit
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ── Load model once at startup ─────────────────────────────────────────────
MODEL_PATH = Path('model/best.pt')
MODEL_URL = "PASTE_YOUR_MODEL_LINK_HERE" 

print("=" * 60)
print("AEROSCAN — Military Aircraft Detection")
print("=" * 60)

# Create model folder
os.makedirs("model", exist_ok=True)

# Download model if not exists
if not MODEL_PATH.exists():
    print("Downloading model...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
if MODEL_PATH.exists():
    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded — classes: {model.names}")
else:
    print("WARNING: model not found. Using demo mode.")
    model = None

print("=" * 60)

# ── Aircraft metadata for the frontend ────────────────────────────────────
AIRCRAFT_INFO = {
    'F16': {
        'full': 'F-16 Fighting Falcon',
        'role': 'Multirole Fighter',
        'origin': 'USA / General Dynamics',
        'speed': 'Mach 2.0',
        'range': '3,200 km',
        'ceiling': '15,240 m',
        'color': '#E63946'
    },
    'F18': {
        'full': 'F/A-18 Hornet',
        'role': 'Carrier Strike Fighter',
        'origin': 'USA / McDonnell Douglas',
        'speed': 'Mach 1.8',
        'range': '3,330 km',
        'ceiling': '15,240 m',
        'color': '#457B9D'
    },
    'F35': {
        'full': 'F-35 Lightning II',
        'role': '5th-Gen Stealth Multirole',
        'origin': 'USA / Lockheed Martin',
        'speed': 'Mach 1.6',
        'range': '2,220 km',
        'ceiling': '18,288 m',
        'color': '#2A9D8F'
    },
    'F15': {
        'full': 'F-15 Eagle',
        'role': 'Air Superiority Fighter',
        'origin': 'USA / McDonnell Douglas',
        'speed': 'Mach 2.5',
        'range': '3,900 km',
        'ceiling': '20,000 m',
        'color': '#E9C46A'
    },
    'C130': {
        'full': 'C-130 Hercules',
        'role': 'Military Transport Aircraft',
        'origin': 'USA / Lockheed Corporation',
        'speed': '643 km/h',
        'range': '6,850 km',
        'ceiling': '10,060 m',
        'color': '#9B5DE5'
    }
}

CLASS_NAMES = ['F16', 'F18', 'F35', 'F15', 'C130']

# ── Helpers ────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img: Image.Image, fmt='JPEG') -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def draw_boxes_on_image(img: Image.Image, results) -> Image.Image:
    """Draw YOLOv8 bounding boxes on a PIL image and return it."""
    import cv2
    import numpy as np

    img_np = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    color_map = {
        'F16':  (70,  57,  232),   # BGR
        'F18':  (157, 123, 69),
        'F35':  (143, 157, 42),
        'F15':  (74,  196, 233),
        'C130': (229, 93,  155),
    }

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = color_map.get(cls_name, (0, 255, 136))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img_bgr, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img_bgr, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 14, 20), 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan')
def scan():
    return render_template('detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receive an image, run YOLOv8 inference, return JSON result."""

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG.'}), 400

    try:
        # Open image
        img = Image.open(file.stream).convert('RGB')
        orig_w, orig_h = img.size

        if model is None:
            # ── Demo mode (no model) ─────────────────────────────────────
            import random
            cls_name = random.choice(CLASS_NAMES)
            conf     = round(random.uniform(0.60, 0.93), 4)
            info     = AIRCRAFT_INFO.get(cls_name, {})
            annotated_b64 = image_to_base64(img)

            return jsonify({
                'demo_mode':  True,
                'detections': [{
                    'class':      cls_name,
                    'confidence': conf,
                    'bbox':       [int(orig_w*0.1), int(orig_h*0.1),
                                   int(orig_w*0.9), int(orig_h*0.9)]
                }],
                'top':         cls_name,
                'top_conf':    conf,
                'aircraft':    info,
                'image_b64':   annotated_b64,
                'image_size':  [orig_w, orig_h]
            })

        # ── Real inference ───────────────────────────────────────────────
        results = model.predict(
            source=img,
            imgsz=1024,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        boxes = results[0].boxes
        detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id   = int(box.cls[0])
                conf     = round(float(box.conf[0]), 4)
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'class':      cls_name,
                    'confidence': conf,
                    'bbox':       [x1, y1, x2, y2]
                })
            # Sort by confidence descending
            detections.sort(key=lambda d: d['confidence'], reverse=True)
            top     = detections[0]['class']
            top_conf = detections[0]['confidence']
        else:
            top      = 'UNKNOWN'
            top_conf = 0.0

        # Draw boxes and encode result image
        annotated_img = draw_boxes_on_image(img, results)
        annotated_b64 = image_to_base64(annotated_img)

        return jsonify({
            'demo_mode':  False,
            'detections': detections,
            'top':        top,
            'top_conf':   top_conf,
            'aircraft':   AIRCRAFT_INFO.get(top, {}),
            'image_b64':  annotated_b64,
            'image_size': [orig_w, orig_h]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
        'device': str(next(model.model.parameters()).device) if model else 'none'
    })


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
