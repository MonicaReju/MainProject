"""
Run this cell in Google Colab AFTER training is complete.
It saves best.pt to your Drive AND lets you download it directly.
"""

from google.colab import drive, files
from pathlib import Path
import shutil

drive.mount('/content/drive')

# ── Source: your trained model ─────────────────────────────────────────────
BEST_PT = '/content/drive/MyDrive/aircraft_runs_improved/phase2_yolov8s_1024_resumed2/weights/best.pt'

# ── Destination in Drive (clean copy) ─────────────────────────────────────
SAVE_PATH = '/content/drive/MyDrive/aeroscan_deployment/best.pt'
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(BEST_PT, SAVE_PATH)
print(f"✅ Saved to Drive: {SAVE_PATH}")
print(f"   Size: {Path(SAVE_PATH).stat().st_size / 1e6:.1f} MB")

# ── Also download directly to your computer ────────────────────────────────
print("\n📥 Downloading best.pt to your computer...")
files.download(BEST_PT)
print("✅ Download started — save it to   aeroscan/model/best.pt   in VS Code")
