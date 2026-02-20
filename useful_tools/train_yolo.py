from ultralytics import YOLO
import os

# ==============================
# Configuration (match your CLI args)
# ==============================
MODEL_PATH = "yolo11m.pt"  # or "yolov8m.pt" if using standard model
DATA_YAML = "truck_parts-13/data.yaml"
EPOCHS = 180
IMG_SIZE = 640
DEVICE = 0  # GPU 0; use 'cpu' for CPU
CACHE = "disk"  # or True/False
OPTIMIZER = "auto"
EXIST_OK = True  # overwrite existing project/name
DETERMINISTIC = True # for reproducibility
AMP = True  # Automatic Mixed Precision
VAL = True  # validate every epoch
PLOTS = True

# Optional: Set project and name for output directory
# PROJECT = "runs/train"
NAME = "truck_parts"

# ==============================
# Training
# ==============================

def main():
    print(f"🚀 Loading model from: {MODEL_PATH}")
    
    # Load model
    if os.path.isfile(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        print(f"⚠️ Model not found locally. Attempting to auto-download '{MODEL_PATH}'...")
        model = YOLO(MODEL_PATH)  # Ultralytics will try to download if it's a known model

    print("🏋️ Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        cache=CACHE,
        optimizer=OPTIMIZER,
        exist_ok=EXIST_OK,
        deterministic=DETERMINISTIC,
        amp=AMP,
        val=VAL,
        plots=PLOTS,
        # project=PROJECT,
        name=NAME,
        verbose=True
    )
    
    print("✅ Training completed!")
    print(f"Results saved to: runs/train/{NAME}")

if __name__ == "__main__":
    main()