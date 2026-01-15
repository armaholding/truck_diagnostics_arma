import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "truck6.pt"
IMAGE_PATH = "../truck_diagnostics_arma/truck_test/"  # Can be a single image or folder
CLASS_NAMES = [
    'carrier', 'lift', 'light_back', 'light_front', 'mirror',
    'mirror_top', 'plate_number', 'stand', 'truck_back', 'truck_front', 'wiper'
]

def safe_imshow(window_name, image):
    """Show image with OpenCV; fallback to saving if GUI not available."""
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        if "The function is not implemented" in str(e):
            fallback = "detected_output.jpg"
            cv2.imwrite(fallback, image)
            print(f"\n⚠️  GUI not available. Result saved to: {fallback}")
            input("Press Enter after viewing the image...")
        else:
            raise

def run_inference_on_image(model, image_path):
    """Run YOLO inference and display result."""
    print(f"Processing: {image_path}")
    results = model(image_path)
    annotated_img = results[0].plot()  # Ultralytics auto-adds labels & boxes

    # Optional: Print detections
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("  → No objects detected.")
    else:
        print(f"  → Detected {len(boxes)} object(s):")
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
            print(f"    - {cls_name} (conf: {conf:.2f})")

    safe_imshow("YOLO Detection - Press any key to close", annotated_img)

def main():
    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # Check if IMAGE_PATH is a file or directory
    if os.path.isfile(IMAGE_PATH):
        run_inference_on_image(model, IMAGE_PATH)
    elif os.path.isdir(IMAGE_PATH):
        image_files = [f for f in os.listdir(IMAGE_PATH) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"No images found in folder: {IMAGE_PATH}")
            return
        image_files.sort()
        for img_file in image_files:
            run_inference_on_image(model, os.path.join(IMAGE_PATH, img_file))
            cont = input("\nPress Enter for next image, or 'q' to quit: ").strip().lower()
            if cont == 'q':
                break
    else:
        print(f"Error: {IMAGE_PATH} is not a valid file or folder.")

if __name__ == "__main__":
    main()