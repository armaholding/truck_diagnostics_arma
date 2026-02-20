import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Paths relative to script location (robust across systems)
MODEL_PATH = SCRIPT_DIR.parent / "truck8.pt"
IMAGE_PATH = SCRIPT_DIR.parent / "truck_test"  # Can be a single image or folder

# --- NEW: Mode Selection Constant ---
MODE = "video"  # Options: "image" or "video"

# --- NEW: Video Configuration Constants ---
VIDEO_PATH = SCRIPT_DIR.parent / "truck_test" / "test_video1.mp4"  # Single video file path
TRACKER_CONFIG = "botsort.yaml"  # Tracker type: "botsort.yaml" or "bytetrack.yaml"
TRACKING_CONF = 0.10  # Tracking confidence threshold (0.0 - 1.0)
PRINT_EVERY_N_FRAMES = 30  # Print console output every N frames

# Define IoU threshold
IOU_THRESHOLD = 0.35  # Maximum overlap allowed (0.0 - 1.0)
                      # Recommended: 0.30 - 0.45 for your use case

# Class names (for console output - model has its own internal names)
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

def run_inference_on_image(model, image_path, iou_thres=0.45):
    """Run YOLO inference and display result."""
    print(f"Processing: {image_path}")
    results = model(image_path, iou=iou_thres)
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
            # Extract and format Bounding Box Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            bbox_str = f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"

            # Print statement to include bbox_str
            print(f"    - {cls_name} (conf: {conf:.2f}, bbox: {bbox_str})")

    safe_imshow("YOLO Detection - Press any key to close", annotated_img)

def run_inference_on_video(model, video_path, iou_thres=0.45, tracker="botsort.yaml", track_conf=0.25, print_every_n=30):
    """Run YOLO tracking on video and display/save results."""
    print(f"Processing video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer for output
    output_path = video_path.parent / f"{video_path.stem}_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    print(f"Output video will be saved to: {output_path}")
    
    frame_count = 0
    tracked_ids = set()  # Track unique IDs seen in video
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run tracking instead of detection
        # model.track() returns results with box.id for each tracked object
        results = model.track(
            frame, 
            iou=iou_thres, 
            conf=track_conf, 
            tracker=tracker, 
            persist=True  # Maintains tracker state between frames
        )
        
        # Get annotated frame with tracking IDs
        annotated_frame = results[0].plot()
        
        # Save annotated frame to output video
        out.write(annotated_frame)
        
        # Display frame in window (real-time)
        cv2.imshow("YOLO Tracking - Press 'q' to exit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nUser requested exit. Stopping video processing...")
            break
        
        # Console output every N frames
        if frame_count % print_every_n == 0:
            boxes = results[0].boxes
            if len(boxes) == 0:
                print(f"  [Frame {frame_count}/{total_frames}] → No objects detected.")
            else:
                print(f"  [Frame {frame_count}/{total_frames}] → Detected {len(boxes)} object(s):")
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    track_id = int(box.id.item()) if box.id is not None else -1
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                    
                    # Extract and format Bounding Box Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist() 
                    bbox_str = f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                    
                    # --- NEW: Include tracking ID in output ---
                    print(f"    - {cls_name} (ID: {track_id}, conf: {conf:.2f}, bbox: {bbox_str})")
                    tracked_ids.add(track_id)
    
    # Cleanup and summary
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Video processing complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Unique tracking IDs observed: {len(tracked_ids)}")
    print(f"  Output saved to: {output_path}")

def main():
    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # --- NEW: Mode selection based on MODE constant ---
    if MODE == "video":
        # Video tracking mode
        if os.path.isfile(VIDEO_PATH):
            run_inference_on_video(
                model, 
                Path(VIDEO_PATH), 
                iou_thres=IOU_THRESHOLD, 
                tracker=TRACKER_CONFIG, 
                track_conf=TRACKING_CONF, 
                print_every_n=PRINT_EVERY_N_FRAMES
            )
        else:
            print(f"Error: Video file not found: {VIDEO_PATH}")

    elif MODE == "image":
        # Image detection mode (existing functionality)
        if os.path.isfile(IMAGE_PATH):
            run_inference_on_image(model, IMAGE_PATH, iou_thres=IOU_THRESHOLD)
        elif os.path.isdir(IMAGE_PATH):
            image_files = [f for f in os.listdir(IMAGE_PATH) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                print(f"No images found in folder: {IMAGE_PATH}")
                return
            image_files.sort()
            for img_file in image_files:
                run_inference_on_image(model, os.path.join(IMAGE_PATH, img_file), iou_thres=IOU_THRESHOLD)
                cont = input("\nPress Enter for next image, or 'q' to quit: ").strip().lower()
                if cont == 'q':
                    break
        else:
            print(f"Error: {IMAGE_PATH} is not a valid file or folder.")
    else:
        print(f"Error: Invalid MODE '{MODE}'. Use 'image' or 'video'.")

if __name__ == "__main__":
    main()