import cv2
import random
from ultralytics import YOLO
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import logging

# =================== LOGGING SETUP ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =================== CONFIGURATION ===================
CONFIDENCE_THRESHOLD = 0.50
DETECTION_INTERVAL = 0.50  # seconds of VIDEO time between detections
DEBUG_BOXES = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trash3.pt')
VIDEO_PATH = os.path.join(BASE_DIR, "trash_video.mp4")
DETECTION_JSON_PATH = os.path.join(BASE_DIR, "trash_detection.json")
TRASH_IMAGE_FOLDER = os.path.join(BASE_DIR, "trash_images_storage")

# # =============== NVR / CAMERAS CONFIG =================
# load_dotenv()
# cameras_ip = os.getenv("CAMERAS_IP")
# username = os.getenv("USERNAME")
# password = os.getenv("PASSWORD")

# # Define your cameras here (you can add/remove entries)
# CAMERAS = [
#     {"name": "cam1_front",  "channel": 1, "subtype": 1},
#     {"name": "cam2_back",   "channel": 2, "subtype": 1},
#     {"name": "cam3_left",   "channel": 3, "subtype": 1},
#     {"name": "cam4_right",  "channel": 4, "subtype": 1},
# ]
# # ======================================================

def get_available_classes(model):
    """Retrieve the list of available classes from the YOLO model."""
    return model.names


def detect_objects_in_frame(frame, model, available_classes):
    """
    Run YOLO on a single frame and return:
      - detections: list of dicts with raw data
      - raw_tracked_objects: set of (class_name, tracker_id) for global mapping
    """
    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        tracker= "botsort.yaml"  # or "bytetrack.yaml"
    )

    detections = []
    raw_tracked_objects = set()

    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes
            track_ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.int().tolist()

            for i in range(len(track_ids)):
                track_id = track_ids[i]
                cls = classes[i]
                conf = confs[i]
                x1, y1, x2, y2 = xyxy[i]

                if conf >= CONFIDENCE_THRESHOLD:
                    class_name = available_classes.get(cls, f'unknown_class_{cls}')
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'class_name': class_name,
                        'tracker_id': track_id,
                        'confidence': conf
                    })
                    raw_tracked_objects.add((class_name, track_id))

                    if DEBUG_BOXES:
                        logger.debug(f"Track ID: {track_id}, Class: {class_name}, Conf: {conf:.2f}")

    return detections, raw_tracked_objects

def draw_frame_with_global_ids(frame, detections, object_global_id_map):
    """
    Draw bounding boxes and labels using GLOBAL IDs.
    Returns annotated frame (numpy array).
    """
    frame_copy = frame.copy()
    color_map = {}  # Cache colors per global ID

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        tracker_id = det['tracker_id']
        key = (class_name, tracker_id)
        
        # Get global ID
        global_id = object_global_id_map.get(key, -1)
        
        # Assign color based on global ID
        if global_id not in color_map:
            # Generate consistent color (e.g., based on ID)
            random.seed(global_id)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color_map[global_id] = color
        else:
            color = color_map[global_id]

        # Draw box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label: "class #global_id"
        label = f"{class_name} #{global_id}"
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame_copy

def decide_action(detected_classes):
    """
    Apply your business rules based on detected_classes
    and return the selected action (also prints it).
    """
    action = None

    if 'heavy bag' in detected_classes and 'rubble trash' in detected_classes:
        action = "mobilize ampliroll with godet"
    elif ('heavy bag' in detected_classes and any(
        with_heavy_bag in detected_classes
        for with_heavy_bag in ['binned trash', 'ordinary trash', 'trash bag', 'loose trash', 'green litter']
    )):
        action = "mobilize bom or bensat"
    elif any(heavy_type in detected_classes for heavy_type in ['rubble trash', 'heavy bag', 'bulky trash']):
        action = "mobilize ampliroll with godet"
    elif 'wood trash' in detected_classes:
        action = "mobilize ampliroll with grappin or ampliroll with godet"
    elif 'green trash' in detected_classes:
        action = "mobilize bensat or ampliroll with grappin"
    else:
        if ('big bin' in detected_classes and any(
            trash_type in detected_classes for trash_type in ['ordinary trash', 'binned trash', 'trash bag']
        )):
            action = "mobilize bom"
        elif any(trash_type in detected_classes for trash_type in ['ordinary trash', 'binned trash', 'trash bag']):
            action = "mobilize bom or bensat"
        elif any(grass_or_litter in detected_classes for grass_or_litter in ['grass', 'loose trash', 'green litter', 'dirt']):
            action = "mobilize agents"
        elif any(clean in detected_classes for clean in ['big bin', 'medium bin', 'special bin', 'cleaner', 'trash truck', 'car']) or not detected_classes:
            action = "area is clean!"

    logger.info(f"Decided action: '{action}' based on classes: {detected_classes}")
    return action

def save_detection(tracked_objects, annotated_frame):
    """
    Save the annotated frame and append detection info into JSON.
    Logs tracked_objects_with_global_ids: set of (class_name, global_id)
    """
    os.makedirs(TRASH_IMAGE_FOLDER, exist_ok=True)

    now = datetime.now()
    timestamp_iso = now.isoformat()
    timestamp_for_filename = now.strftime("%Y%m%d_%H%M%S_%f")

    # image_filename = f"{camera_name}_ch{channel}_det_{timestamp_for_filename}.jpg"
    # image_path = os.path.join(TRASH_IMAGE_FOLDER, image_filename)
    # cv2.imwrite(image_path, annotated_frame)

    image_filename = f"trash_det_{timestamp_for_filename}.jpg"
    image_path = os.path.join(TRASH_IMAGE_FOLDER, image_filename)
    cv2.imwrite(image_path, annotated_frame)

    image_url = f"trash_images_storage/{image_filename}"

    # Convert tracked_objects (frozenset of (class, id)) to list of dicts
    tracked_objects_list = [
        {"class": class_name, "id": global_id} 
        for class_name, global_id in tracked_objects]

    # Derive class names for action decision
    detected_classes_for_action = tuple(sorted(set(cls for cls, _ in tracked_objects)))
    action = decide_action(detected_classes_for_action)

    record = {
        "timestamp": timestamp_iso,
        # "camera_name": camera_name,
        # "channel": channel,
        "source": "trash_video.mp4",
        "tracked_objects": tracked_objects_list,
        "action": action,
        "image_url": image_url
    }

    # Load existing data
    if os.path.exists(DETECTION_JSON_PATH):
        try:
            with open(DETECTION_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded existing JSON with {len(data)} records.")
        except Exception as e:
            logger.error(f"Failed to load JSON, starting fresh: {e}")
            data = []
    else:
        data = []
        logger.debug("JSON file not found. Starting new log.")

    if not isinstance(data, list):
        data = []

    data.append(record)

    try:
        with open(DETECTION_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detection: {image_url}")
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")

    # print(f"📸 [{camera_name}] Detection saved: {image_url}, logged in {os.path.basename(DETECTION_JSON_PATH)}")
    print(f"📸 Detection saved: {image_url}, logged in {os.path.basename(DETECTION_JSON_PATH)}")


# def build_rtsp_url(channel, subtype):
#     """Build Dahua RTSP URL for a given channel & subtype."""
#     return (
#         f"rtsp://{username}:{password}@{cameras_ip}:554/"
#         f"cam/realmonitor?channel={channel}&subtype={subtype}"
#     )

def resize_for_display(frame, max_width=1280, max_height=720):
    """
    Resize a frame to fit within max_width x max_height while preserving aspect ratio.
    Returns scaled frame.
    """
    h, w = frame.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # never upscale

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def draw_annotated_frame(frame, tracked_objects_with_global_ids, raw_boxes_data, available_classes):
    """
    Draw bounding boxes and labels with GLOBAL IDs.
    tracked_objects_with_global_ids: set of (class_name, global_id)
    raw_boxes_data: list of (x1, y1, x2, y2, class_name, tracker_id)
    """
    import cv2
    frame_copy = frame.copy()
    
    # Build a fast lookup: (class_name, tracker_id) → global_id
    global_id_lookup = {}
    for class_name, global_id in tracked_objects_with_global_ids:
        # We don't have tracker_id here → so we must pass it differently
        # BETTER: modify logic to keep (class, tracker_id, global_id, box) together
        pass  # We'll refactor below

def main():
    logger.info("Starting video analysis pipeline...")

    # Validate paths
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Video file not found: {VIDEO_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        logger.error(f"YOLO model not found: {MODEL_PATH}")
        return

    # Load YOLO model
    logger.info("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        available_classes = get_available_classes(model)
        logger.info(f"✅ Model loaded. Classes: {available_classes}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return

    # # Open RTSP streams for all cameras
    # caps = {}
    # last_detection_time = {}
    # previous_classes = {}
    # last_annotated_frame = {}  # <-- NEW: store latest frame with boxes per camera

    # for cam in CAMERAS:
    #     name = cam["name"]
    #     channel = cam["channel"]
    #     subtype = cam["subtype"]

    #     rtsp_url = build_rtsp_url(channel, subtype)
    #     print(f"Connecting to {name} (ch{channel}, sub{subtype}) → {rtsp_url}")
    #     cap = cv2.VideoCapture(rtsp_url)

    #     if not cap.isOpened():
    #         print(f"❌ Could not open RTSP stream for {name}. Check config.")
    #     else:
    #         print(f"✅ Stream opened for {name}.")
    #         caps[name] = cap
    #         last_detection_time[name] = 0
    #         previous_classes[name] = None
    #         last_annotated_frame[name] = None

    # if not caps:
    #     print("❌ No camera streams could be opened. Exiting.")
    #     return

    # print(f"\n✅ Multi-camera detection started. YOLO runs every {DETECTION_INTERVAL} s per camera. Press 'q' to quit.\n")

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback
    logger.info(f"Video opened. FPS: {fps:.2f}")

    # ========== GLOBAL TRACKING STATE ==========
    last_detection_video_time = -DETECTION_INTERVAL
    previous_global_tracked_objects = None  # set of (class, global_id)
    last_annotated_frame = None

    # Global ID management
    global_id_counter = 1
    object_global_id_map = {}  # key: (class_name, tracker_id) → global_id

    # Log initial info
    logger.info(f"Processing '{VIDEO_PATH}'. Detection interval: {DETECTION_INTERVAL}s (video time).")

    # while True:
    #     current_time = time.time()

    #     for cam in CAMERAS:
    #         name = cam["name"]
    #         channel = cam["channel"]

    #         if name not in caps:
    #             continue

    #         cap = caps[name]
    #         ret, frame = cap.read()

    #         if not ret or frame is None:
    #             print(f"⚠️ Failed to grab frame from {name}.")
    #             time.sleep(0.2)
    #             continue

    #         # Run YOLO only every DETECTION_INTERVAL seconds per camera
    #         if current_time - last_detection_time[name] >= DETECTION_INTERVAL:
    #             detected_classes, annotated_frame = detect_objects_in_frame(
    #                 frame, model, available_classes, confidence_threshold=CONFIDENCE_THRESHOLD
    #             )

    #             # store the last annotated frame (with boxes)
    #             last_annotated_frame[name] = annotated_frame

    #             if detected_classes:
    #                 print(f"\n[{name}] Detected Classes:", detected_classes)

    #             if detected_classes != previous_classes[name]:
    #                 if detected_classes:
    #                     action = decide_action(detected_classes)
    #                     save_detection(
    #                         detected_classes,
    #                         action,
    #                         annotated_frame,
    #                         camera_name=name,
    #                         channel=channel
    #                     )
    #                 else:
    #                     action = decide_action(detected_classes)

    #                 previous_classes[name] = detected_classes

    #             last_detection_time[name] = current_time

    #         # Decide what to show:
    #         # Prefer the last annotated frame (with boxes), otherwise raw frame
    #         frame_to_show = last_annotated_frame[name] if last_annotated_frame[name] is not None else frame

    #         cv2.imshow(f"YOLO - {name}", frame_to_show)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # --- Single video processing loop ---
    cv2.namedWindow("YOLO - trash_video.mp4", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.info("✅ End of video reached.")
            break

        # Get current video timestamp in seconds
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Run YOLO only if enough video time has passed
        if current_video_time - last_detection_video_time >= DETECTION_INTERVAL:
            logger.debug(f"Running detection at video time: {current_video_time:.2f}s")
            try:
                detections, raw_tracked_objects = detect_objects_in_frame(
                    frame, model, available_classes
                )
                # Map to global IDs
                current_global_tracked_objects = set()
                for class_name, tracker_id in raw_tracked_objects:
                    key = (class_name, tracker_id)
                    if key not in object_global_id_map:
                        object_global_id_map[key] = global_id_counter
                        logger.debug(f"Assigned new global ID {global_id_counter} to {key}")
                        global_id_counter += 1
                    global_id = object_global_id_map[key]
                    current_global_tracked_objects.add((class_name, global_id))

                # Draw annotated frame with global IDs
                annotated_frame = draw_frame_with_global_ids(frame, detections, object_global_id_map)
                last_annotated_frame = annotated_frame  # cache for display

            except Exception as e:
                logger.error(f"YOLO detection failed at {current_video_time:.2f}s: {e}")
                continue

            # Save only if there's a change in detected objects
            if current_global_tracked_objects != previous_global_tracked_objects:
                if current_global_tracked_objects:
                    # For console readability: show class names (IDs are in JSON)
                    detected_classes_for_print = tuple(sorted(set(name for name, _ in current_global_tracked_objects)))
                    logger.info(f"[Video @ {current_video_time:.2f}s] New detection: {detected_classes_for_print}")
                else:
                    logger.info(f"[Video @ {current_video_time:.2f}s] Scene is clean.")
                
                try:
                    save_detection(current_global_tracked_objects, annotated_frame)
                except Exception as e:
                    logger.error(f"Failed to save detection at {current_video_time:.2f}s: {e}")
                
                previous_global_tracked_objects = current_global_tracked_objects
                last_detection_video_time = current_video_time  # align to actual detection time

        # Display logic: prefer annotated frame, else raw
        frame_to_show = last_annotated_frame if last_annotated_frame is not None else frame

        # Resize for comfortable viewing
        frame_display = resize_for_display(frame_to_show, max_width=1280, max_height=720)

        # Show the frame
        cv2.imshow("YOLO - trash_video.mp4", frame_display)
        
        # Allow early exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("⏹️ Stopped by user.")
            break

    # Cleanup
    # for name, cap in caps.items():
    #     cap.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("✅ Video processing completed, windows closed.")

if __name__ == "__main__":
    main()
