import cv2
from ultralytics import YOLO
import os
import torch
import json
import logging
from datetime import datetime
import re
from config import OCR_LANGUAGES, EXPECTED_COMPONENT_COUNTS, FRONT_EXPECTED_COMPONENTS, BACK_EXPECTED_COMPONENTS
from diagnostic_history import DiagnosticHistory
from parts_diagnostics import run_front_diagnostics, run_back_diagnostics

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = 'truck7.pt'  # Path to the YOLO model
IMAGE_PATH = os.path.join(BASE_DIR, 'truck_test', 'truckb4.jpg')  # Default test image
OUTPUT_IMAGE_PATH = "annotated_image.jpg"
VIDEO_FILE_PATH = os.path.join(BASE_DIR, 'truck_test', 'test_video.mp4') # Default test video
VIDEO_OUTPUT_PATH = "output_video.mp4"
DIAGNOSTICS_PATH = "diagnostics"
SAVE_INTERMEDIATE_DIAGNOSTICS = True  # Set to True to save diagnostics for each frame in video mode
SAVE_CROPS = True  # Set to True to save cropped bounding boxes for validation
CROPPED_PARTS_PATH = "cropped_parts"

# --- Input & Tracking Configuration ---
INPUT_MODE = "video"  # Options: "image", "camera", "video"
CAMERA_INDEX = 0
CONSENSUS_WINDOW_SECONDS = 18
IGNORE_PERIOD_SECONDS = 1
SAVE_INTERVAL_SECONDS = 2.5
TRACKER_TYPE = "botsort.yaml"  # or "bytetrack.yaml"

BLUE = "\033[94m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

# --- GPU/CPU Detection ---
def get_device():
    """
    Automatically select GPU if available, otherwise fall back to CPU.
    
    Returns:
        str: Device identifier ('cuda' if GPU available, 'cpu' otherwise)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")
    return device

# --- YOLO Model Caching Function ---
_yolo_model = None
_yolo_device = None

def get_cached_yolo_model():
    """
    Get cached YOLO model instance to avoid reloading between frames.
    
    Returns:
        tuple: (model, device)
            - model: Cached YOLO model instance
            - device: Device where model is loaded ('cuda' or 'cpu')
    """
    global _yolo_model, _yolo_device
    if _yolo_model is None:
        _yolo_device = get_device()
        _yolo_model = YOLO(YOLO_MODEL_PATH).to(_yolo_device)
        logger.info(f"YOLO model cached on {_yolo_device} - will reuse for all frames")
    return _yolo_model, _yolo_device

# --- Object Detection Helper Functions ---
def get_available_classes(model):
    """
    Retrieve the list of available classes from the YOLO model.
    
    Args:
        model: YOLO model instance
    
    Returns:
        dict: Dictionary mapping class indices to class names
    """
    return model.names

# --- Cropping and Saving Functions ---
def extract_plate_crop(source_img, full_component_data, truck_face):
    """
    Extract plate crop from source image if plate is detected.
    
    Args:
        source_img: Source image as NumPy array (HxWxC)
        full_component_data: Raw detection data dictionary from YOLO
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
    
    Returns:
        numpy.ndarray | None: Cropped plate region as image array, or None if plate not detected/invalid
    """
    plate_crop = None
    if truck_face in ("truck_front", "truck_back") and 'plate_number' in full_component_data:
        plate_entry = full_component_data['plate_number']
        if plate_entry['confidences'] and len(plate_entry['boxes']) > 0:
            x1, y1, x2, y2 = [int(coord) for coord in plate_entry['boxes'][0]]
            h, w = source_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                plate_crop = source_img[y1:y2, x1:x2]
                logger.info("Plate region cropped successfully")
            else:
                logger.warning("Invalid plate crop coordinates")
    return plate_crop

# --- Cropping and Saving Functions ---
def save_cropped_detections(image, component_data, frame_id=None):
    """
    Save cropped regions of all detected objects for validation and debugging.
    
    Args:
        image: Source image/frame as NumPy array
        component_data: Detection data dictionary grouped by component type
        frame_id (int | None): Frame number for filename (None for single images)
    
    Returns:
        None: Saves cropped images to CROPPED_PARTS_PATH directory
    """
    if image is None:
        logger.warning("Could not load image/frame for cropping")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(CROPPED_PARTS_PATH, exist_ok=True)

    h, w = image.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_index = 0

    for class_name, data in component_data.items():
        for i, (conf, bbox) in enumerate(zip(data['confidences'], data['boxes'])):
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
            # Clamp coordinates to image boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Format: {timestamp}[_frame{frame_id}]_{class}_{confidence:.2f}_{index}.jpg
            if frame_id is not None:
                filename = f"{timestamp}_frame{frame_id}_{class_name}_{conf:.2f}_{crop_index}.jpg"
            else:
                filename = f"{timestamp}_{class_name}_{conf:.2f}_{crop_index}.jpg"
                
            filepath = os.path.join(CROPPED_PARTS_PATH, filename)
            cv2.imwrite(filepath, crop)
            crop_index += 1

# --- Core Detection and Tracking Function ---
def detect_or_track_objects(source, is_video=False, frame_id=0):
    """
    Detect objects in image or track objects in video frame using YOLO.
    
    Args:
        source: Image path, video frame, or camera feed
        is_video (bool): True for video/camera processing (enables tracking), False for single image
        frame_id (int): Frame number for logging purposes (default: 0)
    
    Returns:
        tuple: (json_output, annotated_image, component_data) or (None, None, None) on failure
            - json_output: Dictionary containing truck face and component detection results
            - annotated_image: Image with bounding boxes drawn
            - component_data: Raw detection data grouped by component type
    """
    # Use CACHED model instead of reloading every frame
    model, device = get_cached_yolo_model()
    
    # Use tracking for video, detection for image
    if is_video:
        results = model.track(
            source,
            tracker=TRACKER_TYPE,
            verbose=False
        )
    else:
        results = model(source, verbose=False)
    
    if not results or len(results) == 0:
        return None, None, None
        
    result = results[0]
    annotated_result = result.plot()

    # --- COLLECT DETECTIONS (with track IDs if available) ---
    valid_detections = []
    box_areas = {}
    has_tracking = is_video and hasattr(result.boxes, 'id') and result.boxes.id is not None

    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names.get(cls, f'unknown_class_{cls}')
        area = (x2 - x1) * (y2 - y1)
        
        # Get track ID if available
        track_id = None
        if has_tracking:
            track_id = int(result.boxes.id[i]) if i < len(result.boxes.id) else None
        
        valid_detections.append((cls, class_name, conf, (x1, y1, x2, y2), track_id))

        if class_name in ('truck_front', 'truck_back'):
            if class_name not in box_areas or area > box_areas[class_name]:
                box_areas[class_name] = area

    # --- DETERMINE TRUCK FACE ---
    truck_face = "unknown"
    if 'truck_front' in box_areas and 'truck_back' in box_areas:
        truck_face = "truck_front" if box_areas['truck_front'] > box_areas['truck_back'] else "truck_back"
    elif 'truck_front' in box_areas:
        truck_face = "truck_front"
    elif 'truck_back' in box_areas:
        truck_face = "truck_back"

    # --- GROUP DETECTIONS BY COMPONENT ---
    component_data = {}
    for cls, class_name, conf, bbox, track_id in valid_detections:
        if class_name not in component_data:
            component_data[class_name] = {'confidences': [], 'boxes': [], 'track_ids': []}
        component_data[class_name]['confidences'].append(conf)
        component_data[class_name]['boxes'].append(bbox)
        component_data[class_name]['track_ids'].append(track_id)

    # --- EXPECTED COUNTS & COMPONENTS ---
    expected_components = set()
    if truck_face == "truck_front":
        expected_components = FRONT_EXPECTED_COMPONENTS
    elif truck_face == "truck_back":
        expected_components = BACK_EXPECTED_COMPONENTS

    # --- ENFORCE EXPECTED COUNTS ---
    for comp in list(component_data.keys()):
        if comp in EXPECTED_COMPONENT_COUNTS:
            detected_confs = component_data[comp]['confidences']
            expected = EXPECTED_COMPONENT_COUNTS[comp]
            if len(detected_confs) > expected:
                sorted_confs_boxes = sorted(
                    zip(detected_confs, component_data[comp]['boxes'], component_data[comp]['track_ids']),
                    key=lambda x: x[0],
                    reverse=True
                )
                kept = sorted_confs_boxes[:expected]
                component_data[comp]['confidences'] = [c for c, _, _ in kept]
                component_data[comp]['boxes'] = [b for _, b, _ in kept]
                component_data[comp]['track_ids'] = [t for _, _, t in kept]

    # --- BUILD JSON OUTPUT ---
    json_output = {
        "truck_face": truck_face,
        "truck_components": {}
    }

    for comp in expected_components:
        count = len(component_data.get(comp, {}).get('confidences', []))
        confs = component_data.get(comp, {}).get('confidences', [])
        track_ids = component_data.get(comp, {}).get('track_ids', [])
        json_output["truck_components"][comp] = {
            "count": count,
            "confidence": [round(c, 2) for c in confs],
            "track_ids": [tid for tid in track_ids if tid is not None]
        }

    # --- ANNOTATE AND SAVE IMAGE ---
    annotated_image = result.plot()
    if not is_video:
        cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_image)

    return json_output, annotated_image, component_data

# --- Diagnostic Processing and Saving ---
def process_diagnostics_and_save(
    truck_face, 
    components, 
    full_component_data, 
    plate_crop, 
    source_info,
    is_video=False,
    frame_id=0,
    save_to_disk=True
):
    """
    Process diagnostics and save results for both image and video modes.
    
    Args:
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
        components: Component data for diagnostics (subset of full_component_data)
        full_component_data: Raw component data with boxes and confidences from YOLO
        plate_crop: Cropped plate image (NumPy array) or None if not detected
        source_info: Dict with source information (image name or frame details)
        is_video: Boolean indicating if processing video frame
        frame_id: Frame number (for video mode)
        save_to_disk (bool): If False, processes diagnostics but skips disk write (for intermediate frames)
        
    Returns:
        tuple: (diagnostics_log, enhanced_components)
            - diagnostics_log: List of diagnostic messages with status emojis
            - enhanced_components: Dictionary with enhanced component data including OCR results
    """
    # Run diagnostics based on truck face
    diagnostics_log = []
    plate_num = None

    if truck_face == "truck_back":
        logger.info("Running back diagnostics")
        diag_messages, plate_num = run_back_diagnostics(components, plate_crop)
        diagnostics_log = diag_messages
    elif truck_face == "truck_front":
        logger.info("Running front diagnostics")
        diag_messages, plate_num = run_front_diagnostics(components, plate_crop)
        diagnostics_log = diag_messages
    else:
        msg = "truck face not detected — no diagnostics performed"
        print(msg)
        logger.warning(msg)
        diagnostics_log = [msg]

    # --- LOG WEIGHTED COMBINED SCORE (YOLO*0.60 + OCR*0.40) FOR PLATE ---
    if plate_num and truck_face in ("truck_front", "truck_back") and 'plate_number' in full_component_data:
        plate_entry = full_component_data['plate_number']
        yolo_confs = plate_entry.get('confidences', [])
        if yolo_confs:
            yolo_conf = yolo_confs[0]  # Expected count = 1 for plate_number
            logger.info(f"Plate '{plate_num}' detected with YOLO confidence {yolo_conf:.2f}")
        else:
            logger.info(f"Plate '{plate_num}' detected (YOLO confidence unavailable)")

    # Print diagnostics to console
    for msg in diagnostics_log:
        print(msg)

    # --- Build enhanced truck_components using RAW data ---
    enhanced_components = {}
    expected_components = []
    if truck_face == "truck_front":
        expected_components = FRONT_EXPECTED_COMPONENTS
    elif truck_face == "truck_back":
        expected_components = BACK_EXPECTED_COMPONENTS

    for comp in expected_components:
        raw_data = full_component_data.get(comp, {'confidences': [], 'boxes': [], 'track_ids': []})
        comp_entry = {
            "count": len(raw_data['confidences']),
            "confidence": raw_data['confidences']
        }
        # Include track_ids if available (for consistency)
        if 'track_ids' in raw_data and raw_data['track_ids']:
            comp_entry["track_ids"] = [tid for tid in raw_data['track_ids'] if tid is not None]
        
        # Store OCR confidence separately for plate_number
        if comp == 'plate_number' and plate_num is not None:
            comp_entry["number"] = plate_num
        
        enhanced_components[comp] = comp_entry

    # ONLY save diagnostics to JSON file if explicitly requested
    if save_to_disk:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)

        if is_video:
            diag_output_path = os.path.join(DIAGNOSTICS_PATH, f"diagnostics_{timestamp}_frame{frame_id}.json")
        else:
            diag_output_path = os.path.join(DIAGNOSTICS_PATH, f"diagnostics_{timestamp}.json")

        with open(diag_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "frame_id": frame_id,
                "truck_face": truck_face,
                "truck_components": enhanced_components,
                "timestamp": datetime.now().isoformat(),
                "diagnostics": diagnostics_log
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Diagnostic results saved to: {diag_output_path}")
    else:
        logger.debug("Diagnostics processed but not saved to disk (intermediate frame)")

    return diagnostics_log, enhanced_components

# --- Final Consensus Diagnostics Saving ---
def save_final_consensus_diagnostics(
    diagnosis_timestamp: str,
    truck_face: str,
    consensus_components: dict,
    diagnostics_ok: list,
    diagnostics_ng: list
):
    """
    Save final consensus diagnostics with meaningful filename containing plate number.
    
    Args:
        diagnosis_timestamp: ISO 8601 timestamp when diagnostic operation started
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
        consensus_components: Dictionary of consensus component states and data
        diagnostics_ok: List of passing component diagnostic messages (✅)
        diagnostics_ng: List of failing component diagnostic messages (❌/⚠️)
    
    Returns:
        str: Path to saved consensus diagnostic JSON file
    """
    # Parse date/time for filename
    try:
        dt = datetime.strptime(diagnosis_timestamp, "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y%m%d")
        time_str = dt.strftime("%H%M%S")
    except:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
    
    # Extract plate number from consensus_components (canonical location)
    plate_number = consensus_components.get("plate_number", {}).get("number", "")

    # Clean plate number for filename with multi-stage fallback
    plate_clean = ""
    if plate_number and isinstance(plate_number, str) and plate_number.strip():
        # Stage 1: Preserve digits + Arabic characters for Moroccan plates
        plate_clean = re.sub(r'[^\d\u0600-\u06FF]', '', plate_number)
        # Stage 2: Fallback to ASCII if no Arabic/digits found
        if not plate_clean:
            plate_clean = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
        plate_clean = plate_clean[:12] or "UNKNOWN"
    else:
        plate_clean = "UNKNOWN"

    # Generate filename: final_20260127_143522_ABC123.json
    filename = f"final_{date_str}_{time_str}_{plate_clean}.json"
    diag_output_path = os.path.join(DIAGNOSTICS_PATH, filename)
    
    # Build final diagnostic structure with consensus metadata
    final_diagnostics = {
        "diagnosis_timestamp": diagnosis_timestamp,
        "truck_face": truck_face,
        "truck_components": consensus_components,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics_ng": diagnostics_ng,
        "consensus_window_seconds": CONSENSUS_WINDOW_SECONDS,
        "ignore_period_seconds": IGNORE_PERIOD_SECONDS,
        "input_mode": INPUT_MODE
    }
    
    # ALWAYS save final consensus diagnostics
    os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)
    with open(diag_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_diagnostics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ FINAL CONSENSUS DIAGNOSTICS SAVED: {filename}")
    return diag_output_path

def process_single_image(image_path: str, diagnosis_timestamp: str) -> dict | None:
    """
    Complete diagnostic pipeline for a single image from detection to final report.
    
    Args:
        image_path: Path to input image file
        diagnosis_timestamp: ISO 8601 timestamp when diagnostic operation started
    
    Returns:
        dict | None: Result dictionary with diagnostics or None on failure
            - truck_face: Detected truck face ("truck_front" or "truck_back")
            - truck_components: Enhanced component data with counts/confidences/plate number
            - diagnostics_ok: List of passing component diagnostic messages (✅)
            - diagnostics_ng: List of failing component diagnostic messages (❌/⚠️)
            - final_diagnostic_file: Path to saved consensus diagnostic JSON file
    """
    logger.info(f"Processing single image: {image_path}")
    detected_json, annotated_image, full_component_data = detect_or_track_objects(image_path, is_video=False)
    
    if detected_json is None:
        logger.error("Failed to process image")
        return None

    # Print and log detection summary    
    logger.info(f"Detected truck face: {detected_json['truck_face']}")
    # print(json.dumps(detected_json, indent=2))
    
    # Load source image for plate cropping
    source_img = cv2.imread(image_path)
    if source_img is None:
        logger.warning("Failed to load source image for plate cropping")
    
    # Extract plate crop
    plate_crop = extract_plate_crop(source_img, full_component_data, detected_json["truck_face"])
    
    # Process diagnostics and save results
    source_info = {"image_name": os.path.basename(image_path)}
    diagnostics_log, enhanced_components = process_diagnostics_and_save(
        detected_json["truck_face"],
        detected_json["truck_components"],
        full_component_data,
        plate_crop,
        source_info,
        is_video=False,
        save_to_disk=True
    )

    # Optional cropped detections
    if SAVE_CROPS:
        logger.info("Saving cropped detections to 'cropped_parts/' directory")
        save_cropped_detections(source_img, full_component_data)
    
    # Display image if not headless
    if annotated_image is not None:
        try:
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.info(f"Display unavailable (headless environment or missing GUI backend) - skipping image display: {type(e).__name__}: {e}")
    else:
        logger.warning("No annotated image available for display")
    
    # Split diagnostics into OK and NG and compile results
    diagnostics_ok = [m for m in diagnostics_log if m.startswith("✅")]
    diagnostics_ng = [m for m in diagnostics_log if m.startswith(("❌", "⚠️"))]
    
    # Save final consensus for image mode (same as intermediate since single frame)
    final_diag_path = save_final_consensus_diagnostics(
        diagnosis_timestamp=diagnosis_timestamp,
        truck_face=detected_json["truck_face"],
        consensus_components=enhanced_components,
        diagnostics_ok=diagnostics_ok,
        diagnostics_ng=diagnostics_ng
    )

    # Store result for return after cleanup
    return {
        "truck_face": detected_json["truck_face"],
        "truck_components": enhanced_components,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics_ng": diagnostics_ng,
        "final_diagnostic_file": final_diag_path
    }

def process_video_source(
    source_path: str | int, 
    is_camera: bool = False, 
    camera_index: int = 0,
    consensus_window: int = CONSENSUS_WINDOW_SECONDS,
    ignore_period: int = IGNORE_PERIOD_SECONDS,
    save_interval: float = SAVE_INTERVAL_SECONDS
) -> tuple:
    """
    Process video file or camera stream with temporal consensus tracking over multiple frames.
    
    Args:
        source_path: Video file path (str) or camera index (int) if is_camera=True
        is_camera: True for live camera feed, False for video file processing
        camera_index: Camera device index (only used if is_camera=True)
        consensus_window: Duration in seconds for temporal consensus window (default: 18)
        ignore_period: Duration in seconds to ignore at start of stream for stabilization (default: 1)
        save_interval: Interval in seconds between diagnostic saves during processing (default: 2.5)
    
    Returns:
        tuple: (consensus_result, success)
            - consensus_result: Tuple from DiagnosticHistory.get_consensus_diagnostics() or None
                (diagnostics, truck_face, components, ok_list, ng_list)
            - success: True if stream processed successfully, False on failure
    """
    # Initialize video capture
    cap = cv2.VideoCapture(camera_index if is_camera else source_path)
    source_desc = f"camera device {camera_index}" if is_camera else f"video file: {source_path}"
    if is_camera:
        logger.info(f"Using camera device: {source_desc}")
    else:
        logger.info(f"Using video file: {source_desc}")

    if not cap.isOpened():
        logger.error("Failed to open video source")
        return None, False
    
    try:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Output video will be saved to: {VIDEO_OUTPUT_PATH}")
        
        # Initialize diagnostic history tracker with x sec consensus window and y sec ignore period
        diagnostic_history = DiagnosticHistory(
            consensus_window_seconds=consensus_window,
            ignore_period_seconds=ignore_period
        )
        processing_start_time = datetime.now().timestamp()
        diagnostic_history.set_start_time(processing_start_time)
        
        # Processing loop
        last_save_time = datetime.now()
        frame_count = 0
        last_annotated_frame = None    

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            frame_count += 1
            current_time = datetime.now()
            
            # Detect/track objects
            detected_json, annotated_frame, full_component_data = detect_or_track_objects(
                frame, is_video=True, frame_id=frame_count
            )
            
            if detected_json is None:
                annotated_frame = frame.copy()
                last_annotated_frame = annotated_frame
            else:
                last_annotated_frame = annotated_frame
            
            # Run diagnostics every SAVE_INTERVAL_SECONDS
            if (current_time - last_save_time).total_seconds() >= save_interval:
                last_save_time = current_time
                
                # Extract plate crop
                plate_crop = extract_plate_crop(frame, full_component_data, detected_json["truck_face"])
                source_info = {"frame_id": frame_count, "timestamp": current_time.isoformat()}

                # Process diagnostics but only save if debugging
                diagnostics_log, enhanced_components = process_diagnostics_and_save(
                    detected_json["truck_face"],
                    detected_json["truck_components"],
                    full_component_data,
                    plate_crop,
                    source_info,
                    is_video=True,
                    frame_id=frame_count,
                    save_to_disk=SAVE_INTERMEDIATE_DIAGNOSTICS
                )
                
                # Save cropped detections if enabled
                if SAVE_CROPS:
                    save_cropped_detections(frame, full_component_data, frame_count)
                
                # Add to diagnostic history with current timestamp
                diagnostic_entry = {
                    "frame_id": frame_count,
                    "truck_face": detected_json["truck_face"],
                    "truck_components": enhanced_components,
                    "timestamp": current_time.isoformat(),
                    "diagnostics": diagnostics_log
                }
                diagnostic_history.add_diagnostics(current_time.timestamp(), diagnostic_entry)

                       # Write frame and display (camera only)
            if last_annotated_frame is not None:
                video_writer.write(last_annotated_frame)
            
            # Display real-time video (only for camera mode to avoid video playback lag)
            if is_camera:
                try:
                    display_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    if display_frame is not None:
                        cv2.imshow('Truck Diagnostic - Live', display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested exit")
                            break
                except Exception as e:
                    # Expected in headless environments - log at INFO level, not WARNING
                    logger.info(f"Display unavailable (headless environment or missing GUI backend) - continuing without live preview: {type(e).__name__}: {e}")
        
        # Get consensus after stream ends
        consensus_result = diagnostic_history.get_consensus_diagnostics()
        return consensus_result, True
    
    finally:
        # Cleanup resources
        cap.release()
        video_writer.release()
        if is_camera:
            cv2.destroyAllWindows()
        logger.info("Video processing completed")

def compile_final_results(
    diagnosis_timestamp: str,
    truck_face: str,
    consensus_components: dict,
    diagnostics_ok: list,
    diagnostics_ng: list
) -> dict:
    """
    Compile standardized result dictionary with final diagnostics for return to caller.
    
    Args:
        diagnosis_timestamp: ISO 8601 timestamp when diagnostic operation started
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
        consensus_components: Dictionary of consensus component states and data
        diagnostics_ok: List of passing component diagnostic messages (✅)
        diagnostics_ng: List of failing component diagnostic messages (❌/⚠️)
    
    Returns:
        dict: Complete result dictionary ready for return
            - truck_face: Detected truck face
            - truck_components: Consensus component data
            - diagnostics_ok: List of passing diagnostic messages
            - diagnostics_ng: List of failing diagnostic messages
            - final_diagnostic_file: Path to saved consensus diagnostic JSON file
    """
    final_diag_path = save_final_consensus_diagnostics(
        diagnosis_timestamp=diagnosis_timestamp,
        truck_face=truck_face,
        consensus_components=consensus_components,
        diagnostics_ok=diagnostics_ok,
        diagnostics_ng=diagnostics_ng
    )
    
    return {
        "truck_face": truck_face,
        "truck_components": consensus_components,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics_ng": diagnostics_ng,
        "final_diagnostic_file": final_diag_path
    }

# --- Main Execution Function ---
def main():
    """
    Main execution function for truck diagnostic.
    
    Returns:
        tuple: (diagnosis_timestamp, results_dict)
            - diagnosis_timestamp: ISO 8601 timestamp when diagnostic operation started
            - results_dict: Diagnostic results dictionary or None on complete failure
                * truck_face: "truck_front", "truck_back", or "unknown"
                * truck_components: Dict of component data with counts/confidences/plate number
                * diagnostics_ok: List of passing components (✅ messages)
                * diagnostics_ng: List of failing components (❌/⚠️ messages)
                * final_diagnostic_file: Path to saved consensus JSON file
    """
    logger.info("Starting truck diagnostic pipeline")
    logger.info(f"Input mode: {INPUT_MODE}")

    # --- PRE-FLIGHT VALIDATION WITH TIMESTAMP ON FAILURE ---
    if INPUT_MODE not in ("image", "video", "camera"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Invalid INPUT_MODE: {INPUT_MODE}. Must be 'image', 'video', or 'camera'.")
        return timestamp, None

    if INPUT_MODE == "video" and not os.path.exists(VIDEO_FILE_PATH):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Video file not found: {VIDEO_FILE_PATH}")
        return timestamp, None

    if INPUT_MODE == "image" and not os.path.exists(IMAGE_PATH):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Image file not found: {IMAGE_PATH}")
        return timestamp, None

    # Initialize result variable
    result = None  # Store the return value

    # RECORD TIMESTAMP: Start of diagnostic operation (after validation)
    diagnosis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if INPUT_MODE == "image":
            # Single image processing
            logger.info("Starting single image processing")
            result = process_single_image(IMAGE_PATH, diagnosis_timestamp)
    
        elif INPUT_MODE in ("camera", "video"):
            # Video/Camera processing
            logger.info("Starting video/camera processing")
            consensus_result, success = process_video_source(
                source_path=CAMERA_INDEX if INPUT_MODE == "camera" else VIDEO_FILE_PATH,
                is_camera=(INPUT_MODE == "camera"),
                camera_index=CAMERA_INDEX,
                consensus_window=CONSENSUS_WINDOW_SECONDS,
                ignore_period=IGNORE_PERIOD_SECONDS,
                save_interval=SAVE_INTERVAL_SECONDS
            )        

            # Compile final results if successful consensus obtained
            if success and consensus_result[0] is not None:
                consensus_diagnostics, truck_face, consensus_components, diagnostics_ok, diagnostics_ng = consensus_result

                # Split diagnostics into OK and NG
                diagnostics_ok = [m for m in consensus_diagnostics if m.startswith("✅")]
                diagnostics_ng = [m for m in consensus_diagnostics if m.startswith(("❌", "⚠️"))]

                # Extract plate_number from consensus components BEFORE use
                plate_number = consensus_components.get("plate_number", {}).get("number", "N/A")

                # ALWAYS save final consensus diagnostics
                final_diag_path = save_final_consensus_diagnostics(
                    diagnosis_timestamp=diagnosis_timestamp,
                    truck_face=truck_face,
                    consensus_components=consensus_components,
                    diagnostics_ok=diagnostics_ok,
                    diagnostics_ng=diagnostics_ng
                )

                # Build result dict
                result = {
                    "truck_face": truck_face,
                    "truck_components": consensus_components,
                    "diagnostics_ok": diagnostics_ok,
                    "diagnostics_ng": diagnostics_ng,
                    "final_diagnostic_file": final_diag_path
                }

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return diagnosis_timestamp, None
    
    finally:
        # Cleanup for both modes - THIS WILL ALWAYS RUN
        if os.path.exists(OUTPUT_IMAGE_PATH):
            os.remove(OUTPUT_IMAGE_PATH)
            logger.debug(f"Annotated image removed: {OUTPUT_IMAGE_PATH}")
        logger.info("Pipeline completed")
    
    return diagnosis_timestamp, result

if __name__ == "__main__":
    diagnosis_timestamp, results = main()

    # Display timestamp prominently
    print(f"\n⏱️  Diagnosis performed at: {BLUE}{diagnosis_timestamp}{RESET}")
    
    if results is None:
        print(f"{RED}❌ Truck inspection failed: no results generated.{RESET}")

    else:
        truck_face = results["truck_face"]
        truck_components = results["truck_components"]
        diagnostics_ok = results["diagnostics_ok"]
        diagnostics_ng = results["diagnostics_ng"]
        plate_number = truck_components.get("plate_number", {}).get("number", "N/A")
        final_diag_file = results.get("final_diagnostic_file", "N/A")

        print(f"\n✅ {ORANGE}Truck diagnosis summary:{RESET}")
        print(f"   Face: {truck_face}")
        print(f"   Plate: {plate_number}")
        print(f"   Components: {list(truck_components.keys())}")
        print(f"   ✅ Passing components: {len(diagnostics_ok)}")
        for msg in diagnostics_ok:
            print(f"      {msg}")
        print(f"   ❌/⚠️ Failing components: {len(diagnostics_ng)}")
        for msg in diagnostics_ng:
            print(f"      {msg}")
        print(f"   📁 Final diagnostics: {os.path.basename(final_diag_file) if final_diag_file != 'N/A' else 'Not saved'}")

        if not diagnostics_ng:
            print(f"\n{GREEN}✅ Truck passed inspection!{RESET}")
        else:
            print(f"\n{RED}🔧 Required repairs:{RESET}")
            for issue in diagnostics_ng:
                print("  ", issue)