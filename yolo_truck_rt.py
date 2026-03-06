import cv2
import os
import json
import logging
from datetime import datetime
import re
from config import (TEST_IMAGE_PATH, OUTPUT_IMAGE_PATH, TEST_VIDEO_PATH, OUTPUT_VIDEO_PATH, DIAGNOSTICS_PATH,
                    INPUT_MODE, EXPECTED_COMPONENT_COUNTS, PAIRED_COMPONENTS, FRONT_EXPECTED_COMPONENTS, BACK_EXPECTED_COMPONENTS,
                    TRACKER_TYPE, CAMERA_INDEX, COUNT_TO_DECIDE_CONSENSUS, IGNORE_PERIOD_SECONDS, DIAGNOSTIC_INTERVAL_SECONDS, IOU_THRESHOLD,
                    WIPER_FRAMES_TO_COLLECT, WIPER_COLLECTION_INTERVAL_SECONDS, LIGHT_FRAMES_TO_COLLECT, LIGHT_COLLECTION_INTERVAL_SECONDS,
                    SAVE_INTERMEDIATE_DIAGNOSTICS, DELETE_INT_DIAGNOSTICS, SAVE_CROPS, DELETE_SAVED_CROPS, MIN_SAMPLE_GAP_FRAMES,
                    BLUE, RED, ORANGE, GREEN, CYAN, RESET)

from utility import (
    get_cached_yolo_model, get_available_classes, _extract_monthly_folder,
    save_cropped_detections, save_debug_crop, cleanup_intermediate_diagnostics, cleanup_saved_crops
    )
from diagnostic_history import DiagnosticHistory
from parts_diagnostics import run_front_diagnostics, run_back_diagnostics

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Change from INFO or DEBUG to adjust verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Assigns left/right labels based on truck's intrinsic orientation
def assign_left_right_to_detections(component_data, component_name, image_width, truck_face):
    """
    Assign left/right labels to paired component detections based on truck's intrinsic orientation.
    
    Logic hierarchy:
    - CASE 1 (2 detections): Compare center-x RELATIVE to each other (NO image center comparison)
    - CASE 2 (1 detection): Fallback to image center line comparison
    - CASE 3 (0 detections): Both sides return empty
    
    Truck orientation mapping (intrinsic, not image perspective):
    - truck_front: truck's LEFT side appears on image RIGHT (center_x > image_center)
    - truck_back: truck's LEFT side appears on image LEFT (center_x < image_center)
    
    Args:
        component_data: Dict with 'confidences', 'boxes', 'track_ids' from YOLO
        component_name: Original YOLO class name (e.g., "mirror")
        image_width: Width of source image for middle-line fallback
        truck_face: "truck_front", "truck_back", or "unknown"
    
    Returns:
        tuple: (left_comp_data, right_comp_data) each with count 0 or 1
    """
    
    # If not a paired component, return unchanged (caller handles single components)
    if component_name not in PAIRED_COMPONENTS:
        return component_data, None
    
    # Extract detections
    confidences = component_data.get('confidences', [])
    boxes = component_data.get('boxes', [])
    track_ids = component_data.get('track_ids', [])
    
    # Initialize empty component structure
    empty_comp = {"count": 0, "confidences": [], "track_ids": [], "boxes": []}
    
    # If no detections, return empty for both sides
    if not confidences or len(boxes) == 0:
        return empty_comp.copy(), empty_comp.copy()
    
    # Build list of detection dicts with center_x for spatial comparison
    detections = []
    for i, (conf, bbox, tid) in enumerate(zip(confidences, boxes, track_ids)):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        detections.append({
            "conf": conf,
            "bbox": list(bbox),  # Ensure mutable copy
            "track_id": tid,
            "center_x": center_x
        })
    
    # Helper to build side-specific comp_data from a detection
    def build_comp_data(det):
        return {
            "count": 1,
            "confidences": [det["conf"]],
            "track_ids": [det["track_id"]] if det["track_id"] is not None else [],
            "boxes": [det["bbox"]]
        }
    
    # CASE 1: Two detections - assign by RELATIVE x-position (user requirement: NO image center comparison)
    if len(detections) == 2:
        # Sort by center_x to identify leftmost and rightmost in IMAGE coordinates
        # Lower center_x = appears on left of image, Higher center_x = appears on right of image
        sorted_by_x = sorted(detections, key=lambda d: d["center_x"])
        image_left_det = sorted_by_x[0]  # Lower x = appears on left of image
        image_right_det = sorted_by_x[1]  # Higher x = appears on right of image
        
        # Map image position to truck's INTRINSIC side based on truck_face
        if truck_face == "truck_front":
            # Viewing front: truck's LEFT is on image RIGHT, truck's RIGHT is on image LEFT
            truck_left_det = image_right_det
            truck_right_det = image_left_det
        elif truck_face == "truck_back":
            # Viewing back: truck's LEFT is on image LEFT, truck's RIGHT is on image RIGHT
            truck_left_det = image_left_det
            truck_right_det = image_right_det
        else:
            # Unknown orientation: default to image-coordinate assignment
            truck_left_det = image_left_det
            truck_right_det = image_right_det
        
        return build_comp_data(truck_left_det), build_comp_data(truck_right_det)
    
    # CASE 2: One detection - assign by position relative to image middle line (fallback only)
    elif len(detections) == 1:
        det = detections[0]
        image_center_x = image_width / 2
        
        # Determine which side of image the detection falls on
        appears_on_image_left = det["center_x"] < image_center_x
        
        # Map to truck's intrinsic side
        if truck_face == "truck_front":
            # Front view: image-left = truck-right, image-right = truck-left
            is_truck_left = not appears_on_image_left
        elif truck_face == "truck_back":
            # Back view: image-left = truck-left, image-right = truck-right
            is_truck_left = appears_on_image_left
        else:
            # Unknown: default to image-coordinate interpretation
            is_truck_left = appears_on_image_left
        
        if is_truck_left:
            return build_comp_data(det), empty_comp.copy()
        else:
            return empty_comp.copy(), build_comp_data(det)
    
    # CASE 3: Fallback for unexpected counts (shouldn't happen after filtering)
    else:
        return empty_comp.copy(), empty_comp.copy()

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
            iou=IOU_THRESHOLD,
            verbose=False,
            persist=True
        )
    else:
        results = model(
            source,
            iou=IOU_THRESHOLD,
            verbose=False
        )
    
    if not results or len(results) == 0:
        return None, None, None
        
    result = results[0]
    annotated_result = result.plot()

    # # Debug logging for track IDs to verify tracker continuity
    # if is_video and hasattr(result.boxes, 'id') and result.boxes.id is not None:
    #     track_ids = [int(id) for id in result.boxes.id]
    #     logger.info(f"Frame {frame_id}: Track IDs = {track_ids}")

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

    # --- Filter by confidence FIRST (above), THEN assign sides (here) ---
    # Get image dimensions for middle-line fallback in spatial assignment
    if hasattr(source, 'shape'):
        # source is a numpy array (video frame or loaded image)
        img_h, img_w = source.shape[:2]
    else:
        # Fallback for other source types (use reasonable defaults)
        img_w, img_h = 1920, 1080
    
    # Process each component that might be paired
    for comp_name in list(component_data.keys()):
        if comp_name in PAIRED_COMPONENTS:
            # Get filtered component data (already limited by EXPECTED_COMPONENT_COUNTS above)
            orig_comp_data = component_data[comp_name]
            
            # Assign left/right using spatial analysis (NO caching, always compute)
            left_data, right_data = assign_left_right_to_detections(
                orig_comp_data, 
                comp_name, 
                img_w, 
                truck_face
            )
            
            # Get side-specific keys from config
            left_key, right_key = PAIRED_COMPONENTS[comp_name]
            
            # Replace generic key with side-specific keys in component_data
            component_data[left_key] = left_data
            component_data[right_key] = right_data
            del component_data[comp_name]  # Remove generic key to avoid duplication

    # --- BUILD JSON OUTPUT ---
    json_output = {
        "truck_face": truck_face,
        "truck_components": {}
    }

    for comp in expected_components:
        comp_raw = component_data.get(comp, {})

        # Safe access with fallback
        confs = comp_raw.get('confidences', [])
        boxes = comp_raw.get('boxes', [])

        # Ensure confs is a list (defensive programming)
        if not isinstance(confs, list):
            confs = [confs] if confs is not None else []
        if not isinstance(boxes, list):
            boxes = [boxes] if boxes is not None else [] 

        track_ids = comp_raw.get('track_ids', [])
        count = comp_raw.get('count', len(confs))

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
    save_to_disk=True,
    diagnosis_timestamp: str = None,
    session_folder_name: str = None,
    movement_tracker=None,
    image_width=None 
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
        diagnosis_timestamp (str): Session start timestamp in "YYYY-MM-DD HH:MM:SS" format
        session_folder_name (str): Session folder name like "int_diagnostics_20260206_143522_123"
        movement_tracker: Optional DiagnosticHistory instance for wiper movement tracking
        image_width: Image width for relative movement threshold calculation

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
        diag_messages, plate_num = run_back_diagnostics(
            components, plate_crop, 
            movement_tracker=movement_tracker,
            image_width=image_width
        )
        diagnostics_log = diag_messages
    elif truck_face == "truck_front":
        logger.info("Running front diagnostics")
        diag_messages, plate_num = run_front_diagnostics(
            components, plate_crop,
            movement_tracker=movement_tracker,
            image_width=image_width
        )
        diagnostics_log = diag_messages
    else:
        msg = "truck face not detected — no diagnostics performed"
        logger.info(msg)
        logger.warning(msg)
        diagnostics_log = [msg]

    # --- LOG YOLO SCORE FOR PLATE ---
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
        logger.info(msg)

    # --- Build enhanced truck_components using components dict (not full_component_data) ---
    enhanced_components = {}
    expected_components = []
    if truck_face == "truck_front":
        expected_components = FRONT_EXPECTED_COMPONENTS
    elif truck_face == "truck_back":
        expected_components = BACK_EXPECTED_COMPONENTS

    for comp in expected_components:
        # Get raw detection data from full_component_data
        raw_data = full_component_data.get(comp, {'confidences': [], 'boxes': [], 'track_ids': []})
        
        # Safe access with fallback to empty list - ensures confidence is always iterable
        confs = raw_data.get('confidences', [])
        # Ensure confs is a list (defensive: handle case where it might be a float)
        if not isinstance(confs, list):
            confs = [confs] if confs is not None else []

        comp_entry = {
            "count": raw_data.get('count', len(confs)),
            "confidence": confs
        }
        # Include track_ids if available (for consistency)
        if 'track_ids' in raw_data and raw_data['track_ids']:
            comp_entry["track_ids"] = [tid for tid in raw_data['track_ids'] if tid is not None]

        comp_from_components = components.get(comp, {})

        # Include wiper_moving status if available (for wipers)
        if 'wiper_moving' in comp_from_components:
            comp_entry["wiper_moving"] = comp_from_components["wiper_moving"]
        
        # Include light_working status if available (for front lights)
        if 'light_working' in comp_from_components:
            comp_entry["light_working"] = comp_from_components["light_working"]
        if 'light_sections' in comp_from_components:
            comp_entry["light_sections"] = comp_from_components["light_sections"]
                               
        # Store OCR confidence separately for plate_number
        if comp == 'plate_number' and plate_num is not None:
            comp_entry["number"] = plate_num
        
        enhanced_components[comp] = comp_entry

    # ONLY save diagnostics to JSON file if explicitly requested
    if save_to_disk:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(DIAGNOSTICS_PATH, exist_ok=True) # Base directory always exists

        # Determine output directory based on session context
        if diagnosis_timestamp and session_folder_name:
            # NEW HIERARCHY: diagnostics/YYYYMM/int_diagnostics_.../
            monthly_folder = _extract_monthly_folder(diagnosis_timestamp)
            output_dir = os.path.join(DIAGNOSTICS_PATH, monthly_folder, session_folder_name)
        else:
            # FALLBACK: Flat structure (backward compatibility)
            output_dir = DIAGNOSTICS_PATH
            logger.debug("Using flat diagnostics directory (no session context provided)")
        
        # Create nested directories
        os.makedirs(output_dir, exist_ok=True)

        # Build filename
        if is_video:
            filename = f"diagnostics_{timestamp}_frame{frame_id}.json"
        else:
            filename = f"diagnostics_{timestamp}.json"
        
        diag_output_path = os.path.join(output_dir, filename)

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
        # Stage 3: Truncate to 12 chars max, fallback to "UNKNOWN"
        plate_clean = plate_clean[:12] or "UNKNOWN"
    else:
        plate_clean = "UNKNOWN"

    # Generate filename: final_20260127_143522_ABC123.json
    filename = f"final_{date_str}_{time_str}_{plate_clean}.json"
    monthly_folder = _extract_monthly_folder(diagnosis_timestamp)
    output_dir = os.path.join(DIAGNOSTICS_PATH, monthly_folder)
    diag_output_path = os.path.join(output_dir, filename)
    
    # Create directory before saving
    os.makedirs(output_dir, exist_ok=True)
    
    # Build final diagnostic structure with consensus metadata
    final_diagnostics = {
        "diagnosis_timestamp": diagnosis_timestamp,
        "truck_face": truck_face,
        "truck_components": consensus_components,
        "diagnostics_ok": diagnostics_ok,
        "diagnostics_ng": diagnostics_ng,
        "count_to_decide_consensus": COUNT_TO_DECIDE_CONSENSUS,
        "diagnostic_interval_seconds": DIAGNOSTIC_INTERVAL_SECONDS,
        "ignore_period_seconds": IGNORE_PERIOD_SECONDS,
        "input_mode": INPUT_MODE
    }
    
    # ALWAYS save final consensus diagnostics
    os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)
    with open(diag_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_diagnostics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ FINAL CONSENSUS DIAGNOSTICS SAVED: {filename}")
    return diag_output_path

def process_single_image(
    image_path: str, 
    diagnosis_timestamp: str,
    session_folder_name: str,
    parts_session_folder: str
) -> dict | None:
    """
    Complete diagnostic pipeline for a single image from detection to final report.
    
    Args:
        image_path: Path to input image file
        diagnosis_timestamp: ISO 8601 timestamp when diagnostic operation started
        session_folder_name: Session folder name like "int_diagnostics_20260206_143522_123"
        parts_session_folder: Parts session folder name like "parts_20260206_143522_123"

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
        save_to_disk=True,
        diagnosis_timestamp=diagnosis_timestamp,
        session_folder_name=session_folder_name,
        movement_tracker=None,
        image_width=None
    )

    # Optional cropped detections
    if SAVE_CROPS:
        logger.info("Saving cropped detections to 'cropped_parts/' directory")
        save_cropped_detections(
            source_img, 
            full_component_data,
            diagnosis_timestamp=diagnosis_timestamp,
            parts_session_folder=parts_session_folder
        )
    
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
    count_to_decide_consensus: int = None,
    ignore_period: int = None,
    diagnostic_interval: float = None,
    session_folder_name: str = None,
    diagnosis_timestamp: str = None,
    parts_session_folder: str = None
) -> tuple:
    """
    Process video file or camera stream with temporal consensus tracking over multiple frames.
    
    Args:
        source_path: Video file path (str) or camera index (int) if is_camera=True
        is_camera: True for live camera feed, False for video file processing
        camera_index: Camera device index (only used if is_camera=True)
        count_to_decide_consensus: Maximum diagnostic samples to collect (default: 7)
        ignore_period: Duration in seconds to ignore at start for stabilization (default: 1)
        diagnostic_interval: Real-time interval between diagnostics in seconds (default: 3)
        session_folder_name: Session folder name like "int_diagnostics_20260206_143522_123"
        diagnosis_timestamp: Session start timestamp in "YYYY-MM-DD HH:MM:SS" format
        parts_session_folder: Parts session folder name like "parts_20260206_143522_123"

    Returns:
        tuple: (consensus_result, success)
            - consensus_result: Tuple from DiagnosticHistory.get_consensus_diagnostics() or None
                (diagnostics, truck_face, components, ok_list, ng_list)
            - success: True if stream processed successfully, False on failure
    """
    if count_to_decide_consensus is None:
        count_to_decide_consensus = COUNT_TO_DECIDE_CONSENSUS
    if diagnostic_interval is None:
        diagnostic_interval = DIAGNOSTIC_INTERVAL_SECONDS
    if ignore_period is None:
        ignore_period = IGNORE_PERIOD_SECONDS

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
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")
        
        # Initialize diagnostic history tracker with x sec consensus window and y sec ignore period
        diagnostic_history = DiagnosticHistory(
            count_to_decide_consensus=count_to_decide_consensus,
            ignore_period_seconds=ignore_period
        )
        processing_start_time = datetime.now().timestamp()
        diagnostic_history.set_start_time(processing_start_time)

        # Count-based diagnostic collection tracking
        diagnostic_count = 0
        last_diagnostic_time = datetime.now()
        last_diagnostic_frame = 0

        #  Track debug crop collection per diagnostic sample for lights and wipers
        # Format: {track_id: {"count": 0, "last_save_time": None, "sample_id": None}}
        light_crop_tracking = {}
        wiper_crop_tracking = {}

        # Processing loop
        frame_count = 0
        last_annotated_frame = None    

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            frame_count += 1
            current_time = datetime.now()
            
            # Detect/track objects (every frame)
            detected_json, annotated_frame, full_component_data = detect_or_track_objects(
                frame, is_video=True, frame_id=frame_count
            )
            
            if detected_json is None:
                annotated_frame = frame.copy()
                last_annotated_frame = annotated_frame
            else:
                last_annotated_frame = annotated_frame

            # Track wiper positions for movement verification
            if detected_json is not None and full_component_data is not None:
                for side in ["left", "right"]:
                    comp_key = f"{side}_wiper"
                    if full_component_data is not None and comp_key in full_component_data:
                        comp_data = full_component_data[comp_key]
                        # Only track if detected with valid track_id and bbox
                        if comp_data.get("count", 0) == 1 and comp_data.get("track_ids") and comp_data.get("boxes"):
                            track_id = comp_data["track_ids"][0]
                            if track_id is not None:
                                # Get bbox and extract coordinates
                                bbox = comp_data["boxes"][0]  # boxes is list of lists: [[x1,y1,x2,y2]]
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                center_x = (bbox[0] + bbox[2]) / 2
                  
                                # Add observation to diagnostic history
                                diagnostic_history.add_wiper_observation(
                                    track_id=track_id,
                                    side=side,
                                    timestamp=current_time.timestamp(),
                                    center_x=center_x,
                                    image_width=frame_width
                                )
                                logger.debug(f"Tracked {side} wiper (track_id={track_id}) at x={center_x:.1f}px, ")

                                # Save debug crop ONLY at analysis collection intervals, up to limit per diagnostic sample
                                time_since_start = (current_time - datetime.fromtimestamp(processing_start_time)).total_seconds()
                                if time_since_start >= ignore_period and diagnostic_count < count_to_decide_consensus:
                                    current_ts = current_time.timestamp()
                                    
                                    # Initialize tracking for this track_id if needed
                                    if track_id not in wiper_crop_tracking:
                                        wiper_crop_tracking[track_id] = {"count": 0, "last_save_time": None, "sample_id": None}
                                    
                                    tracking = wiper_crop_tracking[track_id]
                                    
                                    # Reset count if we've moved to a new diagnostic sample
                                    if tracking["sample_id"] != diagnostic_count:
                                        tracking["count"] = 0
                                        tracking["last_save_time"] = None
                                        tracking["sample_id"] = diagnostic_count
                                    
                                    # Save crop if: under limit for this sample (wipers: save every detected frame up to limit)
                                    # For wipers, we save every detected frame up to WIPER_MIN_FRAMES_FOR_MOVEMENT per sample
                                    if diagnostic_count >= 1 and (tracking["last_save_time"] is None or 
                                        current_ts - tracking["last_save_time"] >= WIPER_COLLECTION_INTERVAL_SECONDS) and \
                                       tracking["count"] < WIPER_FRAMES_TO_COLLECT:
                                        # Crop wiper region for debug saving
                                        h, w = frame.shape[:2]
                                        x1_clamped, y1_clamped = max(0, x1), max(0, y1)
                                        x2_clamped, y2_clamped = min(w, x2), min(h, y2)
                                        if x2_clamped > x1_clamped and y2_clamped > y1_clamped:
                                            wiper_crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                                            save_debug_crop(
                                                crop_image=wiper_crop,
                                                side=side,
                                                component="wiper",
                                                track_id=track_id,
                                                frame_id=frame_count,
                                                sample_count=diagnostic_count,  # Current diagnostic sample number (1-7)
                                                diagnosis_timestamp=diagnosis_timestamp,
                                                parts_session_folder=parts_session_folder,
                                                crop_type="wiper"
                                            )
                                            tracking["count"] += 1
                                            tracking["last_save_time"] = current_ts

            # Track light crops for brightness change verification (every frame)
            if detected_json is not None and full_component_data is not None:
                for side in ["left", "right"]:
                    comp_key = f"{side}_light_front"
                    if comp_key in full_component_data:
                        comp_data = full_component_data[comp_key]
                        if comp_data.get("count", 0) == 1 and comp_data.get("track_ids") and comp_data.get("boxes"):
                            track_id = comp_data["track_ids"][0]
                            if track_id is not None:
                                # Get bbox and extract coordinates
                                bbox = comp_data["boxes"][0]  # boxes is list of lists: [[x1,y1,x2,y2]]
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                
                                # Crop light region for tracking and debug saving (at diagnostic samples only)
                                h, w = frame.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    crop = frame[y1:y2, x1:x2]
                                    
                                    # Add observation to diagnostic history
                                    diagnostic_history.add_light_observation(
                                        track_id=track_id,
                                        side=side,
                                        timestamp=current_time.timestamp(),
                                        crop_image=crop
                                    )

                                    light_history_key = f"{side}_light_front_{track_id}"
                                    crop_count = len(diagnostic_history.light_history.get(light_history_key, {}).get('crops', []))
                                    logger.debug(
                                        f"Light crop collected: {side} (track_id={track_id}), "
                                        f"total_crops_in_history={crop_count}, crop_shape={crop.shape}"
                                    )

                                    # Save debug crop ONLY at analysis collection intervals, up to limit per diagnostic sample
                                    time_since_start = (current_time - datetime.fromtimestamp(processing_start_time)).total_seconds()
                                    if time_since_start >= ignore_period and diagnostic_count < count_to_decide_consensus:
                                        current_ts = current_time.timestamp()
                                        
                                        # Initialize tracking for this track_id if needed
                                        if track_id not in light_crop_tracking:
                                            light_crop_tracking[track_id] = {"count": 0, "last_save_time": None, "sample_id": None}
                                        
                                        tracking = light_crop_tracking[track_id]
                                        
                                        # Reset count if we've moved to a new diagnostic sample
                                        if tracking["sample_id"] != diagnostic_count:
                                            tracking["count"] = 0
                                            tracking["last_save_time"] = None
                                            tracking["sample_id"] = diagnostic_count
                                        
                                        # Save crop if: enough time passed AND under limit for this sample
                                        if diagnostic_count >= 1 and (tracking["last_save_time"] is None or 
                                            current_ts - tracking["last_save_time"] >= LIGHT_COLLECTION_INTERVAL_SECONDS) and \
                                           tracking["count"] < LIGHT_FRAMES_TO_COLLECT:
                                            
                                            save_debug_crop(
                                                crop_image=crop,
                                                side=side,
                                                component="light_front",
                                                track_id=track_id,
                                                frame_id=frame_count,
                                                sample_count=diagnostic_count,  # Current diagnostic sample number (1-7)
                                                diagnosis_timestamp=diagnosis_timestamp,
                                                parts_session_folder=parts_session_folder,
                                                crop_type="light"
                                            )
                                            tracking["count"] += 1
                                            tracking["last_save_time"] = current_ts

            # Collect diagnostics at real-time intervals until count reached
            time_since_last_diagnostic = (current_time - last_diagnostic_time).total_seconds()
            if time_since_last_diagnostic >= diagnostic_interval and diagnostic_count < count_to_decide_consensus:
                # Check if we're past ignore period
                time_since_start = (current_time - datetime.fromtimestamp(processing_start_time)).total_seconds()
                if time_since_start >= ignore_period:
                    
                    # Enforce minimum frame gap between samples
                    # This ensures each sample has enough frames for crop collection (6 frames × 0.5s = 3 seconds)
                    frames_since_last_sample = frame_count - last_diagnostic_frame if 'last_diagnostic_frame' in locals() else 0
                    if frames_since_last_sample < MIN_SAMPLE_GAP_FRAMES:
                        # Skip - not enough frames since last sample for crop collection
                        continue    
                    
                    diagnostic_count += 1
                    last_diagnostic_time = current_time
                    last_diagnostic_frame = frame_count

                    # Calculate elapsed time for logging
                    elapsed_time = time_since_start
                    logger.info(f"Diagnostic sample {diagnostic_count}/{count_to_decide_consensus} collected (frame {frame_count}, t={elapsed_time:.1f}s)")
                                    
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
                        save_to_disk=SAVE_INTERMEDIATE_DIAGNOSTICS,
                        diagnosis_timestamp=diagnosis_timestamp,
                        session_folder_name=session_folder_name,
                        movement_tracker=diagnostic_history,
                        image_width=frame_width
                    )

                    # Save cropped detections if enabled
                    if SAVE_CROPS:
                        save_cropped_detections(
                            frame, 
                            full_component_data, 
                            frame_id=frame_count,
                            diagnosis_timestamp=diagnosis_timestamp,
                            parts_session_folder=parts_session_folder
                        )
                
                    # Add to diagnostic history with current timestamp
                    diagnostic_entry = {
                        "frame_id": frame_count,
                        "truck_face": detected_json["truck_face"],
                        "truck_components": enhanced_components,
                        "timestamp": current_time.isoformat(),
                        "diagnostics": diagnostics_log
                    }
                    diagnostic_history.add_diagnostics(current_time.timestamp(), diagnostic_entry)

                    # Stop video processing after reaching target diagnostic count
                    if diagnostic_count >= count_to_decide_consensus:
                        logger.info(f"Maximum diagnostic count reached ({diagnostic_count}/{count_to_decide_consensus}). Stopping video processing.")
                        break

            # Write frame to output video (every frame)
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

    if INPUT_MODE == "video" and not os.path.exists(TEST_VIDEO_PATH):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Video file not found: {TEST_VIDEO_PATH}")
        return timestamp, None

    if INPUT_MODE == "image" and not os.path.exists(TEST_IMAGE_PATH):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Image file not found: {TEST_IMAGE_PATH}")
        return timestamp, None

    # Initialize result variable
    result = None  # Store the return value

    # RECORD TIMESTAMP: Start of diagnostic operation (after validation)
    diagnosis_start_dt = datetime.now()
    diagnosis_timestamp = diagnosis_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Millisecond precision for collision prevention (identical ms for both session types)
    ms_precision = diagnosis_start_dt.strftime("%f")[:-3]  # Trim to milliseconds
    
    # Session folder names with IDENTICAL timestamps/ms
    session_folder_name = f"int_diagnostics_{diagnosis_start_dt.strftime('%Y%m%d_%H%M%S')}_{ms_precision}"
    parts_session_folder = f"parts_{diagnosis_start_dt.strftime('%Y%m%d_%H%M%S')}_{ms_precision}"

    logger.info(f"Session folder (diagnostics): {session_folder_name}")
    logger.info(f"Session folder (cropped parts): {parts_session_folder}")

    try:
        if INPUT_MODE == "image":
            # Single image processing
            logger.info("Starting single image processing")
            result = process_single_image(
                TEST_IMAGE_PATH, 
                diagnosis_timestamp, 
                session_folder_name,
                parts_session_folder
            )
    
        elif INPUT_MODE in ("camera", "video"):
            # Video/Camera processing
            logger.info("Starting video/camera processing")
            consensus_result, success = process_video_source(
                source_path=CAMERA_INDEX if INPUT_MODE == "camera" else TEST_VIDEO_PATH,
                is_camera=(INPUT_MODE == "camera"),
                camera_index=CAMERA_INDEX,
                count_to_decide_consensus=COUNT_TO_DECIDE_CONSENSUS,
                ignore_period=IGNORE_PERIOD_SECONDS,
                diagnostic_interval=DIAGNOSTIC_INTERVAL_SECONDS,
                session_folder_name=session_folder_name,
                diagnosis_timestamp=diagnosis_timestamp,
                parts_session_folder=parts_session_folder
            )        

            # Compile final results if successful consensus obtained
            if success and consensus_result[0] is not None:
                consensus_diagnostics, truck_face, consensus_components, diagnostics_ok, diagnostics_ng = consensus_result

                # Split diagnostics into OK and NG (⚠️ counts as failing per user requirement)
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
        
        # 1. Remove temporary annotated image
        if os.path.exists(OUTPUT_IMAGE_PATH):
            os.remove(OUTPUT_IMAGE_PATH)
            logger.debug(f"Annotated image removed: {OUTPUT_IMAGE_PATH}")
        
        # 2. Interactive cleanup of intermediate diagnostics (if enabled)
        if DELETE_INT_DIAGNOSTICS:
            try:
                cleanup_intermediate_diagnostics()
            except Exception as e:
                logger.warning(f"Intermediate diagnostics cleanup failed: {e}")
        
        # 3. Interactive cleanup of saved cropped parts (if enabled)
        if DELETE_SAVED_CROPS:
            try:
                cleanup_saved_crops()
            except Exception as e:
                logger.warning(f"Saved crops cleanup failed: {e}")

        # 4. Final completion log
        logger.info("Pipeline completed")
    
    return diagnosis_timestamp, result

if __name__ == "__main__":
    diagnosis_timestamp, results = main()

    # Display timestamp prominently
    print(f"\n⏱️  Diagnosis performed at: {BLUE}{diagnosis_timestamp}{RESET}")
    
    if results is None:
        logger.error(f"{RED}❌ Truck inspection failed: no results generated.{RESET}")

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
                print(f"  {issue}")