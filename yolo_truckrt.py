import cv2
from ultralytics import YOLO
# from PIL import Image
# import io
# import tempfile
import os
import json
import easyocr
import logging
import datetime
import torch
from config import OCR_LANGUAGES, DIAGNOSTIC_THRESHOLD, EXPECTED_COMPONENT_COUNTS, FRONT_EXPECTED_COMPONENTS, BACK_EXPECTED_COMPONENTS
from parts_diagnostics import run_front_diagnostics, run_back_diagnostics

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
YOLO_MODEL_PATH = 'truck4.pt'  # Path to the YOLO model
IMAGE_PATH = "../truck_diagnostics_arma/truck_test/truckb4.jpg"  # Default test image
OUTPUT_IMAGE_PATH = "annotated_image.jpg"
DIAGNOSTICS_PATH = "diagnostics"
SAVE_CROPS = True  # Set to True to save cropped bounding boxes for validation
CROPPED_PARTS_PATH = "cropped_parts"

# --- Input & Tracking Configuration ---
INPUT_MODE = "image"  # Options: "image", "camera"
CAMERA_INDEX = 0
VIDEO_OUTPUT_PATH = "output_video.mp4"
SAVE_INTERVAL_SECONDS = 2
TRACKER_TYPE = "botsort.yaml"  # or "bytetrack.yaml"

# Initialize EasyOCR reader
reader = easyocr.Reader(OCR_LANGUAGES, verbose=False)

# --- GPU/CPU Detection ---
def get_device():
    """Automatically select GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")
    return device

# --- Object Detection Helper Functions ---
def get_available_classes(model):
    """Retrieve the list of available classes from the YOLO model."""
    return model.names

# def preprocess_image(image_path, max_size=(800, 800), quality=100):
#     with Image.open(image_path) as img:
#         img.thumbnail(max_size)
#         img_byte_arr = io.BytesIO()
#         img.save(img_byte_arr, format='JPEG', quality=quality)
#         img_byte_arr.seek(0)
#         return img_byte_arr

# --- Core Detection and Tracking Function ---
def detect_or_track_objects(source, is_video=False, frame_id=0):
    """Detect objects in image or track objects in video frame."""
    # Auto-select device (GPU if available, else CPU)
    device = get_device()

    # Load YOLO model and move to appropriate device
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    
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

# --- Cropping and Saving Functions ---
def extract_plate_crop(source_img, full_component_data, truck_face):
    """
    Extract plate crop from source image if plate is detected.
    
    Args:
        source_img: Source image (NumPy array)
        full_component_data: Raw detection data
        truck_face: Detected truck face
        
    Returns:
        plate_crop: Cropped plate image or None
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
    """Save cropped regions of all detected objects for validation."""
    if image is None:
        logger.warning("Could not load image/frame for cropping")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(CROPPED_PARTS_PATH, exist_ok=True)

    h, w = image.shape[:2]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

# --- Diagnostic Processing and Saving ---
def process_diagnostics_and_save(
    truck_face, 
    components, 
    full_component_data, 
    plate_crop, 
    source_info,
    is_video=False,
    frame_id=0
):
    """
    Process diagnostics and save results for both image and video modes.
    
    Args:
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
        components: Component data for diagnostics
        full_component_data: Raw component data with boxes and confidences
        plate_crop: Cropped plate image (if available)
        source_info: Dict with source information (image name or frame details)
        is_video: Boolean indicating if processing video frame
        frame_id: Frame number (for video mode)
        
    Returns:
        tuple: (diagnostics_log, extracted_plate_number, enhanced_components)
    """
    # Run diagnostics based on truck face
    diagnostics_log = []
    extracted_plate_number = None

    if truck_face == "truck_back":
        logger.info("Running back diagnostics")
        diag_messages, plate_num = run_back_diagnostics(components, plate_crop, reader)
        diagnostics_log = diag_messages
        extracted_plate_number = plate_num
    elif truck_face == "truck_front":
        logger.info("Running front diagnostics")
        diag_messages, plate_num = run_front_diagnostics(components, plate_crop, reader)
        diagnostics_log = diag_messages
        extracted_plate_number = plate_num
    else:
        msg = "truck face not detected — no diagnostics performed"
        print(msg)
        logger.warning(msg)
        diagnostics_log = [msg]
        extracted_plate_number = None

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
        if comp == 'plate_number' and extracted_plate_number is not None:
            comp_entry["number"] = extracted_plate_number
        enhanced_components[comp] = comp_entry

    # Save diagnostics to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)

    if is_video:
        diag_output_path = os.path.join(DIAGNOSTICS_PATH, f"diagnostics_{timestamp}_frame{frame_id}.json")
        diagnostic_results = {
            "frame_id": frame_id,
            "truck_face": truck_face,
            "truck_components": enhanced_components,
            "timestamp": datetime.datetime.now().isoformat(),
            "diagnostics": diagnostics_log
        }
    else:
        diag_output_path = os.path.join(DIAGNOSTICS_PATH, f"diagnostics_{timestamp}.json")
        diagnostic_results = {
            "frame_id": 0,
            "image": source_info.get("image_name", "unknown"),
            "truck_face": truck_face,
            "truck_components": enhanced_components,
            "timestamp": datetime.datetime.now().isoformat(),
            "diagnostics": diagnostics_log
        }

    with open(diag_output_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced diagnostic results saved to: {diag_output_path}")

    return diagnostics_log, extracted_plate_number, enhanced_components

# --- NEW: Diagnostic History Tracker with Ignore Period ---
class DiagnosticHistory:
    """Track diagnostic history and compute consensus decisions with ignore period."""
    
    def __init__(self, consensus_window_seconds=20, ignore_period_seconds=2):
        self.consensus_window = consensus_window_seconds
        self.ignore_period = ignore_period_seconds
        self.diagnostic_history = []  # List of (timestamp, diagnostics_dict)
        self.component_history = {}   # Track each component's states over time
        self.processing_start_time = None
        
    def set_start_time(self, start_time):
        """Set the processing start time for ignore period calculation."""
        self.processing_start_time = start_time
        
    def add_diagnostics(self, timestamp, diagnostics_result):
        """Add diagnostic result to history."""
        if self.processing_start_time is None:
            self.processing_start_time = timestamp
            
        # Store full diagnostic result
        self.diagnostic_history.append((timestamp, diagnostics_result))
        
        # Update component history
        for component, data in diagnostics_result.get("truck_components", {}).items():
            if component not in self.component_history:
                self.component_history[component] = []
            
            # Determine component state
            state = self._get_component_state(component, data)
            self.component_history[component].append((timestamp, state))
        
        # Clean up old entries (keep full consensus window)
        self._cleanup_old_entries(timestamp)
    
    def _get_component_state(self, component, data):
        """Determine the state of a component based on diagnostic rules."""
        count = data.get("count", 0)
        confidences = data.get("confidence", [])
        
        if component == "plate_number":
            # Plate number state depends on OCR success
            if "number" in data:
                return "✅"
            elif count >= EXPECTED_COMPONENT_COUNTS.get(component, 1) and all(c >= DIAGNOSTIC_THRESHOLD for c in confidences):
                return "⚠️"
            else:
                return "❌"
        
        # For other components
        expected_count = EXPECTED_COMPONENT_COUNTS.get(component, 1)
        if count < expected_count or any(c < DIAGNOSTIC_THRESHOLD for c in confidences):
            return "❌"
        else:
            return "✅"
    
    def _cleanup_old_entries(self, current_time):
        """Remove entries older than consensus window."""
        cutoff_time = current_time - self.consensus_window
        self.diagnostic_history = [
            (ts, diag) for ts, diag in self.diagnostic_history 
            if ts >= cutoff_time
        ]
        
        for component in self.component_history:
            self.component_history[component] = [
                (ts, state) for ts, state in self.component_history[component]
                if ts >= cutoff_time
            ]
    
    def get_consensus_diagnostics(self):
        """Get consensus diagnostic messages based on history (excluding ignore period)."""
        if not self.diagnostic_history or self.processing_start_time is None:
            return None, None, None, None
        
        # Calculate usable time window
        current_time = datetime.datetime.now().timestamp()
        ignore_cutoff = self.processing_start_time + self.ignore_period
        window_end = current_time
        window_start = max(ignore_cutoff, current_time - self.consensus_window)
        
        # Filter diagnostic history to usable window
        usable_diagnostics = [
            (ts, diag) for ts, diag in self.diagnostic_history
            if ts >= window_start and ts <= window_end
        ]
        
        if not usable_diagnostics:
            # No usable data, return latest available
            latest_diag = self.diagnostic_history[-1][1]
            return self._build_diagnostics_from_result(latest_diag)
        
        # Use the most recent truck face from usable data
        latest_usable_diag = usable_diagnostics[-1][1]
        truck_face = latest_usable_diag["truck_face"]
        
        # Build consensus component states from usable window only
        consensus_components = {}
        consensus_diagnostics = []
        extracted_plate_number = None
        
        # Determine which components to check based on truck face
        expected_components = self._get_expected_components(truck_face)
        
        for component in expected_components:
            # Filter component history to usable window
            if component in self.component_history:
                usable_component_history = [
                    (ts, state) for ts, state in self.component_history[component]
                    if ts >= window_start and ts <= window_end
                ]
            else:
                usable_component_history = []
            
            if usable_component_history:
                # Get most frequent state in the usable window
                states = [state for _, state in usable_component_history]
                state_counts = {}
                for state in states:
                    state_counts[state] = state_counts.get(state, 0) + 1
                
                # Get most common state
                consensus_state = max(state_counts, key=state_counts.get)
                
                # Build diagnostic message
                message = self._build_consensus_message(component, consensus_state)
                consensus_diagnostics.append(message)
                
                # Use latest component data from usable window
                latest_component_data = latest_usable_diag["truck_components"].get(component, {})
                consensus_components[component] = latest_component_data
                
                # Extract plate number if available
                if component == "plate_number" and "number" in latest_component_data:
                    extracted_plate_number = latest_component_data["number"]
            else:
                # No usable data for this component, use latest available
                latest_component_data = latest_usable_diag["truck_components"].get(component, {})
                consensus_components[component] = latest_component_data
                
                # Build message based on latest data
                if latest_component_data:
                    state = self._get_component_state(component, latest_component_data)
                    message = self._build_consensus_message(component, state)
                else:
                    message = f"❓ {component}: no usable data"
                consensus_diagnostics.append(message)
        
        return consensus_diagnostics, truck_face, consensus_components, extracted_plate_number
    
    def _get_expected_components(self, truck_face):
        """Get expected components based on truck face."""
        if truck_face == "truck_front":
            return list(FRONT_EXPECTED_COMPONENTS)
        elif truck_face == "truck_back":
            return list(BACK_EXPECTED_COMPONENTS)
        else:
            return list(EXPECTED_COMPONENT_COUNTS.keys())
    
    def _build_consensus_message(self, component, state):
        """Build diagnostic message based on component and consensus state."""
        component_names = {
            'mirror': 'mirrors',
            'light_front': 'front lights', 
            'wiper': 'wipers',
            'mirror_top': 'top mirror',
            'plate_number': 'plate number',
            'carrier': 'carrier',
            'lift': 'lift',
            'light_back': 'back lights',
            'stand': 'stands'
        }
        
        display_name = component_names.get(component, component)
        
        if state == "✅":
            if component == "plate_number":
                return f"✅ {display_name}: visible with number"
            else:
                return f"✅ {display_name}: ok"
        elif state == "⚠️":
            return f"⚠️ {display_name}: visible but could not be read"
        else:  # "❌"
            if component == "plate_number":
                return f"❌ {display_name}: missing or obscured"
            else:
                return f"❌ {display_name}: missing/broken"
    
    def _build_diagnostics_from_result(self, diagnostics_result):
        """Build diagnostics from a single result (fallback)."""
        truck_face = diagnostics_result["truck_face"]
        enhanced_components = diagnostics_result["truck_components"]
        extracted_plate_number = diagnostics_result["extracted_plate_number"]
        diagnostics_log = diagnostics_result["diagnostics"]
        
        return diagnostics_log, truck_face, enhanced_components, extracted_plate_number

# --- Main Execution Function ---
def main():
    """
    Main execution function for truck diagnostic pipeline.
    
    Returns:
        dict or None: Diagnostic results containing:
            - diagnostics: list of diagnostic messages
            - truck_face: detected truck face
            - truck_components: component data with counts and confidence
            - extracted_plate_number: OCR result (if successful)
            Returns None if processing fails.
    """
    logger.info("Starting truck diagnostic pipeline")
    logger.info(f"Input mode: {INPUT_MODE}")

    result = None  # Store the return value

    # # Preprocess the image
    # logger.info(f"Preprocessing image: {IMAGE_PATH}")
    # processed_image_bytes = preprocess_image(IMAGE_PATH)

    # # Save to temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
    #     temp_file.write(processed_image_bytes.getvalue())
    #     temp_file_path = temp_file.name

    try:
        if INPUT_MODE == "image":
            # Single image processing
            logger.info(f"Processing single image: {IMAGE_PATH}")
            detected_json, annotated_image, full_component_data = detect_or_track_objects(IMAGE_PATH, is_video=False)
            
            if detected_json is None:
                logger.error("Failed to process image")
                return None
                
            # Print and log detection summary
            logger.info(f"Detected truck face: {detected_json['truck_face']}")
            # print(json.dumps(detected_json, indent=2))

            # Load source image for plate cropping
            source_img = cv2.imread(IMAGE_PATH)
            if source_img is None:
                logger.warning("Failed to load source image for plate cropping")

            # Extract plate crop
            plate_crop = extract_plate_crop(source_img, full_component_data, detected_json["truck_face"])
            
            # Process diagnostics and save results
            source_info = {"image_name": os.path.basename(IMAGE_PATH)}
            diagnostics_log, extracted_plate_number, enhanced_components = process_diagnostics_and_save(
                detected_json["truck_face"],
                detected_json["truck_components"],
                full_component_data,
                plate_crop,
                source_info,
                is_video=False
            )

            # --- Optional: Save cropped detections for validation ---
            if SAVE_CROPS:
                logger.info("Saving cropped detections to 'cropped_parts/' directory")
                save_cropped_detections(source_img, full_component_data)

            # Display image
            logger.info("Displaying annotated image")
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Store result for return after cleanup
            result = {
                "truck_face": detected_json["truck_face"],
                "truck_components": enhanced_components,
                "extracted_plate_number": extracted_plate_number,
                "diagnostics": diagnostics_log
            }
    
        else:
            # Video/Camera processing (new functionality)
            logger.info("Starting video/camera processing")
            
            # Initialize video capture
            if INPUT_MODE == "camera":
                cap = cv2.VideoCapture(CAMERA_INDEX)
                logger.info(f"Using camera device {CAMERA_INDEX}")
            else:
                logger.error("Only 'image' and 'camera' modes supported")
                return None

            if not cap.isOpened():
                logger.error("Failed to open camera")
                return None

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
            logger.info(f"Output video will be saved to: {VIDEO_OUTPUT_PATH}")

            # Initialize diagnostic history tracker with 20s window and 2s ignore period
            diagnostic_history = DiagnosticHistory(
                consensus_window_seconds=20, 
                ignore_period_seconds=2
            )
            processing_start_time = datetime.datetime.now().timestamp()
            diagnostic_history.set_start_time(processing_start_time)

            # Timing for JSON saving interval
            last_save_time = datetime.datetime.now()
            frame_count = 0
            last_annotated_frame = None

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("End of video stream")
                        break

                    frame_count += 1
                    current_time = datetime.datetime.now()

                    # Track objects in frame using built-in YOLO tracking
                    detected_json, annotated_frame, full_component_data = detect_or_track_objects(
                        frame, is_video=True, frame_id=frame_count
                    )

                    if detected_json is None:
                        # No detections, use original frame
                        annotated_frame = frame.copy()
                        last_annotated_frame = annotated_frame
                    else:
                        last_annotated_frame = annotated_frame
                        
                        # Run diagnostics every SAVE_INTERVAL_SECONDS
                        current_time = datetime.datetime.now()
                        if (current_time - last_save_time).total_seconds() >= SAVE_INTERVAL_SECONDS:
                            last_save_time = current_time
                            
                            # Extract plate crop
                            plate_crop = extract_plate_crop(frame, full_component_data, detected_json["truck_face"])
                            
                            # Process diagnostics and save results
                            source_info = {"frame_id": frame_count}
                            diagnostics_log, extracted_plate_number, enhanced_components = process_diagnostics_and_save(
                                detected_json["truck_face"],
                                detected_json["truck_components"],
                                full_component_data,
                                plate_crop,
                                source_info,
                                is_video=True,
                                frame_id=frame_count
                            )

                            # Save cropped detections if enabled
                            if SAVE_CROPS:
                                save_cropped_detections(frame, full_component_data, frame_count)

                            # Add to diagnostic history
                            diagnostic_result = {
                                "truck_face": detected_json["truck_face"],
                                "truck_components": enhanced_components,
                                "extracted_plate_number": extracted_plate_number,
                                "diagnostics": diagnostics_log
                            }
                            diagnostic_history.add_diagnostics(current_time.timestamp(), diagnostic_result)

                    # Write annotated frame to video
                    if last_annotated_frame is not None:
                        video_writer.write(last_annotated_frame)
                    
                    # Display real-time video
                    cv2.imshow('Truck Diagnostic - Live', last_annotated_frame if last_annotated_frame is not None else frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested exit")
                        break

            finally:
                # Cleanup video resources
                cap.release()
                video_writer.release()
                cv2.destroyAllWindows()
                logger.info("Video processing completed")

            # Return consensus diagnostics after processing ends
            consensus_diagnostics, truck_face, consensus_components, extracted_plate_number = diagnostic_history.get_consensus_diagnostics()
            
            if consensus_diagnostics is not None:
                result = {
                    "diagnostics": consensus_diagnostics,
                    "truck_face": truck_face,
                    "truck_components": consensus_components,
                    "extracted_plate_number": extracted_plate_number
                }
            else:
                result = None

    finally:
        # Cleanup for both modes - THIS WILL ALWAYS RUN
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        #     logger.debug(f"Temporary file removed: {temp_file_path}")
        if os.path.exists(OUTPUT_IMAGE_PATH):
            os.remove(OUTPUT_IMAGE_PATH)
            logger.debug(f"Annotated image removed: {OUTPUT_IMAGE_PATH}")
        logger.info("Pipeline completed")
    
    return result

if __name__ == "__main__":
    diagnostics = main()
    print(f"We will pass this output in next pipeline: {diagnostics}")