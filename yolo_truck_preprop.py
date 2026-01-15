import cv2
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import os
import json
import easyocr
import logging
import datetime

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DIAGNOSTIC_THRESHOLD = 0.40  # for "ok" vs "ng"
YOLO_MODEL_PATH = 'truck4.pt'  # Path to the YOLO model
IMAGE_PATH = "../truck_diagnostics_arma/truck_test/truckb1.jpg"  # Default test image
OCR_LANGUAGES = ['en']  # You can change this to ['ar'] or ['en', 'ar'] etc.
EXPECTED_COMPONENT_COUNTS = {
    'mirror': 2,
    'light_front': 2,
    'wiper': 2,
    'mirror_top': 1,
    'plate_number': 1,
    'carrier': 1,
    'lift': 1,
    'light_back': 2,
    'stand': 2,
    'truck_front': 1,
    'truck_back': 1
}
FRONT_EXPECTED_COMPONENTS = {'mirror', 'light_front', 'wiper', 'mirror_top', 'plate_number'}
BACK_EXPECTED_COMPONENTS = {'carrier', 'lift', 'light_back', 'stand', 'plate_number'}
SAVE_CROPS = True  # Set to True to save cropped bounding boxes for validation

# Initialize EasyOCR reader
reader = easyocr.Reader(OCR_LANGUAGES, verbose=False)

# --- Helper Functions ---
def get_available_classes(model):
    """Retrieve the list of available classes from the YOLO model."""
    return model.names

def preprocess_image(image_path, max_size=(800, 800), quality=100):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)
        return img_byte_arr

def save_cropped_detections(image_path, component_data, output_dir="crops"):
    """
    Save cropped regions of all detected objects for validation.
    
    Args:
        image_path (str): Path to the preprocessed image used for YOLO inference
        component_data (dict): Full detection data from detect_objects_in_image
        output_dir (str): Directory to save cropped images (created if missing)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the image that was used for detection
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not load image for cropping: {image_path}")
        return

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

            # Format: {timestamp}_{class}_{confidence:.2f}_{index}.jpg
            filename = f"{timestamp}_{class_name}_{conf:.2f}_{crop_index}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop)
            crop_index += 1

def detect_objects_in_image(image_path, output_image_path='annotated_image.jpg'):
    """
    Detect truck components using YOLO.
    Returns:
        - json_output: clean dict for logging (no boxes)
        - annotated_image: NumPy array
        - output_image_path: path to saved image
        - component_data: full detection data including boxes (for internal use)
    """
    model = YOLO(YOLO_MODEL_PATH)
    available_classes = get_available_classes(model)
    results = model(image_path)
    result = results[0]

    # --- COLLECT *ALL* DETECTIONS ---
    valid_detections = []
    box_areas = {}

    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = available_classes.get(cls, f'unknown_class_{cls}')
        area = (x2 - x1) * (y2 - y1)
        valid_detections.append((cls, class_name, conf, (x1, y1, x2, y2)))

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
    for cls, class_name, conf, bbox in valid_detections:
        if class_name not in component_data:
            component_data[class_name] = {'confidences': [], 'boxes': []}
        component_data[class_name]['confidences'].append(conf)
        component_data[class_name]['boxes'].append(bbox)

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
                    zip(detected_confs, component_data[comp]['boxes']),
                    key=lambda x: x[0],
                    reverse=True
                )
                kept = sorted_confs_boxes[:expected]
                component_data[comp]['confidences'] = [c for c, _ in kept]
                component_data[comp]['boxes'] = [b for _, b in kept]

    # --- BUILD JSON OUTPUT ---
    json_output = {
        "truck_face": truck_face,
        "truck_components": {}
    }
    for comp in expected_components:
        count = len(component_data.get(comp, {}).get('confidences', []))
        confs = component_data.get(comp, {}).get('confidences', [])
        json_output["truck_components"][comp] = {
            "count": count,
            "confidence": [round(c, 2) for c in confs]
        }

    # --- ANNOTATE AND SAVE IMAGE ---
    annotated_image = result.plot()
    cv2.imwrite(output_image_path, annotated_image)

    return json_output, annotated_image, output_image_path, component_data

# --- OCR Function ---
def run_ocr_on_plate(plate_image):
    """Run EasyOCR and return cleaned alphanumeric plate text, or None."""
    try:
        results = reader.readtext(plate_image)
        if not results:
            return None
        
        # Get the text with highest confidence
        best_text = None
        best_conf = 0.0
        for (bbox, text, conf) in results:
            if conf > best_conf:
                best_conf = conf
                best_text = text

        if best_text and best_conf > 0.1:  # Very low threshold for presence
            # Clean: keep only alphanumeric and space (remove symbols)
            cleaned = ''.join(ch for ch in best_text if ch.isalnum() or ch.isspace())
            cleaned = cleaned.strip().upper()
            if cleaned:
                return cleaned
        return None
    except Exception as e:
        # Optionally log, but for now just suppress
        return None

def crop_plate_from_original(original_image_path, preprocessed_image_path, plate_bbox):
    """Crop the plate region from the ORIGINAL high-resolution image using bounding box coordinates detected on the PREPROCESSED (low-res) image."""
    # Load images
    original_img = cv2.imread(original_image_path)
    preprocessed_img = cv2.imread(preprocessed_image_path)

    if original_img is None:
        logger.error(f"Failed to load original image: {original_image_path}")
        return None
    if preprocessed_img is None:
        logger.error(f"Failed to load preprocessed image: {preprocessed_image_path}")
        return None

    orig_h, orig_w = original_img.shape[:2]
    proc_h, proc_w = preprocessed_img.shape[:2]

    # Avoid division by zero
    if proc_w == 0 or proc_h == 0:
        logger.error("Preprocessed image has zero dimension")
        return None

    # Unpack bounding box (from YOLO on preprocessed image)
    x1_low, y1_low, x2_low, y2_low = plate_bbox

    # Scale coordinates to original image space
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    x1_orig = max(0, int(x1_low * scale_x))
    y1_orig = max(0, int(y1_low * scale_y))
    x2_orig = min(orig_w, int(x2_low * scale_x))
    y2_orig = min(orig_h, int(y2_low * scale_y))

    # Validate crop region
    if x2_orig <= x1_orig or y2_orig <= y1_orig:
        logger.warning("Invalid plate crop region after scaling")
        return None

    # Crop from ORIGINAL high-res image
    plate_crop = original_img[y1_orig:y2_orig, x1_orig:x2_orig]
    if plate_crop.size == 0:
        logger.warning("Empty plate crop")
        return None

    logger.debug(f"Plate cropped from original image: {plate_crop.shape}")
    return plate_crop

# --- MODULAR DIAGNOSTIC FUNCTIONS (return message string) ---
def check_mirrors(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['mirror'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ there is/are missing/broken mirrors"
    else:
        return "✅ the mirrors are ok"

def check_front_lights(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['light_front'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ there is/are missing/broken front lights"
    else:
        return "✅ the front lights are ok"

def check_wipers(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['wiper'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ there is/are missing/broken wipers"
    else:
        return "✅ the wipers are ok"

def check_mirror_top(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['mirror_top'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ mirror top is present"
    else:
        return "❌ mirror top might be missing"

def check_back_lights(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['light_back'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ there is/are missing/broken back lights"
    else:
        return "✅ the back lights are ok"

def check_stands(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['stand'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ there is/are missing/broken stands"
    else:
        return "✅ the stands are ok"

def check_carrier(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['carrier'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ there is a carrier"
    else:
        return "❌ the carrier might have a problem"

def check_lift(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['lift'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ there is a lift"
    else:
        return "❌ the lift might have a problem"

def check_plate_number(comp_data, plate_image=None):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['plate_number'] or not all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ plate number might be missing or obscured", None

    if plate_image is not None:
        extracted = run_ocr_on_plate(plate_image)
        if extracted:
            return f"✅ the plate number is visible, and with the number: {extracted}", extracted
        else:
            return "✅ plate number is visible but text could not be read", None
    else:
        return "✅ plate number is visible", None

# --- Diagnostic Orchestrators (now return list of messages) ---
def run_front_diagnostics(components, plate_crop=None):
    diagnostics = []
    
    diagnostics.append(check_mirrors(components.get('mirror', {"count": 0, "confidence": []})))
    diagnostics.append(check_front_lights(components.get('light_front', {"count": 0, "confidence": []})))
    diagnostics.append(check_wipers(components.get('wiper', {"count": 0, "confidence": []})))
    diagnostics.append(check_mirror_top(components.get('mirror_top', {"count": 0, "confidence": []})))
    plate_msg, plate_number = check_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop)
    diagnostics.append(plate_msg)  # ← Only the string!
    
    return diagnostics, plate_number

def run_back_diagnostics(components, plate_crop=None):
    diagnostics = []
    
    diagnostics.append(check_back_lights(components.get('light_back', {"count": 0, "confidence": []})))
    diagnostics.append(check_stands(components.get('stand', {"count": 0, "confidence": []})))
    diagnostics.append(check_carrier(components.get('carrier', {"count": 0, "confidence": []})))
    diagnostics.append(check_lift(components.get('lift', {"count": 0, "confidence": []})))
    plate_msg, plate_number = check_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop)
    diagnostics.append(plate_msg)
    
    return diagnostics, plate_number

# --- Main Execution Function ---
def main():
    logger.info("Starting truck diagnostic pipeline")

    # Preprocess the image
    logger.info(f"Preprocessing image: {IMAGE_PATH}")
    processed_image_bytes = preprocess_image(IMAGE_PATH)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(processed_image_bytes.getvalue())
        temp_file_path = temp_file.name

    try:
        # Run detection
        logger.info("Running YOLO detection")
        detected_json, annotated_image, output_image_path, full_component_data = detect_objects_in_image(temp_file_path)

        # Print and log detection summary
        print(json.dumps(detected_json, indent=2))
        logger.info(f"Detected truck face: {detected_json['truck_face']}")

        # Run diagnostics
        truck_face = detected_json["truck_face"]
        components = detected_json["truck_components"]
        plate_crop = None

        if truck_face in ("truck_front", "truck_back") and 'plate_number' in full_component_data:
            plate_entry = full_component_data['plate_number']
            # # Plate number cropping from preprocessed image
            # if plate_entry['confidences'] and len(plate_entry['boxes']) > 0:
            #     preprocessed_img = cv2.imread(temp_file_path)
            #     x1, y1, x2, y2 = [int(coord) for coord in plate_entry['boxes'][0]]
            #     h, w = preprocessed_img.shape[:2]
            #     x1, y1 = max(0, x1), max(0, y1)
            #     x2, y2 = min(w, x2), min(h, y2)
            #     if x2 > x1 and y2 > y1:
            #         plate_crop = preprocessed_img[y1:y2, x1:x2]
            #         logger.info("Plate region cropped successfully")
            #     else:
            #         logger.warning("Invalid plate crop coordinates")
            # Plate number cropping from original
            if plate_entry['confidences'] and len(plate_entry['boxes']) > 0:
                plate_bbox = plate_entry['boxes'][0]  # (x1, y1, x2, y2) in preprocessed image coordinates
                plate_crop = crop_plate_from_original(
                    original_image_path=IMAGE_PATH,
                    preprocessed_image_path=temp_file_path,
                    plate_bbox=plate_bbox
                )
                if plate_crop is not None:
                    logger.info("Plate successfully cropped from original high-resolution image")
        
        # Run diagnostics based on truck face
        diagnostics_log = []
        extracted_plate_number = None

        if truck_face == "truck_back":
            logger.info("Running back diagnostics")
            diag_messages, plate_num = run_back_diagnostics(components, plate_crop)
            diagnostics_log = diag_messages
            extracted_plate_number = plate_num
        elif truck_face == "truck_front":
            logger.info("Running front diagnostics")
            diag_messages, plate_num = run_front_diagnostics(components, plate_crop)
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
            raw_data = full_component_data.get(comp, {'confidences': [], 'boxes': []})
            comp_entry = {
                "count": len(raw_data['confidences']),
                "confidence": raw_data['confidences']
            }
            # Add OCR result only for plate_number and only if OCR succeeded
            if comp == 'plate_number' and extracted_plate_number is not None:
                comp_entry["number"] = extracted_plate_number
            enhanced_components[comp] = comp_entry

        # Save diagnostics to JSON file
        import datetime  # <<< Only imported when needed
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_output_path = f"diagnostics_{timestamp}.json"
        diagnostic_results = {
            "image": os.path.basename(IMAGE_PATH),
            "truck_face": truck_face,
            "truck_components": enhanced_components,
            "timestamp": datetime.datetime.now().isoformat(),
            "diagnostics": diagnostics_log  # ← MUST be list of strings!
        }
        with open(diag_output_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced diagnostic results saved to: {diag_output_path}")

        # --- Optional: Save cropped detections for validation ---
        if SAVE_CROPS:
            logger.info("Saving cropped detections to 'crops/' directory")
            save_cropped_detections(temp_file_path, full_component_data)

        # Display image
        logger.info("Displaying annotated image")
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temporary file removed: {temp_file_path}")
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
            logger.debug(f"Annotated image removed: {output_image_path}")
        logger.info("Pipeline completed")

# --- Entry Point ---
if __name__ == "__main__":
    main()