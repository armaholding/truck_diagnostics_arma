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
YOLO_MODEL_PATH = 'truck7.pt'  # Path to the YOLO model
IMAGE_PATH = "../truck_diagnostics_arma/truck_test/truckf2.jpg"  # Default test image
OUTPUT_IMAGE_PATH = "annotated_image.jpg"
DIAGNOSTICS_PATH = "diagnostics"
SAVE_CROPS = True  # Set to True to save cropped bounding boxes for validation
CROPPED_PARTS_PATH = "cropped_parts"

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

# --- Core Detection Function ---
def detect_objects_in_image(image_path):
    """Detect truck components using YOLO."""
    # Auto-select device (GPU if available, else CPU)
    device = get_device()

    # Load YOLO model and move to appropriate device
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)

    available_classes = get_available_classes(model)
    results = model(image_path, verbose=True)
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
):
    """
    Process diagnostics and save results for both image and video modes.
    
    Args:
        truck_face: Detected truck face ("truck_front", "truck_back", or "unknown")
        components: Component data for diagnostics
        full_component_data: Raw component data with boxes and confidences
        plate_crop: Cropped plate image (if available)
        source_info: Dict with source information (image name or frame details)
        
    Returns:
        tuple: (diagnostics_log, enhanced_components)
    """
    # Run diagnostics based on truck face
    diagnostics_log = []

    if truck_face == "truck_back":
        logger.info("Running back diagnostics")
        diag_messages, plate_num = run_back_diagnostics(components, plate_crop, reader)
        diagnostics_log = diag_messages
    elif truck_face == "truck_front":
        logger.info("Running front diagnostics")
        diag_messages, plate_num = run_front_diagnostics(components, plate_crop, reader)
        diagnostics_log = diag_messages
    else:
        msg = "truck face not detected — no diagnostics performed"
        print(msg)
        logger.warning(msg)
        diagnostics_log = [msg]

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
        if comp == 'plate_number' and plate_num is not None:
            comp_entry["number"] = plate_num
        enhanced_components[comp] = comp_entry

    # Save diagnostics to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)

    diag_output_path = os.path.join(DIAGNOSTICS_PATH, f"diagnostics_{timestamp}.json")
    diagnostic_results = {
        "image": source_info.get("image_name", "unknown"),
        "truck_face": truck_face,
        "truck_components": enhanced_components,
        "timestamp": datetime.datetime.now().isoformat(),
        "diagnostics": diagnostics_log
    }

    with open(diag_output_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Enhanced diagnostic results saved to: {diag_output_path}")

    return diagnostics_log, enhanced_components

# --- Main Execution Function ---
def main():
    logger.info("Starting truck diagnostic pipeline")

    result = None  # Store the return value

    try:
        # Run detection
        logger.info(f"Processing single image: {IMAGE_PATH}")
        detected_json, annotated_image, full_component_data = detect_objects_in_image(IMAGE_PATH)

        if detected_json is None:
            logger.error("Failed to process image")
            return
        
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
        diagnostics_log, enhanced_components = process_diagnostics_and_save(
            detected_json["truck_face"],
            detected_json["truck_components"],
            full_component_data,
            plate_crop,
            source_info,
        )

        # --- Optional: Save cropped detections for validation ---
        if SAVE_CROPS:
            logger.info("Saving cropped detections to 'crops/' directory")
            save_cropped_detections(source_img, full_component_data)

        # Display image
        logger.info("Displaying annotated image")
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Split diagnostics into OK and NG
        diagnostics_ok = [m for m in diagnostics_log if m.startswith("✅")]
        diagnostics_ng = [m for m in diagnostics_log if m.startswith(("❌", "⚠️"))]

        # Store result for return after cleanup
        result = {
            "truck_face": detected_json["truck_face"],
            "truck_components": enhanced_components,
            "diagnostics_ok": diagnostics_ok,
            "diagnostics_ng": diagnostics_ng
        }
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary files
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        #     logger.debug(f"Temporary file removed: {temp_file_path}")
        if os.path.exists(OUTPUT_IMAGE_PATH):
            os.remove(OUTPUT_IMAGE_PATH)
            logger.debug(f"Annotated image removed: {OUTPUT_IMAGE_PATH}")
        logger.info("Pipeline completed")
    
    return result

# --- Entry Point ---
if __name__ == "__main__":
    results = main()
    if results is None:
        print("❌ Truck inspection failed: no results generated.")
    else:
        truck_face = results["truck_face"]
        truck_components = results["truck_components"]
        diagnostics_ok = results["diagnostics_ok"]
        diagnostics_ng = results["diagnostics_ng"]
        plate_number = truck_components.get("plate_number", {}).get("number", "N/A")
        print(f"This is the summary of the truck diagnosis:\n{truck_face} \n{truck_components} \n{plate_number} \n{diagnostics_ok} \n{diagnostics_ng}")

        if not diagnostics_ng:
            print("✅ Truck passed inspection!")
        else:
            print(f"❌ Issues found: {len(diagnostics_ng)}")
            for issue in diagnostics_ng:
                print("  ", issue)