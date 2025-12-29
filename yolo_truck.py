import cv2
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import os
import json
import easyocr

# --- Configuration ---
DIAGNOSTIC_THRESHOLD = 0.40  # for "ok" vs "problem"
YOLO_MODEL_PATH = 'truck4.pt'  # Path to the YOLO model
IMAGE_PATH = "../YOLO_Image_Analysis/truck_test/truckb1.jpg"  # Default test image
OCR_LANGUAGES = ['en', 'ar']  # You can change this to ['ar'] or ['en', 'ar'] etc.

# Initialize EasyOCR reader
reader = easyocr.Reader(OCR_LANGUAGES, verbose=False)

# --- Helper Functions ---
def get_available_classes(model):
    """Retrieve the list of available classes from the YOLO model."""
    return model.names

def preprocess_image(image_path, max_size=(800, 800), quality=85):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)
        return img_byte_arr

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
    expected_counts = {
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

    expected_components = set()
    if truck_face == "truck_front":
        expected_components = {'mirror', 'light_front', 'wiper', 'mirror_top', 'plate_number'}
    elif truck_face == "truck_back":
        expected_components = {'carrier', 'lift', 'light_back', 'stand', 'plate_number'}

    # --- ENFORCE EXPECTED COUNTS ---
    for comp in list(component_data.keys()):
        if comp in expected_counts:
            detected_confs = component_data[comp]['confidences']
            expected = expected_counts[comp]
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

# --- Component Diagnostic Functions ---
def check_mirrors(comp_data):
    if comp_data["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ there is/are missing/broken mirrors")
    else:
        print("✅ the mirrors are ok")

def check_front_lights(comp_data):
    if comp_data["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ there is/are missing/broken front lights")
    else:
        print("✅ the front lights are ok")

def check_wipers(comp_data):
    if comp_data["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ there is/are missing/broken wipers")
    else:
        print("✅ the wipers are ok")

def check_mirror_top(comp_data):
    if comp_data["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("✅ mirror top is present")
    else:
        print("❌ mirror top might be missing")

def check_front_plate_number(comp_data, plate_image=None):
    if comp_data["count"] < 1 or not all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ plate number might be missing or obscured")
        return

    if plate_image is not None:
        extracted = run_ocr_on_plate(plate_image)
        if extracted:
            print(f"✅ the plate number is visible, and with the number: {extracted}")
        else:
            print("✅ plate number is visible but text could not be read")
    else:
        # Fallback if no image provided (shouldn't happen in normal flow)
        print("✅ plate number is visible")

def check_back_lights(comp_data):
    if comp_data["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ there is/are missing/broken back lights")
    else:
        print("✅ the back lights are ok")

def check_stands(comp_data):
    if comp_data["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ there is/are missing/broken stands")
    else:
        print("✅ the stands are ok")

def check_carrier(comp_data):
    if comp_data["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("✅ there is a carrier")
    else:
        print("❌ the carrier might have a problem")

def check_lift(comp_data):
    if comp_data["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("✅ there is a lift")
    else:
        print("❌ the lift might have a problem")

def check_back_plate_number(comp_data, plate_image=None):
    if comp_data["count"] < 1 or not all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        print("❌ plate number might be missing or obscured")
        return

    if plate_image is not None:
        extracted = run_ocr_on_plate(plate_image)
        if extracted:
            print(f"✅ the plate number is visible, and with the number: {extracted}")
        else:
            print("✅ plate number is visible but text could not be read")
    else:
        print("✅ plate number is visible")

def run_front_diagnostics(components, plate_crop=None):
    check_mirrors(components.get('mirror', {"count": 0, "confidence": []}))
    check_front_lights(components.get('light_front', {"count": 0, "confidence": []}))
    check_wipers(components.get('wiper', {"count": 0, "confidence": []}))
    check_mirror_top(components.get('mirror_top', {"count": 0, "confidence": []}))
    check_front_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop)

def run_back_diagnostics(components, plate_crop=None):
    check_back_lights(components.get('light_back', {"count": 0, "confidence": []}))
    check_stands(components.get('stand', {"count": 0, "confidence": []}))
    check_carrier(components.get('carrier', {"count": 0, "confidence": []}))
    check_lift(components.get('lift', {"count": 0, "confidence": []}))
    check_back_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop)

# --- Main Execution Function ---
def main():
    # Preprocess the image
    processed_image_bytes = preprocess_image(IMAGE_PATH)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(processed_image_bytes.getvalue())
        temp_file_path = temp_file.name

    try:
        # Run detection
        detected_json, annotated_image, output_image_path, full_component_data = detect_objects_in_image(temp_file_path)

        # Print clean JSON
        print(json.dumps(detected_json, indent=2))

        # Run diagnostics
        truck_face = detected_json["truck_face"]
        components = detected_json["truck_components"]
        plate_crop = None

        if truck_face in ("truck_front", "truck_back") and 'plate_number' in full_component_data:
            plate_entry = full_component_data['plate_number']
            if plate_entry['confidences'] and len(plate_entry['boxes']) > 0:
                preprocessed_img = cv2.imread(temp_file_path)
                x1, y1, x2, y2 = [int(coord) for coord in plate_entry['boxes'][0]]
                h, w = preprocessed_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    plate_crop = preprocessed_img[y1:y2, x1:x2]
        
        # Run diagnostics based on truck face
        if truck_face == "truck_back":
            run_back_diagnostics(components, plate_crop)
        elif truck_face == "truck_front":
            run_front_diagnostics(components, plate_crop)
        else:
            print("truck face not detected — no diagnostics performed")

        # Display image
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

# --- Entry Point ---
if __name__ == "__main__":
    main()