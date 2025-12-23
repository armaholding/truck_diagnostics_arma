import cv2
from ultralytics import YOLO
from PIL import Image
import io
import tempfile
import os

DIAGNOSTIC_THRESHOLD = 0.50  # for "ok" vs "problem"
YOLO_MODEL_PATH = 'truck3.pt'  # Path to the YOLO model

# Example usage
# image_path = "../YOLO_Image_Analysis/truck_test/truckf4.jpg"
image_path = "../YOLO_Image_Analysis/truck_test/truckb3.jpg"

def get_available_classes(model):
    """Retrieve the list of available classes from the YOLO model."""
    return model.names

def preprocess_image(image_path, max_size=(800, 800), quality=85):
    with Image.open(image_path) as img:
    # with load_image_from_url(image_path) as img:
        img.thumbnail(max_size)  # Resize the image while maintaining aspect ratio
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)  # Reset the buffer position to the beginning
        return img_byte_arr

def detect_objects_in_image(image_path, output_image_path='annotated_image.jpg'):
    """
    Detect truck components in the given image using a custom YOLO model.
    All detected objects are included in analysis, counting, and JSON output.
    Expected component counts are enforced by keeping the highest-confidence instances.
    Diagnostics use DIAGNOSTIC_THRESHOLD (0.50) to assess component health.
    """
    # Load the YOLO model
    model = YOLO(YOLO_MODEL_PATH)
    
    # Retrieve available class names from the model
    available_classes = get_available_classes(model)

    # Perform inference on the input image
    results = model(image_path)
    result = results[0]  # Only one image is processed

    # --- COLLECT *ALL* DETECTIONS (no confidence filtering) ---
    valid_detections = []  # Will store: (cls, class_name, conf, (x1, y1, x2, y2))
    box_areas = {}         # Track largest area for 'truck_front' and 'truck_back'

    for box in result.boxes:
        # Extract bounding box coordinates, confidence, and class ID
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = available_classes.get(cls, f'unknown_class_{cls}')

        # Calculate box area and store detection as valid_detection
        area = (x2 - x1) * (y2 - y1)
        valid_detections.append((cls, class_name, conf, (x1, y1, x2, y2)))

        # For truck face determination, track the largest box area per relevant class
        if class_name in ('truck_front', 'truck_back'):
            if class_name not in box_areas or area > box_areas[class_name]:
                box_areas[class_name] = area

    # --- DETERMINE TRUCK FACE BASED ON LARGEST DETECTED TRUCK FRONT/BACK ---
    truck_face = "unknown"
    if 'truck_front' in box_areas and 'truck_back' in box_areas:
        truck_face = "truck_front" if box_areas['truck_front'] > box_areas['truck_back'] else "truck_back"
    elif 'truck_front' in box_areas:
        truck_face = "truck_front"
    elif 'truck_back' in box_areas:
        truck_face = "truck_back"

    # --- GROUP ALL DETECTIONS BY COMPONENT NAME ---
    component_data = {}
    for cls, class_name, conf, bbox in valid_detections:
        if class_name not in component_data:
            component_data[class_name] = {'confidences': [], 'boxes': []}
        component_data[class_name]['confidences'].append(conf)
        component_data[class_name]['boxes'].append(bbox)

    # --- DEFINE EXPECTED INSTANCE COUNTS PER COMPONENT ---
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

    # --- DEFINE EXPECTED COMPONENTS BASED ON DETECTED TRUCK FACE ---
    expected_components = set()
    if truck_face == "truck_front":
        expected_components = {'mirror', 'light_front', 'wiper', 'mirror_top', 'plate_number'}
    elif truck_face == "truck_back":
        expected_components = {'carrier', 'lift', 'light_back', 'stand', 'plate_number'}

    # --- ENFORCE EXPECTED COUNTS: KEEP ONLY TOP-N HIGHEST-CONFIDENCE DETECTIONS ---
    for comp in list(component_data.keys()):
        if comp in expected_counts:
            detected_confs = component_data[comp]['confidences']
            expected = expected_counts[comp]
            if len(detected_confs) > expected:
                # Sort by confidence (descending) and keep only top `expected` instances
                sorted_confs_boxes = sorted(
                    zip(detected_confs, component_data[comp]['boxes']),
                    key=lambda x: x[0],
                    reverse=True
                )
                kept = sorted_confs_boxes[:expected]
                component_data[comp]['confidences'] = [c for c, _ in kept]
                component_data[comp]['boxes'] = [b for _, b in kept]

    # --- BUILD FINAL JSON OUTPUT (INCLUDES ALL EXPECTED COMPONENTS) ---
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

    # --- GENERATE ANNOTATED IMAGE (YOLO'S .plot() SHOWS ALL DETECTIONS BY DEFAULT) ---
    annotated_image = result.plot()  # Draws all boxes, labels, and scores
    cv2.imwrite(output_image_path, annotated_image)

    return json_output, annotated_image, output_image_path

# Preprocess the image
processed_image_bytes = preprocess_image(image_path)

# Save the preprocessed image to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
    temp_file.write(processed_image_bytes.getvalue())
    temp_file_path = temp_file.name

detected_json, annotated_image, output_image_path = detect_objects_in_image(temp_file_path)

# Pretty-print JSON
import json
print(json.dumps(detected_json, indent=2))

# Run diagnostics based on truck face
truck_face = detected_json["truck_face"]
components = detected_json["truck_components"]

# Helper: get component data
def get_comp(comp_name):
    return components.get(comp_name, {"count": 0, "confidence": []})

# --- Diagnostics for truck_back ---
if truck_face == "truck_back":
    # light_back: expected 2, each >= 0.50
    lb = get_comp('light_back')
    if lb["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in lb["confidence"]):
        print("❌ there is/are missing/broken back lights")
    else:
        print("✅ the back lights are ok")

    # stand: expected 2
    st = get_comp('stand')
    if st["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in st["confidence"]):
        print("❌ there is/are missing/broken stands")
    else:
        print("✅ the stands are ok")

    # carrier: expected 1, >= 0.50
    ca = get_comp('carrier')
    if ca["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in ca["confidence"]):
        print("✅ there is a carrier")
    else:
        print("❌ the carrier might have a problem")    

    # lift: expected 1
    li = get_comp('lift')
    if li["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in li["confidence"]):
        print("✅ there is a lift")
    else:
        print("❌ the lift might have a problem")

    # plate_number: expected 1
    pn = get_comp('plate_number')
    if pn["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in pn["confidence"]):
        print("✅ plate number is visible")
    else:
        print("❌ plate number might be missing or obscured")

# --- Diagnostics for truck_front ---
elif truck_face == "truck_front":
    # mirror: expected 2
    mr = get_comp('mirror')
    if mr["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in mr["confidence"]):
        print("❌ there is/are missing/broken mirrors")
    else:
        print("✅ the mirrors are ok")

    # light_front: expected 2
    lf = get_comp('light_front')
    if lf["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in lf["confidence"]):
        print("there is/are missing/broken front lights")
    else:
        print("✅ the front lights are ok")

    # wiper: expected 2
    wp = get_comp('wiper')
    if wp["count"] < 2 or any(c < DIAGNOSTIC_THRESHOLD for c in wp["confidence"]):
        print("❌ there is/are missing/broken wipers")
    else:
        print("✅ the wipers are ok")

    # mirror_top: expected 1
    mt = get_comp('mirror_top')
    if mt["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in mt["confidence"]):
        print("✅ mirror top is present")
    else:
        print("mirror top might be missing")

    # plate_number: expected 1
    pn = get_comp('plate_number')
    if pn["count"] >= 1 and all(c >= DIAGNOSTIC_THRESHOLD for c in pn["confidence"]):
        print("✅ plate number is visible")
    else:
        print("❌ plate number might be missing or obscured")

else:
    print("truck face not detected — no diagnostics performed")

# Display the annotated image
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Clean up the temporary file
os.remove(temp_file_path)
os.remove(output_image_path)