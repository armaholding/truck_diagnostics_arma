# parts_diagnostics.py
"""Modular diagnostic functions for truck components."""

from config import DIAGNOSTIC_THRESHOLD, EXPECTED_COMPONENT_COUNTS

# --- OCR Helper Functions ---
def run_ocr_on_plate(plate_image, reader):
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

# --- Modular Component Diagnostic Functions ---
def check_mirrors(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['mirror'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ mirrors: missing/broken"
    else:
        return "✅ mirrors: ok"

def check_front_lights(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['light_front'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ front lights: missing/broken"
    else:
        return "✅ front lights: ok"

def check_wipers(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['wiper'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ wipers: missing/broken"
    else:
        return "✅ wipers: ok"

def check_mirror_top(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['mirror_top'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ top mirror: ok"
    else:
        return "❌ top mirror: missing/broken"

def check_back_lights(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['light_back'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ back lights: missing/broken"
    else:
        return "✅ back lights: ok"

def check_stands(comp_data):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['stand'] or any(c < DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ stands: missing/broken"
    else:
        return "✅ stands: ok"

def check_carrier(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['carrier'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ carrier: ok"
    else:
        return "❌ carrier: missing/broken"

def check_lift(comp_data):
    if comp_data["count"] >= EXPECTED_COMPONENT_COUNTS['lift'] and all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "✅ lift: ok"
    else:
        return "❌ lift: missing/broken"

def check_plate_number(comp_data, plate_image=None, reader=None):
    if comp_data["count"] < EXPECTED_COMPONENT_COUNTS['plate_number'] or not all(c >= DIAGNOSTIC_THRESHOLD for c in comp_data["confidence"]):
        return "❌ plate number: missing or obscured", None

    if plate_image is not None and reader is not None:
        extracted = run_ocr_on_plate(plate_image, reader)
        if extracted:
            return f"✅ plate number: visible, with number: {extracted}", extracted
        else:
            return "⚠️ plate number: visible but could not be read", None
    else:
        return "⚠️ plate number: visible but could not be read", None

# --- Diagnostic Orchestrators ---
def run_front_diagnostics(components, plate_crop=None, reader=None):
    diagnostics = []
    
    diagnostics.append(check_mirrors(components.get('mirror', {"count": 0, "confidence": []})))
    diagnostics.append(check_front_lights(components.get('light_front', {"count": 0, "confidence": []})))
    diagnostics.append(check_wipers(components.get('wiper', {"count": 0, "confidence": []})))
    diagnostics.append(check_mirror_top(components.get('mirror_top', {"count": 0, "confidence": []})))
    plate_msg, plate_number = check_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop, reader)
    diagnostics.append(plate_msg)
    
    return diagnostics, plate_number

def run_back_diagnostics(components, plate_crop=None, reader=None):
    diagnostics = []
    
    diagnostics.append(check_back_lights(components.get('light_back', {"count": 0, "confidence": []})))
    diagnostics.append(check_stands(components.get('stand', {"count": 0, "confidence": []})))
    diagnostics.append(check_carrier(components.get('carrier', {"count": 0, "confidence": []})))
    diagnostics.append(check_lift(components.get('lift', {"count": 0, "confidence": []})))
    plate_msg, plate_number = check_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop, reader)
    diagnostics.append(plate_msg)
    
    return diagnostics, plate_number