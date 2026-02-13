# parts_diagnostics.py
"""Modular diagnostic functions for truck components."""

import logging
import cv2
import torch
import os
from PIL import Image
import tempfile
from config import DIAGNOSTIC_THRESHOLD, EXPECTED_COMPONENT_COUNTS
from utility import get_cached_vlm_model
from qwen_vl_utils import process_vision_info

# Configure module-specific logger
logger = logging.getLogger(__name__)

# --- Segment plate directly from ndarray ---
def segment_moroccan_plate(plate_image):
    """
    Segment Moroccan plate into 3 regions (left digits, Arabic char, right digits)
    directly from ndarray input. Returns tuple of 3 ndarray regions.
    """
    h, w = plate_image.shape[:2]
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find all character contours (digits + Arabic)
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort contours by x-position
    char_bboxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)  # cw/ch = contour dimensions

        # Keep only valid characters (not separators)
        if cw > 5 and ch > 15:  # Minimum size for actual characters
            char_bboxes.append((x, x + cw, y, y + ch))

    char_bboxes.sort(key=lambda b: b[0])  # Sort left-to-right
    
    # Validate we have 8+ characters (6 digits + Arabic + 2 digit)
    if len(char_bboxes) < 8:
        # Fallback: use heuristic width splits (52%/73%)
        left_x, right_x = int(0.52 * w), int(0.73 * w)
        return plate_image[:, :left_x], plate_image[:, left_x:right_x], plate_image[:, right_x:]
    
    # Get exact boundaries for each region
    left_region = plate_image[:, char_bboxes[0][0]:char_bboxes[5][1]]      # Chars 0-5 (6 digits)
    middle_region = plate_image[:, char_bboxes[6][0]:char_bboxes[6][1]]    # Char 6 (Arabic)
    right_region = plate_image[:, char_bboxes[7][0]:char_bboxes[-1][1]]    # Chars 7+ (right digits)
    
    return left_region, middle_region, right_region

# --- VLM OCR Helper Function (Moroccan plate specific) ---
def extract_with_qari_ocr(image_np, region_type="left"):
    """
    Perform OCR with region-specific prompts and preprocessing.
    region_type: 'left', 'middle', or 'right'
    Format: digits + single Arabic character + digits
    Returns: raw OCR text string
    """
    # Convert OpenCV BGR to PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # Use temporary file with file:// URI (required by process_vision_info)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        pil_img.save(temp_path)

    # Get cached model
    model, processor, device = get_cached_vlm_model()

    # Region-specific prompts
    prompts = {
        "left": "Recognize ONLY the 6 Western digits (0-9). Thin vertical lines are NOT '1'. Output digits only.",
        "middle": "Recognize ONLY the single Arabic character. Output single arabic character only",
        "right": "Recognize ONLY Western digits (0-9). Thin vertical lines are NOT '1'. Output digits only."
    }

    try:
        # Prepare input with image URL
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{temp_path}"},
                {"type": "text", "text": prompts[region_type]}
            ]
        }]

        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50  # Constrained for plate segments
            )
        
        # Trim input tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- OCR on segmented regions with post-processing (adapted from segment_and_extract.py) ---
def ocr_plate_segments(left_region, middle_region, right_region):
    """Perform OCR on segmented plate regions with region-specific processing."""
    # OCR each region with specialized prompts
    left_text = extract_with_qari_ocr(left_region, region_type="left")
    middle_text = extract_with_qari_ocr(middle_region, region_type="middle")
    right_text = extract_with_qari_ocr(right_region, region_type="right")
    
    # Post-processing with plate constraints
    left_clean = ''.join(filter(str.isdigit, left_text))[:6]  # 5-6 digit registration number
    middle_clean = ''.join(c for c in middle_text if '\u0621' <= c <= '\u064A' and c != '\u0640')[:1]  # Single Arabic character
    right_clean = ''.join(filter(str.isdigit, right_text))[:2]  # 1-2 digit regional code

    return left_clean, middle_clean, right_clean

# --- Segmentation-aware OCR for Moroccan plates (replaces original implementation) ---
def run_vlm_ocr_on_plate(plate_image):
    """
    Extract Moroccan plate number using segmentation + region-specific OCR.
    Format: digits + single Arabic character + digits (e.g., '1234ب56')
    Returns: cleaned_plate_text (str) or None on failure
    """
    try:
        # STEP 1: Segment plate into 3 regions directly from ndarray
        left_region, middle_region, right_region = segment_moroccan_plate(plate_image)
        
        # STEP 2: Perform OCR on each segment with region-specific prompts
        left_digits, arabic_char, right_digits = ocr_plate_segments(left_region, middle_region, right_region)
        
        # STEP 3: Concatenate segments into full plate number
        cleaned = f"{left_digits}{arabic_char}{right_digits}"
        
        # Validation: Must have at least 1 digit on each side + Arabic character
        if not (left_digits and arabic_char and right_digits):
            logger.warning(f"Rejected incomplete Moroccan plate: '{cleaned}' (left='{left_digits}', middle='{arabic_char}', right='{right_digits}')")
            return None
            
        return cleaned
    
    except Exception as e:
        logger.error(f"Segmented OCR failed: {type(e).__name__}: {e}")
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

    if plate_image is not None:
        extracted = run_vlm_ocr_on_plate(plate_image)
        if extracted:
            return f"✅ plate number: visible, with number: {extracted}", extracted
        else:
            return "⚠️ plate number: visible but could not be read", None
    else:
        return "⚠️ plate number: visible but could not be read", None

# --- Diagnostic Orchestrators ---
def run_front_diagnostics(components, plate_crop=None):
    diagnostics = []
    
    diagnostics.append(check_mirrors(components.get('mirror', {"count": 0, "confidence": []})))
    diagnostics.append(check_front_lights(components.get('light_front', {"count": 0, "confidence": []})))
    diagnostics.append(check_wipers(components.get('wiper', {"count": 0, "confidence": []})))
    diagnostics.append(check_mirror_top(components.get('mirror_top', {"count": 0, "confidence": []})))
    plate_msg, plate_number = check_plate_number(components.get('plate_number', {"count": 0, "confidence": []}), plate_crop)
    diagnostics.append(plate_msg)
    
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