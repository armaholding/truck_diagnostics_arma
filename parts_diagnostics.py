# parts_diagnostics.py
"""Modular diagnostic functions for truck components."""

import logging
import cv2
import re
import os
from PIL import Image
import tempfile
from config import DIAGNOSTIC_THRESHOLD, EXPECTED_COMPONENT_COUNTS
from utility import get_cached_vlm_model

logger = logging.getLogger(__name__)

# --- VLM OCR Helper Function (Moroccan plate specific) ---
def run_vlm_ocr_on_plate(plate_image):
    """
    Extract Moroccan plate number using GLM-OCR VLM.
    Format: digits + single Arabic character + digits (e.g., '1234ب56')
    Returns: (cleaned_plate_text, None) - confidence always None per requirements
    """
    try:
        # Get cached VLM model/processor
        model, processor, device = get_cached_vlm_model()

        # STEP 1: Convert directly to grayscale (skip unnecessary RGB conversion)
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # STEP 2: CLAHE - Contrast enhancement that preserves local details
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # CRITICAL: Prompt tailored for Moroccan plates with Arabic character
        prompt = (
            "Extract ONLY the Moroccan license plate number characters. "
            "Format: digits, followed by exactly ONE Arabic character (like ب ت ج د ه), followed by more digits. "
            "Output the RAW sequence with NO spaces, NO pipes, NO punctuation, NO extra text. "
        )

        # Save to temp file for VLM (requires file path, not PIL object)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir='.') as tmp:
            cv2.imwrite(tmp.name, enhanced)
            image_path = tmp.name

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }]

            # Process and generate
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)
            
            # Remove token_type_ids if present (not needed for this model)
            inputs.pop("token_type_ids", None)

            # Inference with timeout protection via torch.cuda.amp
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,          # Constrained for plate numbers (was 8192)
                do_sample=False,             # Deterministic output
                pad_token_id=processor.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens (after prompt)
            raw_output = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # CLEANING: Remove spaces/punctuation BUT PRESERVE ARABIC CHARACTERS (U+0600-U+06FF)
            # Keep: digits (0-9) + Arabic letters (\u0600-\u06FF)
            cleaned = re.sub(r'[^\d\u0600-\u06FF]', '', raw_output)

            # if not re.search(r'[\u0620-\u064A]', cleaned):  # Arabic letters ONLY (excludes digits ٠-٩)
            #     logger.warning(f"Rejected invalid Moroccan plate (missing Arabic LETTER): '{cleaned}' (raw: '{raw_output}')")
            #     return None

            return cleaned
        
        finally:
            # print("no clean up for checking purposes")  # Comment out cleanup for debugging
            # Cleanup temp file
            if os.path.exists(image_path):
                os.unlink(image_path)

    except Exception as e:
        logger.error(f"GLM-OCR inference failed: {type(e).__name__}: {e}")
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