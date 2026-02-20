import cv2
import tempfile
import os
import torch
from PIL import Image
from pathlib import Path
from utility_test import get_cached_vlm_model
from qwen_vl_utils import process_vision_info

# Get script directory for relative path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PLATE_FOLDER = SCRIPT_DIR / "plate_number"  # Images now in "plate_number" folder
DEFAULT_PLATE_IMAGE = PLATE_FOLDER / "moroccan_plate_number3.jpg"


def segment_moroccan_plate(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize grayscale image to consistent dimensions ---
    TARGET_WIDTH, TARGET_HEIGHT = 250, 96
    
    # Choose interpolation based on resize direction for best quality
    if gray.shape[1] > TARGET_WIDTH:  # Downscaling
        interpolation = cv2.INTER_AREA
    else:  # Upscaling
        interpolation = cv2.INTER_LINEAR
    
    gray_resized = cv2.resize(gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=interpolation)

    # Apply Otsu's thresholding to get binary image
    _, binary = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find all character contours (digits + Arabic)
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort contours by x-position
    char_bboxes = []
    for cnt in contours:
        # x, y, w, h = cv2.boundingRect(cnt)  # ✅ w/h = bounding box dimensions
        x, y, cw, ch = cv2.boundingRect(cnt)  # ✅ cw/ch = contour dimensions

        # Keep only valid characters (not separators)
        if cw > 5 and ch > 15:  # Minimum size for actual characters
            # char_bboxes.append((x, x + w, y, y + h))
            char_bboxes.append((x, x + cw, y, y + ch))

    char_bboxes.sort(key=lambda b: b[0])  # Sort left-to-right
    
    # Validate we have 8 characters (6 digits + Arabic + 2 digit)
    if len(char_bboxes) < 8:
        w = img.shape[1]
        left_x, right_x = int(0.52 * w), int(0.73 * w)
        return img[:, :left_x], img[:, left_x:right_x], img[:, right_x:]
    
    # Get exact boundaries for each region
    left_region = img[:, char_bboxes[0][0]:char_bboxes[5][1]]
    middle_region = img[:, char_bboxes[6][0]:char_bboxes[6][1]]
    right_region = img[:, char_bboxes[7][0]:char_bboxes[7][1]]
    
    return left_region, middle_region, right_region

def ocr_with_qwen(image_np, region_type="left"):
    """
    Perform OCR with region-specific prompts and preprocessing.
    region_type: 'left', 'middle', or 'right'
    """
    # Convert OpenCV BGR to PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # Use temporary file with file:// URI (required by older process_vision_info)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        pil_img.save(temp_path)

    # Get cached model
    model, processor, device = get_cached_vlm_model()

    # Region-specific prompts
    prompts = {
        "left": "Recognize ONLY the 6 Western digits (0-9). Thin vertical lines are NOT '1'. Output digits only.",
        "middle": "Recognize ONLY the single Arabic character. Output the arabic character only.",
        "right": "Recognize ONLY the single Western digit (0-9). Thin vertical lines are NOT '1'. Output digits only."
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

def ocr_plate_segments(left_region, middle_region, right_region):
    """Perform OCR on segmented plate regions with region-specific processing."""
    # OCR each region with specialized prompts
    left_text = ocr_with_qwen(left_region, region_type="left")
    middle_text = ocr_with_qwen(middle_region, region_type="middle")
    right_text = ocr_with_qwen(right_region, region_type="right")
    
    # Post-processing with plate constraints
    left_clean = ''.join(filter(str.isdigit, left_text))[:6] # 5 to 6-digit registration number
    middle_clean = ''.join(c for c in middle_text if '\u0621' <= c <= '\u064A' and c != '\u0640')[:1]  # Single Arabic character
    right_clean = ''.join(filter(str.isdigit, right_text))[:2]  # 1 to 2-digit regional code

    return left_clean, middle_clean, right_clean

# Usage example
if __name__ == "__main__":
    # Use DEFAULT_PLATE_IMAGE constant from configuration
    print(f"Processing plate image: {DEFAULT_PLATE_IMAGE}")

    # Step 1: Segment the plate
    left, middle, right = segment_moroccan_plate(DEFAULT_PLATE_IMAGE)
    
    # Step 2: Perform OCR on segments
    left_digits, arabic_char, right_digits = ocr_plate_segments(left, middle, right)
    
    # Save segmented regions
    cv2.imwrite(str(PLATE_FOLDER / f"left.jpg"), left)
    cv2.imwrite(str(PLATE_FOLDER / f"middle.jpg"), middle)
    cv2.imwrite(str(PLATE_FOLDER / f"right.jpg"), right)
    
    # Print results
    print(f"Detected plate: {left_digits} | {arabic_char or 'MISSING'} | {right_digits}")
    print(f"Full plate number: {left_digits}{arabic_char}{right_digits}")