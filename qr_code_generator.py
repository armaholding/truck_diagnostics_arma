import os
import qrcode
from PIL import Image, ImageDraw, ImageFont
import json
import logging
import base64
from config import (DRIVER_SOURCE_PATH, QR_CODE_PATH, GENERATED_NAMES_JSON, QRCODE_PREFIX,
                    QR_VERSION, BOX_SIZE, BORDER, FONT_SIZE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

def load_generated_names() -> set:
    """Load previously generated names from JSON file."""
    if not os.path.exists(GENERATED_NAMES_JSON):
        return set()
    try:
        with open(GENERATED_NAMES_JSON, 'r', encoding='utf-8') as f:
            names_list = json.load(f)
        if isinstance(names_list, list):
            return {name for name in names_list if isinstance(name, str)}
        else:
            logging.warning(f"Unexpected format in {GENERATED_NAMES_JSON}. Expected list of strings.")
            return set()
    except (json.JSONDecodeError, Exception) as e:
        logging.warning(f"Failed to load {GENERATED_NAMES_JSON}: {e}. Starting fresh.")
        return set()

def save_generated_names(names_set: set) -> None:
    """Persist generated names to JSON as a sorted list."""
    try:
        names_list = sorted(names_set)
        with open(GENERATED_NAMES_JSON, 'w', encoding='utf-8') as f:
            json.dump(names_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save {GENERATED_NAMES_JSON}: {e}")

def generate_qr_code(name: str, font) -> str:
    """
    Generate and save a QR code image for a single name.
    Returns the saved filename (without path).
    """
    # Encode payload: 'arma_driver: <name>' → Base64
    qr_data_raw = QRCODE_PREFIX + name
    qr_data = base64.b64encode(qr_data_raw.encode('utf-8')).decode('ascii')

    # Create QR code instance
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=BOX_SIZE,
        border=BORDER,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)

    # Render QR image
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    qr_width, qr_height = qr_img.size

    # Create composite image with name label below QR
    text_height = FONT_SIZE + 10
    total_height = qr_height + text_height
    final_img = Image.new('RGB', (qr_width, total_height), 'white')
    final_img.paste(qr_img, (0, 0))

    # Center-align name text
    draw = ImageDraw.Draw(final_img)
    text_bbox = draw.textbbox((0, 0), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    x = (qr_width - text_width) // 2
    y = qr_height + 5
    draw.text((x, y), name, fill="black", font=font)

    # Generate safe filename and save
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
    filename = f"{safe_name.replace(' ', '_')}.png"
    output_path = os.path.join(QR_CODE_PATH, filename)
    final_img.save(output_path)

    return filename
    
def main():
    """Main entry point: orchestrate QR code batch generation."""
    os.makedirs(QR_CODE_PATH, exist_ok=True)
    generated_names = load_generated_names()

    # Load font with cross-platform fallback
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()

    # Read input names
    try:
        with open(DRIVER_SOURCE_PATH, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            logging.warning("No names found in 'names.txt'.")
            return
    except FileNotFoundError:
        logging.error(f"File not found: {DRIVER_SOURCE_PATH}")
        return

    # Process each name
    for name in names:
        if name in generated_names:
            logging.info(f"Skipped (already generated): {name}")
            continue

        try:
            filename = generate_qr_code(name, font)
            logging.info(f"Saved: {filename} (label: '{name}')")

            # Record success to avoid reprocessing (crash-safe)
            generated_names.add(name)
            save_generated_names(generated_names)

        except qrcode.exceptions.DataOverflowError:
            logging.error(f"Name too long for QR version {QR_VERSION}: '{name}'. Increase version.")

if __name__ == "__main__":
    print(f"🚀 Generating QR codes from {DRIVER_SOURCE_PATH} (skipping names where qr codes have been generated)...")
    main()
    print("✅ Done!")