import os
import qrcode
from PIL import Image, ImageDraw, ImageFont
import json
import logging
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(BASE_DIR, 'names.txt')
QR_CODE_PATH = os.path.join(BASE_DIR, 'qr_codes')
GENERATED_NAMES_JSON = os.path.join(BASE_DIR, 'generated_qr_names.json')
QR_VERSION = 2  # QR version (1–40); 2 ≈ 25×25 modules
BOX_SIZE = 20    # Pixels per QR module
BORDER = 4       # Quiet zone (modules)
FONT_SIZE = 40
QRCODE_PREFIX = "arma_driver: "

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

def generate_qr_codes():
    """Generate QR codes for names in names.txt, skipping already processed entries."""
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
        with open(TEXT_PATH, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            logging.warning("No names found in 'names.txt'.")
            return
    except FileNotFoundError:
        logging.error(f"File not found: {TEXT_PATH}")
        return

    # Process each name
    for name in names:
        if name in generated_names:
            logging.info(f"Skipped (already generated): {name}")
            continue

        try:
            qr = qrcode.QRCode(
                version=QR_VERSION,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=BOX_SIZE,
                border=BORDER,
            )
            # Encode payload: 'arma_driver: <name>' → Base64
            qr_data_raw = QRCODE_PREFIX + name
            qr_data = base64.b64encode(qr_data_raw.encode('utf-8')).decode('ascii')
            qr.add_data(qr_data)
            qr.make(fit=True)

            # Render QR image
            qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            qr_width, qr_height = qr_img.size

            # Create composite image with name label
            text_height = FONT_SIZE + 10
            total_height = qr_height + text_height
            final_img = Image.new('RGB', (qr_width, total_height), 'white')
            final_img.paste(qr_img, (0, 0))

            # Center-align name below QR code
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

            logging.info(f"Saved: {filename} (label: '{name}')")

            # Record success to avoid reprocessing
            generated_names.add(name)
            save_generated_names(generated_names)

        except qrcode.exceptions.DataOverflowError:
            logging.error(f"Name too long for QR version {QR_VERSION}: '{name}'. Increase version.")

def main():
    """Main entry point."""
    print(f"🚀 Generating QR codes from {TEXT_PATH} (skipping names where qr codes have been generated)...")
    generate_qr_codes()
    print("✅ Done!")

if __name__ == "__main__":
    main()