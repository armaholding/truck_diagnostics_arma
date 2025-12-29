import os
import qrcode
from PIL import Image, ImageDraw, ImageFont
import json

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(BASE_DIR, 'names.txt')
QR_CODE_PATH = os.path.join(BASE_DIR, 'qr_codes')
GENERATED_NAMES_JSON = os.path.join(BASE_DIR, 'generated_qr_names.json')
QR_VERSION = 2  # QR code version defines the resolution
BOX_SIZE = 20
BORDER = 4
FONT_SIZE = 40
QRCODE_PREFIX = "arma_driver: "

def load_generated_names() -> set:
    """Load the set of names that already have QR codes generated."""
    if not os.path.exists(GENERATED_NAMES_JSON):
        return set()
    try:
        with open(GENERATED_NAMES_JSON, 'r', encoding='utf-8') as f:
            names_list = json.load(f)
        if isinstance(names_list, list):
            return set(name for name in names_list if isinstance(name, str))
        else:
            print(f"⚠️  Unexpected format in {GENERATED_NAMES_JSON}. Expected a list of strings.")
            return set()
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Failed to load {GENERATED_NAMES_JSON}: {e}. Starting fresh.")
        return set()

def save_generated_names(names_set: set) -> None:
    """Save the set of generated names to JSON as a sorted list."""
    try:
        # Convert set to sorted list for consistent, readable output
        names_list = sorted(names_set)
        with open(GENERATED_NAMES_JSON, 'w', encoding='utf-8') as f:
            json.dump(names_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Failed to save {GENERATED_NAMES_JSON}: {e}")

def generate_qr_codes():
    """
    Generate QR codes from names.txt, skipping names already recorded in generated_qr_names.json.
    The QR code encodes 'arma_driver: <name>', but only '<name>' is shown on the image.
    """
    os.makedirs(QR_CODE_PATH, exist_ok=True)

    # Load existing generated names
    generated_names = load_generated_names()

    # Load font with fallback
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()

    # Read names
    try:
        with open(TEXT_PATH, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            print("⚠️  No names found in 'names.txt'.")
            return
    except FileNotFoundError:
        print(f"❌ File not found: {TEXT_PATH}")
        return

    # Generate QR codes
    for name in names:
        # Skip if already generated
        if name in generated_names:
            print(f"⏭️  Skipped (already generated): {name}")
            continue

        try:
            qr = qrcode.QRCode(
                version=QR_VERSION,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=BOX_SIZE,
                border=BORDER,
            )
            qr_data = QRCODE_PREFIX + name
            qr.add_data(qr_data)
            qr.make(fit=True)

            qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            qr_width, qr_height = qr_img.size

            # Create canvas with space for text
            text_height = FONT_SIZE + 10
            total_height = qr_height + text_height
            final_img = Image.new('RGB', (qr_width, total_height), 'white')
            final_img.paste(qr_img, (0, 0))

            # Draw centered name (original, without prefix)
            draw = ImageDraw.Draw(final_img)
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x = (qr_width - text_width) // 2
            y = qr_height + 5
            draw.text((x, y), name, fill="black", font=font)

            # Safe filename (based on original name)
            safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
            filename = f"{safe_name.replace(' ', '_')}.png"
            final_img.save(os.path.join(QR_CODE_PATH, filename))

            print(f"Saved: {filename} with QR data '{qr_data}' and label '{name}'")

            # Record successful generation
            generated_names.add(name)
            save_generated_names(generated_names)

        except qrcode.exceptions.DataOverflowError:
            print(f"❌ ERROR: '{name}' too long for QR version {QR_VERSION}. Use higher version.")

def main():
    """Main entry point."""
    print(f"🚀 Generating QR codes from {TEXT_PATH} (skipping names where qr codes have been generated)...")
    generate_qr_codes()
    print("✅ Done!")

if __name__ == "__main__":
    main()