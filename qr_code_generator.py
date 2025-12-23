import os
import qrcode
from PIL import Image, ImageDraw, ImageFont

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(BASE_DIR, 'names.txt')
QR_CODE_PATH = os.path.join(BASE_DIR, 'qr_codes')
QR_VERSION = 2
BOX_SIZE = 20
BORDER = 4
FONT_SIZE = 40

def generate_qr_codes_from_txt():
    """
    Generate QR codes from names.txt, each with the name displayed below the QR image.
    """
    os.makedirs(QR_CODE_PATH, exist_ok=True)

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
        try:
            qr = qrcode.QRCode(
                version=QR_VERSION,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=BOX_SIZE,
                border=BORDER,
            )
            qr.add_data(name)
            qr.make(fit=True)

            qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            qr_width, qr_height = qr_img.size

            # Create canvas with space for text
            text_height = FONT_SIZE + 10
            total_height = qr_height + text_height
            final_img = Image.new('RGB', (qr_width, total_height), 'white')
            final_img.paste(qr_img, (0, 0))

            # Draw centered name
            draw = ImageDraw.Draw(final_img)
            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x = (qr_width - text_width) // 2
            y = qr_height + 5
            draw.text((x, y), name, fill="black", font=font)

            # Safe filename
            safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
            filename = f"{safe_name.replace(' ', '_')}.png"
            final_img.save(os.path.join(QR_CODE_PATH, filename))

            print(f"Saved: {filename} with name '{name}'")

        except qrcode.exceptions.DataOverflowError:
            print(f"❌ ERROR: '{name}' too long for QR version {QR_VERSION}. Use higher version.")

def main():
    """Main entry point."""
    print("🚀 Generating QR codes from 'names.txt'...")
    generate_qr_codes_from_txt()
    print("✅ Done!")

if __name__ == "__main__":
    main()