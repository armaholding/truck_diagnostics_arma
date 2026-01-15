import os
import glob
import random
from PIL import Image
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
IMAGE_FOLDER = "image_storage"
TARGET_WIDTH = 478
TARGET_HEIGHT = 850
CROP_WIDTH = 478   # Set to <= TARGET_WIDTH
CROP_HEIGHT = 600  # Set to <= TARGET_HEIGHT
CROP_PROBABILITY = 0.5  # 50% chance to crop

def is_valid_jpeg(path: str) -> bool:
    """
    Check if the file is a valid JPEG by inspecting its header (first 2 bytes).
    Returns True only if file starts with JPEG SOI marker: 0xFFD8.
    """
    try:
        with open(path, 'rb') as f:
            return f.read(2) == b'\xff\xd8'
    except Exception:
        return False

def crop_center(img: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    """
    Crops the image from the center to the specified width and height.
    Assumes crop_width <= img.width and crop_height <= img.height.
    """
    img_width, img_height = img.size
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))

def process_image_folders() -> None:
    """Process all valid JPEG images in subdirectories of IMAGE_FOLDER."""
    try:
        all_entries = os.listdir(IMAGE_FOLDER)
    except FileNotFoundError:
        logger.error(f"Directory '{IMAGE_FOLDER}' not found.")
        return
    except Exception as e:
        logger.error(f"Failed to read directory '{IMAGE_FOLDER}': {e}")
        return

    subdirs = [
        os.path.join(IMAGE_FOLDER, d)
        for d in all_entries
        if os.path.isdir(os.path.join(IMAGE_FOLDER, d))
    ]

    if not subdirs:
        logger.warning(f"No subdirectories found in {IMAGE_FOLDER}. Nothing to process.")
        return

    total_images = 0
    cropped_count = 0

    for subdir in subdirs:
        # Collect all possible JPEG file paths (case-insensitive)
        jpg_files = (
            glob.glob(os.path.join(subdir, "*.jpg")) +
            glob.glob(os.path.join(subdir, "*.jpeg")) +
            glob.glob(os.path.join(subdir, "*.JPG")) +
            glob.glob(os.path.join(subdir, "*.JPEG"))
        )

        for img_path in jpg_files:
            # Skip if not a real file or zero-sized
            if not os.path.isfile(img_path):
                logger.debug(f"Skipping non-file: {img_path}")
                continue
            if os.path.getsize(img_path) == 0:
                logger.debug(f"Skipping empty file: {img_path}")
                continue

            # Verify it's a real JPEG (not just .jpg extension)
            if not is_valid_jpeg(img_path):
                logger.debug(f"Skipping non-JPEG (invalid header): {img_path}")
                continue

            total_images += 1
            try:
                with Image.open(img_path) as img:
                    # Double-check format and dimensions
                    if img.format != 'JPEG':
                        logger.debug(f"Skipping non-JPEG (Pillow format={img.format}): {img_path}")
                        continue
                    if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
                        logger.debug(f"Skipping {img_path}: size {img.size} != ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                        continue

                    # Apply random crop with 50% probability
                    if random.random() < CROP_PROBABILITY:
                        cropped_img = crop_center(img, CROP_WIDTH, CROP_HEIGHT)
                        # Save with high fixed quality (avoids 'keep' error)
                        cropped_img.save(img_path, "JPEG", quality=95, optimize=True)
                        cropped_count += 1
                        logger.debug(f"Cropped and saved: {img_path}")

            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}", exc_info=True)

    logger.info(f"✅ Processing complete. Total valid JPEGs: {total_images}, Cropped: {cropped_count}")

if __name__ == "__main__":
    logger.info("🚀 Starting random center-cropping of images in subfolders...")
    process_image_folders()
    logger.info("🎉 Random cropping workflow finished.")