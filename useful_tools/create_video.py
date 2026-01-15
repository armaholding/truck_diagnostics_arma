import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
import re

FILENAME_PREFIX = "frame"

def normalize_image_sizes(image_folder, target_size=(1920, 1080), output_folder=None, filename_prefix=None):
    """
    Resize and pad all JPEG images in a folder to a uniform target size,
    preserving aspect ratio. Filter by filename prefix.
    Saves normalized images to output_folder.
    If output_folder is None, overwrites original images (not recommended).
    
    Args:
        image_folder (str or Path): Folder containing input JPEGs.
        target_size (tuple): (width, height) for output images.
        output_folder (str or Path, optional): Where to save normalized images.
        filename_prefix (str, optional): Only process files starting with this prefix (e.g., "frame").
    
    Returns:
        Path: Path to the folder containing normalized images.
    """
    image_folder = Path(image_folder).resolve()
    if output_folder is None:
        # Create a temporary directory to avoid overwriting originals
        output_folder = Path(tempfile.mkdtemp(prefix="normalized_images_", dir=image_folder.parent))
    else:
        output_folder = Path(output_folder).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)

    # Gather all .jpg and .jpeg files
    all_images = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.jpeg"))

    # Filter by prefix if specified
    if filename_prefix is not None:
        filtered_images = [
            p for p in all_images
            if p.stem.startswith(filename_prefix) and p.stem[len(filename_prefix):].isdigit()
        ]
        if not filtered_images:
            raise ValueError(f"No JPEG images found in {image_folder} with prefix '{filename_prefix}' followed by a number (e.g., '{filename_prefix}1.jpg').")
        # Sort numerically by the number after the prefix
        def extract_number(path):
            num_str = path.stem[len(filename_prefix):]
            return int(num_str) if num_str.isdigit() else float('inf')
        images = sorted(filtered_images, key=extract_number)
    else:
        images = all_images

    if not images:
        raise ValueError(f"No JPEG images found in {image_folder}")
    
    print(f"🖼️ Normalizing {len(images)} images to {target_size[0]}x{target_size[1]}...")
    for img_path in images:
        with Image.open(img_path) as img:
            # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize and pad to target size (preserves aspect ratio)
            img = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

            # Save to output folder with same name
            output_path = output_folder / img_path.name
            img.save(output_path, "JPEG", quality=95)

    print(f"✅ Images normalized and saved to: {output_folder}")
    return output_folder


def create_video_from_images(
    image_folder,
    output_video,
    duration_per_image=3,
    fps=1,
    target_resolution=(1920, 1080),
    normalize_first=True,
    filename_prefix=None
):
    """
    Create an MP4 video from JPEG images using FFmpeg.
    
    Args:
        image_folder (str): Path to folder containing JPEG images.
        output_video (str): Output MP4 video file path.
        duration_per_image (int): Display time per image in seconds.
        fps (int): Video frame rate (1 fps is sufficient for static images).
        target_resolution (tuple): (width, height) to normalize images to.
        normalize_first (bool): If True, normalize image sizes before video creation.
        filename_prefix (str, optional): Only use images starting with this prefix (e.g., "frame").
    """
    image_folder = Path(image_folder).resolve()

    # Optionally normalize image sizes first
    if normalize_first:
        normalized_folder = normalize_image_sizes(
            image_folder,
            target_size=target_resolution,
            filename_prefix=filename_prefix  # Pass prefix down
        )
        temp_normalized = True
    else:
        # Gather and filter images directly
        all_images = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.jpeg"))
        if filename_prefix is not None:
            filtered_images = [
                p for p in all_images
                if p.stem.startswith(filename_prefix) and p.stem[len(filename_prefix):].isdigit()
            ]
            if not filtered_images:
                raise ValueError(f"No JPEG images found in {image_folder} with prefix '{filename_prefix}' followed by a number (e.g., '{filename_prefix}1.jpg').")
            def extract_number(path):
                num_str = path.stem[len(filename_prefix):]
                return int(num_str) if num_str.isdigit() else float('inf')
            images = sorted(filtered_images, key=extract_number)
        else:
            images = all_images

        if not images:
            raise ValueError(f"No JPEG images found in {image_folder}")
        normalized_folder = image_folder
        temp_normalized = False

    try:
        # Reuse the image list if already filtered (when normalize_first=False)
        if not normalize_first:
            pass  # `images` already defined above
        else:
            # When normalize_first=True, `normalized_folder` has all needed images
            all_images = sorted(normalized_folder.glob("*.jpg")) + sorted(normalized_folder.glob("*.jpeg"))
            if filename_prefix is not None:
                filtered_images = [
                    p for p in all_images
                    if p.stem.startswith(filename_prefix) and p.stem[len(filename_prefix):].isdigit()
                ]
                def extract_number(path):
                    num_str = path.stem[len(filename_prefix):]
                    return int(num_str) if num_str.isdigit() else float('inf')
                images = sorted(filtered_images, key=extract_number)
            else:
                images = all_images

        if not images:
            raise ValueError(f"No JPEG images found in {normalized_folder}")
        
        # Create FFmpeg input text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_file = f.name
            for img in images:
                abs_path = str(img.resolve()).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {duration_per_image}\n")
            # Repeat last frame to avoid truncation
            abs_path = str(images[-1].resolve()).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', input_file,
            '-vf', f'fps={fps}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(Path(output_video).resolve())
        ]

        # Run FFmpeg
        print("🎥 Creating video with FFmpeg...")
        subprocess.run(cmd, check=True)
        print(f"✅ Video saved to: {output_video}")

    finally:
        # Clean up
        os.unlink(input_file)
        if temp_normalized and normalize_first:
            import shutil
            shutil.rmtree(normalized_folder, ignore_errors=True)


# Example usage
if __name__ == "__main__":
    create_video_from_images(
        image_folder="../YOLO_Image_Analysis/trash_test",
        output_video="trash_video.mp4",
        duration_per_image=3,
        fps=1,
        target_resolution=(1920, 1080),
        normalize_first=True,
        filename_prefix=FILENAME_PREFIX  # Use the constant
    )