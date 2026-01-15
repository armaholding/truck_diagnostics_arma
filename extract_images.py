import os
import subprocess
import glob
import json
from datetime import timedelta
import logging
import shutil
from typing import Dict, List, Tuple, Any

# This extracts images from video files in a specified folder and saves them as JPEG files.

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
VIDEO_FOLDER = "video_storage"          # Folder containing video chunks
IMAGE_FOLDER = "image_storage"          # Folder to save extracted images  
IMAGE_EXTRACT_JSON = os.path.join(IMAGE_FOLDER, "image_extract_log.json")  # JSON file to track extracted images
IMAGE_INTERVAL = 0.25                   # Interval in seconds between extracted frames
JPEG_QUALITY = 2                        # JPEG quality setting
GPU_HWACCEL = "-hwaccel"                # Hardware acceleration option for GPU
VF_FPS = "fps=1/{interval}"             # Video filter for frame rate

def check_ffmpeg() -> bool:
    """Check that both ffmpeg and ffprobe are available."""
    missing = []
    for tool in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([tool, "-version"],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         check=True)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError):
            missing.append(tool)
   
    if missing:
        logger.critical(f"Required tools not found: {', '.join(missing)}. "
                       "Install FFmpeg package which includes both.")
        return False
   
    logger.info("ffmpeg and ffprobe are available")
    return True

def is_gpu_available_image() -> bool:
    """
    Check if h264_cuvid (NVIDIA GPU decoder) is available in FFmpeg.
    """
    if shutil.which("nvidia-smi") is not None:
        try:
            # Check if nvidia-smi runs successfully
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10
            )
            logger.debug("nvidia-smi check passed.")

            # Check if FFmpeg lists the h264_cuvid decoder
            result = subprocess.run(
                ['ffmpeg', '-decoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and 'h264_cuvid' in result.stdout:
                logger.debug("h264_cuvid found in FFmpeg decoders.")
                return True
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            logger.debug("nvidia-smi or ffmpeg -decoders check failed or timed out.")
            pass

    logger.info("💻 GPU decoder not available, falling back to CPU decoding.")
    return False

def parse_time_string(time_str: str) -> int:
    """Convert time string like '00h00m30s' to seconds"""
    time_str = time_str.lower()
    h, m, s = 0, 0, 0
    if 'h' in time_str:
        h_part, time_str = time_str.split('h', 1)
        h = int(h_part)
    if 'm' in time_str:
        m_part, time_str = time_str.split('m', 1)
        m = int(m_part)
    if 's' in time_str:
        s_part = time_str.split('s', 1)[0]
        s = int(s_part)
    return h * 3600 + m * 60 + s

def seconds_to_hms_ms(seconds: float) -> str:
    """Convert a float representing seconds (e.g., 3661.25) to a string in the format: 'HHhMMmSSsMMM' where MMM = milliseconds (000–999)."""
    total_ms = round(seconds * 1000)  # Round to nearest millisecond
    # Ensure non-negative (robustness)
    total_ms = max(0, int(total_ms))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s{ms:03d}"

def load_image_extract_log() -> Dict[str, Any]:
    """Load the JSON file tracking files that have images extracted."""
    if not os.path.exists(IMAGE_EXTRACT_JSON):
        logger.info("No existing tracking file found. Creating new one.")
        return {}
        
    try:
        with open(IMAGE_EXTRACT_JSON, 'r') as f:
            data = json.load(f)           
        logger.debug(f"Loaded tracking data for {len(data)} videos")
        return data

    except (json.JSONDecodeError, FileNotFoundError):
        logger.error("Tracking file corrupted or unreadable. Starting fresh.")
        return {}
        
def save_image_extract_log(extracted_images: Dict[str, Any]) -> bool:
    """Save the extracted images log to JSON."""
    try:
        with open(IMAGE_EXTRACT_JSON, 'w') as f:
            json.dump(extracted_images, f, indent=2)
        logger.debug("Successfully saved extracted images log")
        return True
    except Exception as e:
        logger.error(f"Failed to save extracted images log: {e}", exc_info=True)
        return False

def get_video_files() -> List[str]:
    """Get list of all video files in chunk storage"""
    try:
        # List all entries in VIDEO_FOLDER
        all_entries = os.listdir(VIDEO_FOLDER)
        
        # Define allowed extensions (lowercase for case-insensitive matching) - this one is appendix
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
        
        # Filter files: must be file (not dir) and match extension
        matched_files = []
        for entry in all_entries:
            filepath = os.path.join(VIDEO_FOLDER, entry)
            if os.path.isfile(filepath):
                ext = os.path.splitext(entry)[1].lower()
                if ext in video_extensions:
                    matched_files.append(filepath)

        # Remove duplicates and sort for consistent processing order
        unique_sorted_files = sorted(list(set(matched_files)))
        logger.info(f"Found {len(unique_sorted_files)} video files in {VIDEO_FOLDER}, sorted by name")
        return unique_sorted_files
    
    except Exception as e:
        logger.error(f"Failed to read directory '{VIDEO_FOLDER}': {e}", exc_info=True)
        return []

def rename_and_cleanup_frames(output_folder: str, original_base_name: str) -> List[str]:
    """Rename frames with timestamped filenames based on original filename (with millisecond precision)."""
    try:
        frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))
        if not frame_files:
            logger.warning("No frames found to rename.")
            return []
                
        # Parse base name and start time from original filename
        parts = original_base_name.split('_')
        if len(parts) >= 2 and '-' in parts[-1]:
            base_name = '_'.join(parts[:-1])
            time_part = parts[-1]
            # Parse start time (assumes whole seconds in input filename)
            if '-' in time_part:
                start_str, _ = time_part.split('-', 1)
                start_seconds = parse_time_string(start_str)
            else:
                start_seconds = 0
        else:
            base_name = original_base_name
            start_seconds = 0
            
        logger.debug(f"Base name: {base_name}, Start time: {seconds_to_hms_ms(start_seconds)}")

        # Rename each frame with high-precision timestamped filenames
        image_filenames = []
        for idx, old_path in enumerate(frame_files):
            if not os.path.exists(old_path):
                continue

            frame_start = start_seconds + idx * IMAGE_INTERVAL
            frame_end = start_seconds + (idx + 1) * IMAGE_INTERVAL
            
            # Use millisecond-precision formatter
            start_str = seconds_to_hms_ms(frame_start)
            end_str = seconds_to_hms_ms(frame_end)            
            new_filename = f"{base_name}_{start_str}-{end_str}.jpg"
            new_path = os.path.join(output_folder, new_filename)
            
            if os.path.exists(new_path):
                logger.debug(f"Skipped: {new_filename} already exists")
                image_filenames.append(new_filename)
                continue

            try:
                os.rename(old_path, new_path)
                logger.debug(f"Renamed: {os.path.basename(old_path)} → {new_filename}")
                image_filenames.append(new_filename)
            except Exception as e:
                logger.error(f"Failed to rename {old_path} → {new_path}: {e}")

        logger.info(f"Successfully renamed {len(image_filenames)} frames")
        return image_filenames

    except Exception as e:
        logger.error(f"Frame renaming failed: {e}", exc_info=True)
        return []

def extract_frames_cpu_ffmpeg(input_path: str, output_pattern: str) -> bool:
    """Extract frames using FFmpeg with CPU decoding."""
    logger.info(f"💻 Extracting frames with CPU fallback: {input_path}")
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', VF_FPS.format(interval=IMAGE_INTERVAL), # Video filter for frame rate
        '-q:v', str(JPEG_QUALITY),       # Set JPEG quality to 8 (50% quality)
        '-start_number', '0',       # Ensure numbering starts at 0
        '-y',                       # Overwrite output files
        output_pattern
    ]
    try:
        logger.debug(f"Running FFmpeg (CPU) command: {' '.join(cmd)}")
        # Use check=True to raise CalledProcessError on failure
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL, # Suppress normal output like ffmpeg-python's quiet=True
            stderr=subprocess.PIPE,    # Capture errors
            text=True                  # Decode bytes to string
        )
        logger.info("✅ Frame extraction (CPU) completed")
        return True
    except subprocess.CalledProcessError as e:
        # FFmpeg failed (non-zero exit code)
        stderr_output = e.stderr if e.stderr else "No stderr output captured"
        logger.error(f"❌ FFmpeg (CPU) failed for {input_path}: {stderr_output}")
        return False
    except Exception as e:
        # Unexpected error (e.g., timeout, file permission)
        logger.error(f"⚠️ Unexpected error during FFmpeg (CPU) execution for {input_path}: {e}", exc_info=True)
        return False
    
def extract_frames_ffmpeg_gpu(video_path: str) -> Tuple[bool, List[str]]:
    """Extract frames using FFmpeg with NVIDIA GPU decoding, falling back to CPU."""
    logger.info(f"Attempting GPU-accelerated frame extraction from: {video_path}")
    filename = os.path.basename(video_path) # e.g., 'video_00h01m00s-00h01m30s.mp4'

    # Validate input
    if not os.path.exists(video_path):
        logger.error(f"Input file does not exist: {video_path}")
        return False, []

    # Quick duration check to skip if video < 1 second
    try:
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(
            duration_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10  # Prevent hanging on corrupt files
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                duration_sec = float(result.stdout.strip())
                if duration_sec < IMAGE_INTERVAL:
                    logger.info(f"⏭️ Skipping {filename}: duration ({duration_sec:.2f}s) < 1 second")
                    return True, []  # Treat as "successfully skipped"
            except ValueError:
                logger.warning(f"Could not parse duration for {filename}, proceeding anyway")
        else:
            logger.warning(f"Could not determine duration for {filename}, proceeding anyway")
    except subprocess.TimeoutExpired:
        logger.warning(f"Duration check timed out for {filename}, proceeding anyway")
    except Exception as e:
        logger.warning(f"Error during duration check for {filename}: {e}, proceeding anyway")

    # Generate output folder path inside image_storage
    base_name = os.path.splitext(filename)[0] # e.g., 'video_00h01m00s-00h01m30s'
    output_folder = os.path.join(IMAGE_FOLDER, base_name)
    try:
        os.makedirs(output_folder, exist_ok=True)
        logger.debug(f"Output folder created: {output_folder}")
    except Exception as e:
        logger.error(f"Failed to create output folder {output_folder}: {e}")
        return False, []

    output_pattern = os.path.join(output_folder, "frame_%05d.jpg")

    # Build Base FFmpeg Command
    base_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', VF_FPS.format(interval=IMAGE_INTERVAL), # Video filter for frame rate, set IMAGE_INTERVAL = 1 frame every N seconds
        '-q:v', str(JPEG_QUALITY),                          # Set JPEG quality to 8 (50% quality)
        '-start_number', '0',                          # Ensure numbering starts at 0
        '-y',                                          # Overwrite output files
        output_pattern
    ]

    # Check for GPU and Prepare GPU Acceleration Options
    use_gpu = is_gpu_available_image()
    cmd_to_run = base_cmd
    if use_gpu:
        logger.info("💻 GPU-accelerated decoding with CUDA")
        hwaccel_options = [
            GPU_HWACCEL, 'cuda',
            # Optional: explicitly specify decoder/format if needed
            # '-c:v', GPU_DECODER, # Example for H.264
            # HWACCEL_OUTPUT_FORMAT, 'cuda'
        ]
        try:
            i_index = base_cmd.index('-i')
        except ValueError:
            logger.error("Internal Error: '-i' not found in base FFmpeg command.")
            return False, []
        cmd_to_run = base_cmd[:i_index] + hwaccel_options + base_cmd[i_index:]
        # else: cmd_to_run remains base_cmd
    
    # Initialize image extraction success parameter
    extraction_succeeded = False

    # Attempt image extraction (GPU or CPU fallback)
    try:
        logger.debug(f"Running FFmpeg command: {' '.join(cmd_to_run)}")
        subprocess.run(
            cmd_to_run,
            check=True,
            stdout=subprocess.DEVNULL, # Suppress normal output
            stderr=subprocess.PIPE,    # Capture errors
            text=True
        )
        extraction_succeeded = True
        logger.info(f"✅ Frame extraction completed ({'GPU' if use_gpu else 'CPU'}): {output_folder}")
    
    # --- Handle FFmpeg Failure (CalledProcessError)
    except subprocess.CalledProcessError as e:
        # If GPU was attempted and failed, try fallback to CPU
        if use_gpu:
            logger.warning("GPU frame extraction failed, falling back to CPU decoding")
            extraction_succeeded = extract_frames_cpu_ffmpeg(video_path, output_pattern)
            if extraction_succeeded:
                logger.info("✅ Frame extraction succeeded with CPU fallback")
            else:
                stderr_msg = e.stderr.strip() if e.stderr else "No stderr"
                logger.error(f"Frame extraction failed for {filename} (GPU and CPU): {stderr_msg}", exc_info=False)
        else:
            stderr_msg = e.stderr.strip() if e.stderr else "No stderr"
            logger.error(f"Frame extraction failed for {filename}: {stderr_msg}", exc_info=False)
    except Exception as e:
        logger.error(f"⚠️ Unexpected error during FFmpeg execution: {e}", exc_info=True)
        return False, []

    # --- Unified post-processing: only if extraction succeeded
    if extraction_succeeded:
        original_base_name = os.path.splitext(filename)[0]
        image_filenames = rename_and_cleanup_frames(output_folder, original_base_name)
        if not image_filenames:
            logger.warning(f"No images were generated or renamed for {filename}")
            return True, []
        logger.info(f"✅ Frame renaming completed: {len(image_filenames)} images")
        return True, image_filenames
    else:
        return False, []
                                
def main() -> None:
    """Main function for image extraction workflow."""
    logger.info("🚀 Starting image extraction workflow")

    # Ensure source directory exists
    if not os.path.exists(VIDEO_FOLDER):
        logger.error(f"Input folder '{VIDEO_FOLDER}' does not exist.")
        return

    # Ensure output directory exists    
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # Check dependencies
    if not check_ffmpeg():
        logger.error("Cannot proceed without FFmpeg.")
        return
    
    # Load extracted data
    data = load_image_extract_log()
    logger.info(f"📄 Found {len(data)} previously processed videos")
    
    # Get list of video files (sorted) to process
    video_files = get_video_files()
    if not video_files:
        logger.warning("No video files found in chunk storage. Nothing to process.")
        return
        
    # Process each video
    for video_path in video_files:
        filename = os.path.basename(video_path)

        # Check if already extracted
        if filename in data:
            logger.info(f"⏭️ Skipping already processed: {filename}")
            continue
   
        # Extract frames of only new videos
        success, image_filenames = extract_frames_ffmpeg_gpu(video_path)
        if success:        
            # Update shared state and save log
            data[filename] = {"images": image_filenames}
            if not save_image_extract_log(data):
                logger.error("❌ Failed to update tracking log after successful extraction")
                # Continue to next file even if log save failed
            logger.info(f"✅ Completed processing: {filename} → {len(image_filenames)} images")
        else:
            logger.error(f"❌ Failed to extract images from: {filename}")

    logger.info("✅ Image extraction workflow complete.")

if __name__ == "__main__":
    main()
    logger.info("🎉 Application successfully extracted frames from video chunks")