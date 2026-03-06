# utility.py
"""Pure utility functions for truck diagnostic system"""

import os
import sys
import shutil
import re
import cv2
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import configuration constants
from config import (
    DIAGNOSTICS_PATH, CROPPED_PARTS_PATH, YOLO_MODEL_PATH, VLM_MODEL_PATH,
    SAVE_DEBUG_CROPS_LIGHT, SAVE_DEBUG_CROPS_WIPER,
    BLUE, RED, ORANGE, GREEN, CYAN, RESET
)
    
# Configure module-specific logger
logger = logging.getLogger(__name__)

# Load environment variables and validate HF token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable not set. "
        "Get your token from https://huggingface.co/settings/tokens and set it in .env file"
    )

# --- GPU/CPU Detection ---
def get_device():
    """
    Automatically select GPU if available, otherwise fall back to CPU.
    
    Returns:
        str: Device identifier ('cuda' if GPU available, 'cpu' otherwise)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")
    return device

# --- YOLO Model Caching Function ---
_yolo_model = None
_yolo_device = None

def get_cached_yolo_model():
    """
    Get cached YOLO model instance to avoid reloading between frames.
    
    Returns:
        tuple: (model, device)
            - model: Cached YOLO model instance
            - device: Device where model is loaded ('cuda' or 'cpu')
    """
    global _yolo_model, _yolo_device
    if _yolo_model is None:
        _yolo_device = get_device()
        _yolo_model = YOLO(YOLO_MODEL_PATH).to(_yolo_device)
        logger.info(f"YOLO model cached on {_yolo_device} - will reuse for all frames")
    return _yolo_model, _yolo_device

# --- Qari-OCR Model Caching (mirrors YOLO caching pattern) ---
_qwen_model = None
_qwen_processor = None
_qwen_device = None

def get_cached_vlm_model():
    """
    Get cached Qari-OCR model instance to avoid reloading between frames.
    
    Returns:
        tuple: (model, processor, device)
            - model: Cached Qwen2VL model instance
            - processor: Associated processor for input preparation
            - device: Device where model is loaded ('cuda' or 'cpu')
    """
    global _qwen_model, _qwen_processor, _qwen_device
    if _qwen_model is None:
        _qwen_device = get_device()
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH,
            torch_dtype=torch.bfloat16 if _qwen_device == "cuda" else torch.float32,
            device_map="auto"
        )
        _qwen_processor = AutoProcessor.from_pretrained(
            VLM_MODEL_PATH
            )
    return _qwen_model, _qwen_processor, _qwen_device

# --- Object Detection Helper Functions ---
def get_available_classes(model):
    """
    Retrieve the list of available classes from the YOLO model.
    
    Args:
        model: YOLO model instance
    
    Returns:
        dict: Dictionary mapping class indices to class names
    """
    return model.names

# --- Monthly Folder Extraction ---
def _extract_monthly_folder(timestamp_str: str) -> str:
    """
    Extract YYYYMM from diagnosis timestamp string in format "YYYY-MM-DD HH:MM:SS".
    
    Returns:
        str: "YYYYMM" string. On parse failure, returns current month as fallback.
    """
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y%m")
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}' for monthly folder: {e}. Using current month.")
        return datetime.now().strftime("%Y%m")

# --- Cropping and Saving Functions ---
def save_cropped_detections(
    image, 
    component_data, 
    frame_id=None,
    diagnosis_timestamp: str = None,
    parts_session_folder: str = None
):
    """
    Save cropped regions of all detected objects for validation and debugging.
    
    Args:
        image: Source image/frame as NumPy array
        component_data: Detection data dictionary grouped by component type
        frame_id (int | None): Frame number for filename (None for single images)
        diagnosis_timestamp (str): Session start timestamp in "YYYY-MM-DD HH:MM:SS" format
        parts_session_folder (str): Session folder name like "parts_20260206_143522_123"
    
    Returns:
        None: Saves cropped images to CROPPED_PARTS_PATH directory
    """
    if image is None:
        logger.warning("Could not load image/frame for cropping")
        return
        
    # Base directory always exists conceptually
    base_output_dir = CROPPED_PARTS_PATH
    
    # Determine actual output path based on session context
    if diagnosis_timestamp and parts_session_folder:
        # NEW HIERARCHY: cropped_parts/YYYYMM/parts_YYYYMMDD_HHMMSS_mmm/
        monthly_folder = _extract_monthly_folder(diagnosis_timestamp)
        output_dir = os.path.join(base_output_dir, monthly_folder, parts_session_folder)
        logger.debug(f"Saving cropped parts to session folder: {output_dir}")
    else:
        # FALLBACK: Flat structure (backward compatibility)
        output_dir = base_output_dir
        logger.debug("Using flat cropped_parts directory (no session context provided)")
    
    # Create nested directories (idempotent)
    os.makedirs(output_dir, exist_ok=True)

    h, w = image.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_index = 0

    for class_name, data in component_data.items():
        for i, (conf, bbox) in enumerate(zip(data['confidences'], data['boxes'])):
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
            # Clamp coordinates to image boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Format: {timestamp}[_frame{frame_id}]_{class}_{confidence:.2f}_{index}.jpg
            if frame_id is not None:
                filename = f"{timestamp}_frame{frame_id}_{class_name}_{conf:.2f}_{crop_index}.jpg"
            else:
                filename = f"{timestamp}_{class_name}_{conf:.2f}_{crop_index}.jpg"
                
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop)
            crop_index += 1
            logger.debug(f"Cropped part saved: {filepath}")

# --- Intermediate Diagnostics Cleanup ---
def cleanup_intermediate_diagnostics():
    """
    Interactive cleanup of intermediate diagnostic folders after processing completes.
    
    Scans diagnostics/YYYYMM/ directories for int_diagnostics_* folders, presents numbered
    list to user, and deletes selected folders after explicit confirmation.
    
    Safety guarantees:
      - ONLY deletes folders matching 'int_diagnostics_*' pattern
      - NEVER deletes final reports (final_*.json) or monthly folders
      - Skips entirely in non-interactive environments
      - Requires explicit 'CONFIRM' for deletion
      - Validates all paths before deletion to prevent path traversal
    
    Returns:
        bool: True if cleanup was attempted (regardless of success), False if skipped
    """
    # Skip if not interactive terminal (prevent blocking automated pipelines)
    if not sys.stdin.isatty():
        logger.info("Skipping intermediate diagnostics cleanup (non-interactive environment)")
        return False
    
    # Skip if diagnostics directory doesn't exist
    if not os.path.exists(DIAGNOSTICS_PATH):
        logger.info("No diagnostics directory found - skipping cleanup")
        return False
    
    # Scan for intermediate diagnostic folders across all monthly folders
    monthly_folders = []
    intermediate_folders = []  # List of (month_str, folder_path, folder_name) tuples
    
    # Find all YYYYMM monthly folders
    for entry in os.scandir(DIAGNOSTICS_PATH):
        if entry.is_dir() and re.match(r'^\d{6}$', entry.name):  # YYYYMM pattern
            monthly_folders.append(entry.name)
    
    if not monthly_folders:
        logger.info("No monthly diagnostic folders found - skipping cleanup")
        return False
    
    # Within each monthly folder, find int_diagnostics_* folders
    for month in sorted(monthly_folders):
        month_path = os.path.join(DIAGNOSTICS_PATH, month)
        try:
            for entry in os.scandir(month_path):
                if entry.is_dir() and entry.name.startswith('int_diagnostics_'):
                    # Get folder stats for display
                    try:
                        file_count = sum(1 for _ in Path(entry.path).rglob('*.json'))
                        folder_size = sum(f.stat().st_size for f in Path(entry.path).rglob('*') if f.is_file())
                        size_mb = folder_size / (1024 * 1024)
                        size_display = f"{size_mb:.1f} MB" if folder_size > 0 else "0 KB"
                    except Exception:
                        file_count = 0
                        size_display = "unknown"
                    
                    intermediate_folders.append((month, entry.path, entry.name, file_count, size_display))
        except FileNotFoundError:
            continue  # Monthly folder disappeared during scan (unlikely but safe)
    
    if not intermediate_folders:
        logger.info("No intermediate diagnostic folders found - skipping cleanup")
        return False
    
    # Present numbered list grouped by month
    logger.info(f"\n{CYAN}🗑️  INTERMEDIATE DIAGNOSTICS CLEANUP{RESET}")
    logger.info(f"Found {len(intermediate_folders)} intermediate diagnostic folder(s) across {len(set(m for m,_,_,_,_ in intermediate_folders))} month(s):\n")
    
    current_month = None
    folder_map = {}  # Maps number -> (month, folder_path, folder_name)
    counter = 1
    
    for month, folder_path, folder_name, file_count, size_display in sorted(intermediate_folders, key=lambda x: x[1]):
        if month != current_month:
            current_month = month
            # Convert YYYYMM to human-readable month name
            try:
                month_dt = datetime.strptime(month, "%Y%m")
                month_name = month_dt.strftime("%B %Y")
            except:
                month_name = f"{month[:4]}-{month[4:]}"
            logger.info(f"{BLUE}[{month}] {month_name}:{RESET}")
        
        logger.info(f"  {counter}) {folder_name} ({file_count} file(s), {size_display})")
        folder_map[counter] = (month, folder_path, folder_name)
        counter += 1
    
    logger.info(f"\n{ORANGE}⚠️  WARNING:{RESET} This will PERMANENTLY DELETE selected folders and their contents.")
    logger.info(f"   Final diagnostic reports ({GREEN}final_*.json{RESET}) will NOT be affected.\n")
    
    # Get user selection
    selection_input = input(f"Enter folder numbers to delete (e.g., {GREEN}\"1,3,5\"{RESET}), {GREEN}\"all\"{RESET} for all folders, or press Enter to skip: ").strip()
    
    # Handle empty selection
    if not selection_input:
        logger.info(f"\n{BLUE}ℹ️  No folders selected for deletion - cleanup skipped.{RESET}\n")
        return True
    
    # Parse selection
    selected_numbers = []
    if selection_input.lower() == 'all':
        selected_numbers = list(folder_map.keys())
        logger.info(f"\n{ORANGE}⚠️  BULK DELETION MODE:{RESET} You selected ALL {len(selected_numbers)} intermediate diagnostic folders.")
        extra_confirm = input(f"Type {RED}'CONFIRM ALL'{RESET} to proceed with bulk deletion, or anything else to cancel: ").strip()
        if extra_confirm != 'CONFIRM ALL':
            logger.info(f"\n{BLUE}ℹ️  Bulk deletion cancelled - cleanup skipped.{RESET}\n")
            return True
    else:
        # Parse comma-separated numbers
        try:
            parts = [p.strip() for p in selection_input.split(',')]
            selected_numbers = [int(p) for p in parts if p.isdigit()]
            if not selected_numbers:
                logger.info(f"\n{RED}❌ Invalid selection - no valid numbers found. Cleanup skipped.{RESET}\n")
                return True
        except ValueError:
            logger.info(f"\n{RED}❌ Invalid selection format. Cleanup skipped.{RESET}\n")
            return True
    
    # Validate selections
    invalid = [n for n in selected_numbers if n not in folder_map]
    if invalid:
        logger.info(f"\n{RED}❌ Invalid folder number(s): {', '.join(map(str, invalid))}. Cleanup skipped.{RESET}\n")
        return True
    
    # Build deletion preview
    folders_to_delete = [folder_map[n] for n in selected_numbers]
    logger.info(f"\n{ORANGE}✅ PREVIEW:{RESET} You are about to delete {len(folders_to_delete)} folder(s):")
    for month, folder_path, folder_name in folders_to_delete:
        # Show relative path for safety
        rel_path = os.path.relpath(folder_path, start=DIAGNOSTICS_PATH)
        logger.info(f"   - {rel_path}/")
    
    # Final confirmation
    confirm = input(f"\nType {RED}'CONFIRM'{RESET} to proceed with deletion, or anything else to cancel: ").strip()
    if confirm != 'CONFIRM':
        logger.info(f"\n{BLUE}ℹ️  Deletion cancelled - cleanup skipped.{RESET}\n")
        return True
    
    # Perform deletion with safety validations
    success_count = 0
    failure_count = 0
    
    for month, folder_path, folder_name in folders_to_delete:
        try:
            # Safety validation 1: Must be within diagnostics directory
            folder_path_abs = os.path.abspath(folder_path)
            diagnostics_abs = os.path.abspath(DIAGNOSTICS_PATH)
            if not folder_path_abs.startswith(diagnostics_abs + os.sep):
                raise ValueError(f"Path traversal attempt blocked: {folder_path}")
            
            # Safety validation 2: Must match int_diagnostics_ pattern
            if not os.path.basename(folder_path).startswith('int_diagnostics_'):
                raise ValueError(f"Refusing to delete non-intermediate folder: {folder_path}")
            
            # Safety validation 3: Must be a directory
            if not os.path.isdir(folder_path):
                raise ValueError(f"Not a directory: {folder_path}")
            
            # Perform deletion
            shutil.rmtree(folder_path)
            logger.info(f"Deleted intermediate diagnostics folder: {folder_path}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to delete folder {folder_path}: {e}")
            failure_count += 1
    
    # Report results
    if success_count > 0:
        logger.info(f"\n{GREEN}🗑️  Successfully deleted {success_count} intermediate diagnostic folder(s).{RESET}")
    if failure_count > 0:
        logger.info(f"{RED}⚠️  Failed to delete {failure_count} folder(s) - see logs for details.{RESET}")
    if success_count == 0 and failure_count == 0:
        logger.info(f"{BLUE}ℹ️  No folders deleted.{RESET}")
    
    logger.info("")  # Blank line for readability
    return True

# --- Saved Cropped Parts Cleanup ---
def cleanup_saved_crops():
    """
    Interactive cleanup of saved cropped parts folders after processing completes.
    
    Scans cropped_parts/YYYYMM/ directories for parts_* folders, presents numbered
    list to user, and deletes selected folders after explicit confirmation.
    
    Safety guarantees:
      - ONLY deletes folders matching 'parts_*' pattern
      - NEVER deletes individual .jpg files (legacy validation images preserved)
      - NEVER deletes monthly folders (YYYYMM/)
      - Skips entirely in non-interactive environments
      - Requires explicit 'CONFIRM' for deletion
      - Validates all paths before deletion to prevent path traversal
    
    Returns:
        bool: True if cleanup was attempted (regardless of success), False if skipped
    """
    # Skip if not interactive terminal (prevent blocking automated pipelines)
    if not sys.stdin.isatty():
        logger.info("Skipping saved crops cleanup (non-interactive environment)")
        return False
    
    # Skip if cropped_parts directory doesn't exist
    if not os.path.exists(CROPPED_PARTS_PATH):
        logger.info("No cropped_parts directory found - skipping cleanup")
        return False
    
    # Scan for parts session folders across all monthly folders
    monthly_folders = []
    parts_folders = []  # List of (month_str, folder_path, folder_name) tuples
    
    # Find all YYYYMM monthly folders
    for entry in os.scandir(CROPPED_PARTS_PATH):
        if entry.is_dir() and re.match(r'^\d{6}$', entry.name):  # YYYYMM pattern
            monthly_folders.append(entry.name)
    
    if not monthly_folders:
        logger.info("No monthly cropped parts folders found - skipping cleanup")
        return False
    
    # Within each monthly folder, find parts_* folders
    for month in sorted(monthly_folders):
        month_path = os.path.join(CROPPED_PARTS_PATH, month)
        try:
            for entry in os.scandir(month_path):
                if entry.is_dir() and entry.name.startswith('parts_'):
                    # Get folder stats for display (count .jpg files only)
                    try:
                        image_files = list(Path(entry.path).rglob('*.jpg'))
                        file_count = len(image_files)
                        folder_size = sum(f.stat().st_size for f in image_files)
                        size_mb = folder_size / (1024 * 1024)
                        size_display = f"{size_mb:.1f} MB" if folder_size > 0 else "0 KB"
                    except Exception:
                        file_count = 0
                        size_display = "unknown"
                    
                    parts_folders.append((month, entry.path, entry.name, file_count, size_display))
        except FileNotFoundError:
            continue  # Monthly folder disappeared during scan (unlikely but safe)
    
    if not parts_folders:
        logger.info("No saved cropped parts session folders found - skipping cleanup")
        return False
    
    # Present numbered list grouped by month
    logger.info(f"\n{CYAN}🖼️  SAVED CROPPED PARTS CLEANUP{RESET}")
    logger.info(f"Found {len(parts_folders)} cropped parts session folder(s) across {len(set(m for m,_,_,_,_ in parts_folders))} month(s):\n")
    
    current_month = None
    folder_map = {}  # Maps number -> (month, folder_path, folder_name)
    counter = 1
    
    for month, folder_path, folder_name, file_count, size_display in sorted(parts_folders, key=lambda x: x[1]):
        if month != current_month:
            current_month = month
            # Convert YYYYMM to human-readable month name
            try:
                month_dt = datetime.strptime(month, "%Y%m")
                month_name = month_dt.strftime("%B %Y")
            except:
                month_name = f"{month[:4]}-{month[4:]}"
            logger.info(f"{BLUE}[{month}] {month_name}:{RESET}")
        
        logger.info(f"  {counter}) {folder_name} ({file_count} image(s), {size_display})")
        folder_map[counter] = (month, folder_path, folder_name)
        counter += 1
    
    logger.info(f"\n{ORANGE}⚠️  WARNING:{RESET} This will PERMANENTLY DELETE selected folders and their contents.")
    logger.info(f"   Legacy validation images ({GREEN}individual .jpg files{RESET}) will NOT be affected.\n")
    
    # Get user selection
    selection_input = input(f"Enter folder numbers to delete (e.g., {GREEN}\"1,3,5\"{RESET}), {GREEN}\"all\"{RESET} for all folders, or press Enter to skip: ").strip()
    
    # Handle empty selection
    if not selection_input:
        logger.info(f"\n{BLUE}ℹ️  No folders selected for deletion - cleanup skipped.{RESET}\n")
        return True
    
    # Parse selection
    selected_numbers = []
    if selection_input.lower() == 'all':
        selected_numbers = list(folder_map.keys())
        logger.info(f"\n{ORANGE}⚠️  BULK DELETION MODE:{RESET} You selected ALL {len(selected_numbers)} cropped parts session folders.")
        extra_confirm = input(f"Type {RED}'CONFIRM ALL'{RESET} to proceed with bulk deletion, or anything else to cancel: ").strip()
        if extra_confirm != 'CONFIRM ALL':
            logger.info(f"\n{BLUE}ℹ️  Bulk deletion cancelled - cleanup skipped.{RESET}\n")
            return True
    else:
        # Parse comma-separated numbers
        try:
            parts = [p.strip() for p in selection_input.split(',')]
            selected_numbers = [int(p) for p in parts if p.isdigit()]
            if not selected_numbers:
                logger.info(f"\n{RED}❌ Invalid selection - no valid numbers found. Cleanup skipped.{RESET}\n")
                return True
        except ValueError:
            logger.info(f"\n{RED}❌ Invalid selection format. Cleanup skipped.{RESET}\n")
            return True
    
    # Validate selections
    invalid = [n for n in selected_numbers if n not in folder_map]
    if invalid:
        logger.info(f"\n{RED}❌ Invalid folder number(s): {', '.join(map(str, invalid))}. Cleanup skipped.{RESET}\n")
        return True
    
    # Build deletion preview
    folders_to_delete = [folder_map[n] for n in selected_numbers]
    logger.info(f"\n{ORANGE}✅ PREVIEW:{RESET} You are about to delete {len(folders_to_delete)} folder(s):")
    for month, folder_path, folder_name in folders_to_delete:
        # Show relative path for safety
        rel_path = os.path.relpath(folder_path, start=CROPPED_PARTS_PATH)
        logger.info(f"   - {rel_path}/")
    
    # Final confirmation
    confirm = input(f"\nType {RED}'CONFIRM'{RESET} to proceed with deletion, or anything else to cancel: ").strip()
    if confirm != 'CONFIRM':
        logger.info(f"\n{BLUE}ℹ️  Deletion cancelled - cleanup skipped.{RESET}\n")
        return True
    
    # Perform deletion with safety validations
    success_count = 0
    failure_count = 0
    
    for month, folder_path, folder_name in folders_to_delete:
        try:
            # Safety validation 1: Must be within cropped_parts directory
            folder_path_abs = os.path.abspath(folder_path)
            cropped_parts_abs = os.path.abspath(CROPPED_PARTS_PATH)
            if not folder_path_abs.startswith(cropped_parts_abs + os.sep):
                raise ValueError(f"Path traversal attempt blocked: {folder_path}")
            
            # Safety validation 2: Must match parts_ pattern
            if not os.path.basename(folder_path).startswith('parts_'):
                raise ValueError(f"Refusing to delete non-session folder: {folder_path}")
            
            # Safety validation 3: Must be a directory
            if not os.path.isdir(folder_path):
                raise ValueError(f"Not a directory: {folder_path}")
            
            # Perform deletion
            shutil.rmtree(folder_path)
            logger.info(f"Deleted cropped parts session folder: {folder_path}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to delete folder {folder_path}: {e}")
            failure_count += 1
    
    # Report results
    if success_count > 0:
        logger.info(f"\n{GREEN}🖼️  Successfully deleted {success_count} cropped parts session folder(s).{RESET}")
    if failure_count > 0:
        logger.warning(f"{RED}⚠️  Failed to delete {failure_count} folder(s) - see logs for details.{RESET}")
    if success_count == 0 and failure_count == 0:
        logger.info(f"{BLUE}ℹ️  No folders deleted.{RESET}")
    
    logger.info("")  # Blank line for readability
    return True

# --- Save debug crop images for light/wiper analysis ---
def save_debug_crop(
    crop_image,
    side,
    component,
    track_id,
    frame_id,
    sample_count,
    diagnosis_timestamp,
    parts_session_folder,
    crop_type
):
    """
    Save debug crop image for light or wiper analysis review.
    
    Args:
        crop_image: NumPy array of cropped region (BGR format)
        side: "left" or "right"
        component: "light_front" or "wiper"
        track_id: YOLO track ID for the component
        frame_id: Frame number where crop was captured
        sample_count: Diagnostic sample number (1-7)
        diagnosis_timestamp: Session start timestamp in "YYYY-MM-DD HH:MM:SS" format
        parts_session_folder: Session folder name like "parts_20260206_143522_123"
        crop_type: "light" or "wiper" to determine subfolder
    
    Returns:
        str: Path to saved debug crop image, or None if saving disabled
    """
    # Check if saving is enabled for this crop type
    if crop_type == "light" and not SAVE_DEBUG_CROPS_LIGHT:
        return None
    if crop_type == "wiper" and not SAVE_DEBUG_CROPS_WIPER:
        return None
    
    # Validate crop image
    if crop_image is None or crop_image.size == 0:
        logger.warning(f"Invalid {crop_type} crop image for debugging")
        return None
    
    # Extract date/time for folder structure
    try:
        dt = datetime.strptime(diagnosis_timestamp, "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y%m%d")
        time_str = dt.strftime("%H%M%S")
        monthly_folder = dt.strftime("%Y%m")
    except:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        monthly_folder = datetime.now().strftime("%Y%m")
    
    # Build folder path: cropped_parts/YYYYMM/parts_YYYYMMDD_HHMMSS_mmm/{crop_type}_debug_YYYYMMDD_HHMMSS/
    debug_subfolder = f"{crop_type}_debug_{date_str}_{time_str}"
    base_output_dir = os.path.join(CROPPED_PARTS_PATH, monthly_folder, parts_session_folder, debug_subfolder)
    
    # Create directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Filename format: {component}_{int_N}_{id_N}_frame{N}.jpg, Order: Component → Intermediate Group → Tracking ID → Frame Number
    component_name = f"{side}_{component}"  # e.g., "right_wiper" or "left_light_front"
    int_group = f"int_{sample_count}"       # e.g., "int_1" through "int_7" (NEVER int_0)
    track_id_str = f"id{track_id}"          # e.g., "id5", "id15"
    frame_str = f"frame{frame_id}"          # e.g., "frame127"
    
    # Filename structure (replaces old: {side}_{component}_track{track_id}_frame{frame_id}_sample{sample_count}.jpg)
    filename = f"{component_name}_{int_group}_{track_id_str}_{frame_str}.jpg"
    filepath = os.path.join(base_output_dir, filename)
    
    # Save crop image
    try:
        cv2.imwrite(filepath, crop_image)
        logger.debug(f"Debug {crop_type} crop saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save debug {crop_type} crop: {e}")
        return None