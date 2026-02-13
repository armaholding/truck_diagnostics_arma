import cv2
import os
import base64
import logging
import time
from collections import Counter
import re
from datetime import datetime
import sys
from config import (DRIVER_SOURCE_PATH, TEST_QR_IMAGE_PATH, TEST_VIDEO_PATH, INPUT_MODE,
                    QRCODE_PREFIX, BLUE, RED, ORANGE, GREEN, CYAN, RESET)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_name_in_database(name_to_check):
    """Check if name exists in names.txt (case-insensitive). Returns original casing if found."""
    if not os.path.exists(DRIVER_SOURCE_PATH):
        logger.error(f"Database file not found: {DRIVER_SOURCE_PATH}")
        return None

    target_lower = name_to_check.strip().lower()
    try:
        with open(DRIVER_SOURCE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                original_name = line.strip()
                if original_name and original_name.lower() == target_lower:
                    return original_name
        return None
    except Exception as e:
        logger.error(f"Error reading database: {e}")
        return None

def extract_name_from_scanned(scanned_value: str) -> str:
    """Decode Base64 QR payload and extract name after validating prefix."""
    try:
        decoded_bytes = base64.b64decode(scanned_value.strip())
        decoded_str = decoded_bytes.decode('utf-8')
        if not decoded_str.startswith(QRCODE_PREFIX):
            raise ValueError("Missing expected prefix")
        return decoded_str[len(QRCODE_PREFIX):]
    except (base64.binascii.Error, UnicodeDecodeError, ValueError) as e:
        raise IndexError(f"Invalid encoding or prefix: {e}")

def get_consensus_qr_from_frames(frame_generator, total_samples=7, interval_sec=0.5, fps=30, allow_cancel=False):
    """
    Generic function to collect QR samples from a frame source and return the most frequent valid value.
    
    Args:
        frame_generator: callable that yields (frame, should_continue) or just frame
        total_samples: number of samples to collect (default 7)
        interval_sec: time between samples in seconds (default 0.5)
        fps: assumed or actual FPS for timing (used only if frame_generator doesn't control timing)
        allow_cancel: if True, check for 'q' key press during sampling (for camera mode)
    
    Returns:
        Most frequent valid decoded QR string, or None if none found.
    """
    sampled_qr_values = []

    for i in range(total_samples):
        try:
            if allow_cancel:
                frame, should_continue = frame_generator()
                if not should_continue:
                    break
            else:
                frame = frame_generator()
                if frame is None:
                    break
        except StopIteration:
            break

        if frame is None or frame.size == 0:
            continue

        # Use 
        qr_decoder = cv2.QRCodeDetector()
        decoded_info, points, _ = qr_decoder.detectAndDecode(frame)

        if decoded_info and isinstance(decoded_info, str) and decoded_info.strip():
            scanned_str = decoded_info.strip()

            # Quick validation: must look like Base64 (length and chars)
            if len(scanned_str) >= 16 and re.fullmatch(r'[A-Za-z0-9+/=]+', scanned_str):
                sampled_qr_values.append(scanned_str)

        if allow_cancel:
            # Show live preview
            if points is not None:
                pts = points[0].astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.imshow("QR Scanner - Sampling (3s window, press 'q' to cancel)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            time.sleep(interval_sec)

    cv2.destroyAllWindows()

    if not sampled_qr_values:
        return None

    # Vote only among VALID samples
    qr_counter = Counter(sampled_qr_values)
    most_common_qr, _ = qr_counter.most_common(1)[0]
    return most_common_qr

def scan_from_image_file():
    """Decode QR from a static image file."""
    if not os.path.exists(TEST_QR_IMAGE_PATH):
        logger.error(f"Image file not found: {TEST_QR_IMAGE_PATH}")
        return None
    
    img = cv2.imread(TEST_QR_IMAGE_PATH)
    if img is None:
        logger.error("Failed to load image")
        return None
    
    qr_decoder = cv2.QRCodeDetector()
    decoded_info, points, _ = qr_decoder.detectAndDecode(img)

    if decoded_info and isinstance(decoded_info, str) and decoded_info.strip():
        scanned_str = decoded_info.strip()
        if len(scanned_str) >= 16 and re.fullmatch(r'[A-Za-z0-9+/=]+', scanned_str):
            return scanned_str
    return None

def scan_from_camera():
    """Capture QR codes from camera over 3 seconds (7 samples), return consensus value."""
    print("📸 Opening camera... Hold QR code steady for 3 seconds.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not access camera")
        return None

    def camera_frame_generator():
        ret, frame = cap.read()
        return (frame, True) if ret else (None, False)

    try:
        # Estimate FPS (not critical since we sleep)
        return get_consensus_qr_from_frames(
            frame_generator=lambda: camera_frame_generator(),
            total_samples=7,
            interval_sec=0.5,
            allow_cancel=True
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

def scan_from_video_file():
    """Process first 3 seconds of video, collect all valid QR codes, return consensus."""
    print("📸 Opening video...")
    if not os.path.exists(TEST_VIDEO_PATH):
        logger.error(f"Video file not found: {TEST_VIDEO_PATH}")
        return None

    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Could not open video file")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # Calculate total frames in first 3 seconds
    total_frames_to_process = int(3.0 * fps)
    sampled_qr_values = []
    qr_decoder = cv2.QRCodeDetector()

    frame_count = 0
    while frame_count < total_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        # Use detectAndDecode (same as working camera version)
        decoded_info, points, _ = qr_decoder.detectAndDecode(frame)

        if decoded_info and isinstance(decoded_info, str) and decoded_info.strip():
            scanned_str = decoded_info.strip()
            # Validate as Base64-like
            if len(scanned_str) >= 16 and re.fullmatch(r'[A-Za-z0-9+/=]+', scanned_str):
                sampled_qr_values.append(scanned_str)

        frame_count += 1

    cap.release()

    if not sampled_qr_values:
        logger.info("No QR codes detected in video sampling window")
        return None

    # Vote only among VALID samples
    qr_counter = Counter(sampled_qr_values)
    most_common_qr, _ = qr_counter.most_common(1)[0]
    return most_common_qr

def scan_one_qr():
    """Capture a single QR code from the default camera with live preview."""
    print("📸 Opening camera... Show a QR code.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not access camera")
        return None

    qr_decoder = cv2.QRCodeDetector()
    window_name = "QR Scanner - Show QR Code (Press 'q' to cancel)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab camera frame")
                break

            # Decode QR in current frame
            decoded_info, points, _ = qr_decoder.detectAndDecode(frame)

            # Draw detection box if QR is found
            if decoded_info and points is not None:
                pts = points[0].astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)

            if decoded_info:
                cv2.waitKey(300)  # Brief visual confirmation
                return decoded_info.strip()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return None

def main():
    """Main function to run QR code scanning based on INPUT_MODE."""
    matched_names = []

    # Pre-flight validation with timestamp on failure paths
    input_mode = INPUT_MODE.strip().lower()
    if input_mode not in ("camera", "video", "image"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Invalid INPUT_MODE: {INPUT_MODE}. Must be 'camera', 'video', or 'image'.")
        return timestamp, None

    # File existence checks with timestamp on failure
    if input_mode == 'video':
        if not os.path.exists(TEST_VIDEO_PATH):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"Video file not found: {TEST_VIDEO_PATH}")
            return timestamp, None
    
    if input_mode == 'image':
        if not os.path.exists(TEST_QR_IMAGE_PATH):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.error(f"Image file not found: {TEST_QR_IMAGE_PATH}")
            return timestamp, None

    # ✅ RECORD TIMESTAMP HERE: Start of actual scan operation
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Unified single-scan execution (no loops/prompts)
    scanned = None

    try:
        if input_mode == 'camera':
            print("📸 Opening camera... Hold QR code steady for 3 seconds.")
            scanned = scan_from_camera()

        elif input_mode == 'video':
            print("🎬 Processing video file (first 3 seconds)...")
            if not os.path.exists(TEST_VIDEO_PATH):
                logger.error(f"Video file not found: {TEST_VIDEO_PATH}")
                return None
            scanned = scan_from_video_file()

        elif input_mode == 'image':
            print("🖼️  Processing image file...")
            if not os.path.exists(TEST_QR_IMAGE_PATH):
                logger.error(f"Image file not found: {TEST_QR_IMAGE_PATH}")
                return None
            scanned = scan_from_image_file()

    except Exception as e:
        logger.error(f"Scan execution failed: {e}")
        return timestamp, None

    # Determine scan outcome
    if scanned is None:
        # Distinguish cancellation from detection failure (for UX only)
        if input_mode == 'camera':
            print("⚠️  Scan canceled by user")
        else:
            print("⚠️  Scan failed: No QR code detected")
        return timestamp, None

    # QR successfully detected and decoded → process immediately
    print(f"🔍 RAW SCANNED VALUE: [{repr(scanned)}]")
    print(f"    Length: {len(scanned)}")
          
    try:
        name_to_check = extract_name_from_scanned(scanned)
        matched_name = is_name_in_database(name_to_check)
        if matched_name is not None:
            print(f"✅ {matched_name} is in the database")
            matched_names.append(matched_name)
        else:
            print(f"❌ Scanned name not in database: '{name_to_check}'")
    except IndexError as e:
        print(f"❌ Invalid QR format: {e}")
        return timestamp, None
    
    return timestamp, matched_names


if __name__ == "__main__":
    print("👋 Welcome to the QR Code Scanner!")
    qr_scan_timestamp, match_name_list = main()

    # Display timestamp prominently
    print(f"\n⏱️  {BLUE}Scan performed at: {qr_scan_timestamp}{RESET}")
    
    # Final output (always shown, even if empty)
    print(f"✅ {BLUE}Matched driver's names: {match_name_list}{RESET}")
    if match_name_list is None:
        print(f"⚠️ {RED}Scan failed or canceled{RESET}")
    elif not match_name_list:
        print("ℹ️  No valid driver found in scan")
    
    # Exit code for standalone CLI usage (does not affect orchestrator)
    sys.exit(0 if match_name_list is not None else 1)