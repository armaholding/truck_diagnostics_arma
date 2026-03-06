# config.py
"""Configuration constants for truck diagnostic system."""
import os

# --- File Paths ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DRIVER_SOURCE_PATH = os.path.join(BASE_PATH, 'names.txt')
QR_CODE_PATH = os.path.join(BASE_PATH, 'qr_codes')
GENERATED_NAMES_JSON = os.path.join(BASE_PATH, 'generated_qr_names.json')
TEST_QR_IMAGE_PATH = os.path.join(BASE_PATH, 'qr_codes', 'alvin_hernandez.png') # Test qr code image
TEST_IMAGE_PATH = os.path.join(BASE_PATH, 'truck_test', 'truckb4.jpg')          # Test image
TEST_VIDEO_PATH = os.path.join(BASE_PATH, 'truck_test', 'test_video1.mp4')      # Test video
OUTPUT_IMAGE_PATH = os.path.join(BASE_PATH, "annotated_image.jpg")
OUTPUT_VIDEO_PATH = os.path.join(BASE_PATH, "output_video.mp4")
DIAGNOSTICS_PATH = os.path.join(BASE_PATH, "diagnostics")                       # Directory to store the diagnostic files
CROPPED_PARTS_PATH = os.path.join(BASE_PATH, "cropped_parts")                   # Directory to store cropped part images

# QR code generation parameters
QR_VERSION = 2   # QR version (1–40); 2 ≈ 25×25 modules
BOX_SIZE = 20    # Pixels per QR module
BORDER = 4       # Quiet zone (modules)
FONT_SIZE = 40   # Font size for name label

# --- Operational Constants ---
QRCODE_PREFIX = "arma_driver: "                              # Prefix for QR payload
INPUT_MODE = "video"                                         # Options: "image", "camera", "video"
YOLO_MODEL_PATH = 'truck9.pt'                                # Path to the YOLO model
VLM_MODEL_PATH = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"  # Path to the VLM OCR model
MAINENANCE_BOT_MODEL = "gpt-4o-mini"                         # OpenAI model for repair and maintenance instructions

# --- Input & Tracking Configuration ---
CAMERA_INDEX = 0                # Default camera index for live feed
COUNT_TO_DECIDE_CONSENSUS = 7   # Maximum number of diagnostic samples to collect before stopping
DIAGNOSTIC_INTERVAL_SECONDS = 5 # Real-time interval between diagnostic collections
IGNORE_PERIOD_SECONDS = 2       # Initial ignore period for unstable detections
MIN_SAMPLE_GAP_FRAMES = 150     # Minimum frames between samples (ensures crop collection window)
TRACKER_TYPE = "botsort.yaml"   # or "bytetrack.yaml"
IOU_THRESHOLD = 0.35
# CONSENSUS_WINDOW_SECONDS = 18   # Time window for consensus diagnostics
# SAVE_INTERVAL_SECONDS = 3       # Interval to save intermediate diagnostics

# --- Diagnostic Configuration ---
DIAGNOSTIC_THRESHOLD = 0.40  # for "ok" vs "ng"
EXPECTED_COMPONENT_COUNTS = {
    'mirror': 2,
    'light_front': 2,
    'wiper': 2,
    'mirror_top': 1,
    'plate_number': 1,
    'carrier': 1,
    'lift': 1,
    'light_back': 2,
    'stand': 2,
    'truck_front': 1,
    'truck_back': 1
}
PAIRED_COMPONENTS = {
    "mirror": ("left_mirror", "right_mirror"),
    "light_front": ("left_light_front", "right_light_front"),
    "light_back": ("left_light_back", "right_light_back"),
    "wiper": ("left_wiper", "right_wiper"),
    "stand": ("left_stand", "right_stand"),
}
FRONT_EXPECTED_COMPONENTS = {
    "left_mirror", "right_mirror",
    "left_light_front", "right_light_front",
    "left_wiper", "right_wiper",
    "mirror_top",
    "plate_number"
}
BACK_EXPECTED_COMPONENTS = {
    "left_light_back", "right_light_back",
    "left_stand", "right_stand",
    "carrier",
    "lift",
    "plate_number"
}


# Wiper Configuration
WIPER_FRAMES_TO_COLLECT = 6                 # Number of frames to collect for wiper analysis
WIPER_COLLECTION_INTERVAL_SECONDS = 0.5     # Real-time interval between wiper frame collection
WIPER_MIN_FRAMES_FOR_ANALYSIS = 3           # Minimum frames required to assess wiper movement (fallback to detection-only if less)
WIPER_MOVEMENT_THRESHOLD_RELATIVE = 0.10    # Movement threshold: 10% of image width for "significant" displacement

# Light Configuration
LIGHT_FRAMES_TO_COLLECT = 6                 # Number of frames to collect for light analysis
LIGHT_COLLECTION_INTERVAL_SECONDS = 0.5     # Real-time interval between frame collection (seconds)
LIGHT_MIN_FRAMES_FOR_ANALYSIS = 3           # Minimum frames required for light analysis (fallback to detection-only if less)
LIGHT_BRIGHTNESS_CHANGE_THRESHOLD = 15      # Brightness change threshold (standard deviation on 0-255 scale)
LIGHT_GRID_ROWS = 10                        # Grid resolution for light analysis
LIGHT_GRID_COLS = 10                        # Grid resolution for light analysis
LIGHT_MIN_SECTION_GRIDS = 2                 # Minimum total grids for valid working section (Equivalent to 1x2 or 2x1)

# --- Output Configuration ---
SAVE_INTERMEDIATE_DIAGNOSTICS = True  # Set to True to save diagnostics for each frame in video mode
SAVE_CROPS = True                     # Set to True to save cropped bounding boxes for validation
DELETE_INT_DIAGNOSTICS = True         # When True, triggers interactive deletion prompt AFTER processing completes
DELETE_SAVED_CROPS = True             # When True, triggers interactive deletion prompt for cropped parts AFTER processing completes
SAVE_DEBUG_CROPS_LIGHT = True        # Save debug crop images for light brightness analysis
SAVE_DEBUG_CROPS_WIPER = True        # Save debug crop images for wiper movement analysis

# --- OCR Configuration ---
OCR_LANGUAGES = ['en']  # You can change this to ['ar'] or ['en', 'ar'] etc.

# --- CLI Color Codes ---
BLUE = "\033[94m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"