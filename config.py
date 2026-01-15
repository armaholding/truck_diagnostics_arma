# config.py
"""Configuration constants for truck diagnostic system."""

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
FRONT_EXPECTED_COMPONENTS = {'mirror', 'light_front', 'wiper', 'mirror_top', 'plate_number'}
BACK_EXPECTED_COMPONENTS = {'carrier', 'lift', 'light_back', 'stand', 'plate_number'}

# --- OCR Configuration ---
OCR_LANGUAGES = ['en']  # You can change this to ['ar'] or ['en', 'ar'] etc.