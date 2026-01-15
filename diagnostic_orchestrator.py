# Module names (must match filenames without .py)
import logging
import time
from typing import Callable, Any, Optional

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LIBRARY_SEEKER_CODE = "qr_code_scanner"
ANSWER_QUERY_CODE = "yolo_truck"
RATE_LIMIT_DELAY = 1.0  # seconds between stages to avoid overload


# === Import main functions directly ===
def import_main(module_name: str) -> Optional[Callable[..., Any]]:
    try:
        mod = __import__(module_name, fromlist=['main'])
        return getattr(mod, 'main', None)
    except Exception as e:
        logger.error(f"❌ Failed to import {module_name}: {e}")
    return None

# === EXECUTE QUERY PIPELINE (CORE FUNCTIONALITY) ===
# --- STAGE 1: QR code scanner ---
qs_main = import_main(LIBRARY_SEEKER_CODE)
if qs_main:
    logger.info("🔹 Running qr_code_scanner to find relevant video...")
    driver_names = qs_main()
else:
    logger.critical("❌ Skipping qr_code_scanner: main() function not available")
    driver_names = None

# Handle no match
if not driver_names:
    logger.warning(f"🔍 No drivers found")

logger.info(f"✅ Matched drivers: '{driver_names}'")
time.sleep(RATE_LIMIT_DELAY)


# --- STAGE 2: Truck diagnostics ---
yt_main = import_main(ANSWER_QUERY_CODE)
if yt_main:
    logger.info("🔹 Running yolo_truck to get timestamped answer...")
    diagnostics = yt_main()
else:
    logger.critical("❌ Skipping yolo_truck: main() function not available")
    diagnostics = None

if diagnostics is None:
    logger.warning(f"❌ Could not generate diagnostics from video")

logger.info(f"✅ Generated diagnostics: {diagnostics}")
time.sleep(RATE_LIMIT_DELAY)