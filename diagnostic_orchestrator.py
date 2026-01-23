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

QR_CODE_SCANNER = "qr_code_scannerrt"
TRUCK_DIAGNOSER = "yolo_truckrt"
AI_RECOMMENDER = "maintenance_bot"
RATE_LIMIT_DELAY = 1.0  # seconds between stages to avoid overload

BLUE = "\033[94m"
RED = "\033[91m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

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
driver_names = None

qs_main = import_main(QR_CODE_SCANNER)
if qs_main:
    logger.info("🔹 Running qr_code_scanner to find relevant video...")
    driver_names = qs_main()
else:
    logger.critical("❌ Skipping qr_code_scanner: main() function not available")

# Handle output
if driver_names is not None:
    logger.info(f"✅ Matched drivers: '{driver_names}'")
else:
    logger.warning(f"🔍 No drivers found")

time.sleep(RATE_LIMIT_DELAY)

# --- STAGE 2: Truck diagnostics ---
results = None
truck_face = None
truck_components = None
diagnostics_ok = None
diagnostics_ng = None
plate_number = None

yt_main = import_main(TRUCK_DIAGNOSER)
if yt_main:
    logger.info("🔹 Running yolo_truck to get timestamped answer...")
    results = yt_main()
    if results is not None:
        truck_face = results["truck_face"]
        truck_components = results["truck_components"]
        diagnostics_ok = results["diagnostics_ok"]
        diagnostics_ng = results["diagnostics_ng"]
        plate_number = truck_components.get("plate_number", {}).get("number", "N/A")
else:
    logger.critical("❌ Skipping yolo_truck: main() function not available")

# Handle output
if results is not None:
    logger.info(f"✅ Truck face timestamp: {truck_face}")
    logger.info(f"✅ Plate number: {plate_number}")
    logger.info(f"✅ Truck components detected: {list(truck_components.keys())}")
    logger.info(f"✅ Diagnostics NG: {diagnostics_ng}")
    logger.info(f"✅ Diagnostics OK: {diagnostics_ok}")
else:
    logger.warning("❌ No diagnostics results available")

time.sleep(RATE_LIMIT_DELAY)

# --- Stage 3: LLM Recommendations ---
recommendations = None
repair_instructions = None
maintenance_tips = None

mb_main = import_main(AI_RECOMMENDER)
if mb_main:
    logger.info("🔹 Running maintenance_bot to get recommendations...")
    recommendations = mb_main(diagnostics_ok, diagnostics_ng)
    if recommendations is not None:
        repair_instructions = recommendations.get("fixes")
        maintenance_tips = recommendations.get("maintenance")
else:
    logger.warning("❌ Skipping maintenance_bot: module not available")

if recommendations is not None:
    logger.info(f"✅ Repair instructions: {repair_instructions}")
    logger.info(f"✅ Maintenance tips: {maintenance_tips}")
else:
    logger.warning("❌ No recommendations available")

# --- Output Processing ---
if driver_names is not None and results is not None and repair_instructions is not None:
    logger.info("🔹🔹🔹 REPORTING FINAL OUTPUT 🔹🔹🔹")
    # Extract driver name and plate number
    if isinstance(driver_names, list):
        if len(driver_names) == 0:
            driver_display = "Unknown"
        elif len(driver_names) == 1:
            driver_display = driver_names[0]
        else:
            driver_display = ", ".join(driver_names)  # e.g., "Alvin, Maria"
    else:
        driver_display = str(driver_names)  # fallback if it's already a string
    
    print(f"\n{BLUE}--- Driver and Truck Information ---{RESET}")
    print(f"Driver: {driver_display} is using the truck with plate Number: {plate_number}")
    
    # Extract diagnostics information
    print(f"\n{RED}--- Parts that need repair ---{RESET}")
    if diagnostics_ng:
        for issue in diagnostics_ng:
            print(f"- {issue}")
    else:
        print("- None")

    print(f"\n{ORANGE}--- Repair Instructions ---{RESET}")
    print(repair_instructions)

    print(f"\n{GREEN}--- Parts that are working well ---{RESET}")
    if diagnostics_ok:
        for item in diagnostics_ok:
            print(f"- {item}")
    else:
        print("- None")

    print(f"\n{CYAN}--- Maintenance Tips ---{RESET}")
    print(maintenance_tips)

    logger.info("\n🔹🔹🔹 END OF FINAL OUTPUT REPORT 🔹🔹🔹")
else:
    logger.warning("❌❌❌ Incomplete data, final output report not generated")