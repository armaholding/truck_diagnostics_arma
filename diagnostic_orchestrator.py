import logging
import time
from typing import Callable, Any, Optional
from datetime import datetime
from config import BLUE, RED, ORANGE, GREEN, CYAN, RESET

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Module Names ---
QR_CODE_SCANNER = "qr_code_scanner_rt"
TRUCK_DIAGNOSER = "yolo_truck_rt"
AI_RECOMMENDER = "maintenance_bot"
STATUS_TRACKER = "status_tracker"
STATUS_ANALYZER = "status_analyzer"

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
# --- Initialize ALL timestamps upfront ---
qr_scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
diagnosis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
recommendation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tracking_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- STAGE 1: QR code scanner ---
driver_names = None

qs_main = import_main(QR_CODE_SCANNER)
if qs_main:
    logger.info("🔹 Running qr_code_scanner to find relevant video...")
    try:
        qr_scan_timestamp, driver_names = qs_main()
    except Exception as e:
        logger.error(f"QR scanner execution failed: {e}")
        driver_names = None
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
    try:
        diagnosis_timestamp, results = yt_main()  # CRITICAL: Wrapped in try/except
        if results is not None:
            truck_face = results["truck_face"]
            truck_components = results["truck_components"]
            diagnostics_ok = results["diagnostics_ok"]
            diagnostics_ng = results["diagnostics_ng"]
            plate_number = truck_components.get("plate_number", {}).get("number", "N/A")
    except Exception as e:
        logger.error(f"Truck diagnostics execution failed: {e}")
        results = None
else:
    logger.critical("❌ Skipping yolo_truck: main() function not available")

# Handle output
if results is not None:
    logger.info(f"✅ Truck face: {truck_face}")
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
    # Ensure diagnostics are lists even on failure
    safe_diagnostics_ok = diagnostics_ok if diagnostics_ok is not None else []
    safe_diagnostics_ng = diagnostics_ng if diagnostics_ng is not None else []
    try:
        recommendation_timestamp, recommendations = mb_main(safe_diagnostics_ok, safe_diagnostics_ng)  # CRITICAL: Wrapped in try/except
        if recommendations is not None:
            repair_instructions = recommendations.get("fixes")
            maintenance_tips = recommendations.get("maintenance")
    except Exception as e:
        logger.error(f"Maintenance bot execution failed: {e}")
        recommendations = None
else:
    logger.warning("❌ Skipping maintenance_bot: module not available")

if recommendations is not None:
    logger.info(f"✅ Repair instructions: {repair_instructions}")
    logger.info(f"✅ Maintenance tips: {maintenance_tips}")
else:
    logger.warning("❌ No recommendations available")

# --- Stage 4: Output Processing ---
# Timestamps are ALWAYS defined (initialized upfront)
try:
    start_dt = datetime.strptime(qr_scan_timestamp, "%Y-%m-%d %H:%M:%S")
    finish_dt = datetime.strptime(recommendation_timestamp, "%Y-%m-%d %H:%M:%S")
    duration = (finish_dt - start_dt).total_seconds()
    logger.info(f"⏱️  Total processing time: {duration:.1f} seconds")
except Exception as e:
    logger.warning(f"⚠️ Could not calculate duration: {e}")
    logger.info(f"⏱️  Scan time: {qr_scan_timestamp}, Recommendation time: {recommendation_timestamp}")

if driver_names is not None and results is not None and repair_instructions is not None:
    logger.info("🔹🔹🔹 REPORTING FINAL OUTPUT 🔹🔹🔹")
    # Extract driver name and plate number
    if isinstance(driver_names, list):
        if len(driver_names) == 0:
            driver_display = "Unknown (no match)"
        else:
            driver_display = ", ".join(driver_names)
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
    # Show partial data for debugging
    logger.info(f"  Driver data: {'available' if driver_names is not None else 'missing'}")
    logger.info(f"  Diagnostics: {'available' if results is not None else 'missing'}")
    logger.info(f"  Recommendations: {'available' if repair_instructions is not None else 'missing'}")

# --- STAGE 5: Session Tracking & CSV Logging ---
st_main = import_main(STATUS_TRACKER)
if st_main:
    logger.info("🔹 Logging session to diagnostics CSV...")
    try:
        # Pass all required data to session tracker
        tracking_timestamp, session_uuid = st_main(
            start_time=qr_scan_timestamp,
            finish_time=recommendation_timestamp,
            plate_number=plate_number,
            driver_names=driver_names,
            diagnostics_ng=diagnostics_ng,
            diagnostics_ok=diagnostics_ok
        )
        logger.info(f"✅ Session logged with UUID: {session_uuid} at {tracking_timestamp}")
    except Exception as e:
        logger.error(f"Session tracking failed: {e}")
else:
    logger.warning("❌ Skipping session tracking: status_tracker module not available")

# --- STAGE 6: Truck Status Analysis ---
user_input = input("Do you want to proceed with the truck status analysis? Press Enter to proceed, or 'n' to exit: ")
if user_input == '':
    sa_main = import_main(STATUS_ANALYZER)
    if sa_main:
        logger.info("🔹 Running status analyzer to compare going/coming sessions...")
        try:
            analysis_timestamp, files_processed = sa_main()
            logger.info(f"✅ Status analysis completed at {analysis_timestamp}: {files_processed} file(s) with new pairs")
        except Exception as e:
            logger.error(f"Status analysis execution failed: {e}")
    else:
        logger.warning("❌ Skipping status analysis: status_analyzer module not available")
else:
    print("Exiting without truck status analysis")