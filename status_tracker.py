import csv
import os
import re
import uuid
from datetime import datetime
from typing import List, Optional, Tuple
from config import DIAGNOSTICS_PATH, BLUE, RED, ORANGE, GREEN, CYAN, RESET


def normalize_plate(plate: Optional[str]) -> str:
    """
    Normalize plate number to uppercase alphanumeric format.
    Handles None/empty values with placeholder.
    """
    if not plate or plate.strip() == "" or plate == "N/A":
        return "UNKNOWN_PLATE"
    # Keep only alphanumeric chars, uppercase
    normalized = re.sub(r'[^A-Z0-9]', '', plate.upper().strip())
    return normalized if normalized else "UNKNOWN_PLATE"


def normalize_driver(driver_names: Optional[List[str]]) -> str:
    """
    Normalize driver names to single string.
    Handles None/empty lists with placeholder.
    """
    if not driver_names or not isinstance(driver_names, list) or len(driver_names) == 0:
        return "UNKNOWN_DRIVER"
    
    # In reality only 1 driver expected, but handle multi-driver safely
    valid_drivers = [d.strip() for d in driver_names if d and d.strip()]
    if not valid_drivers:
        return "UNKNOWN_DRIVER"
    
    # Sort alphabetically for consistency (though typically single driver)
    valid_drivers.sort()
    return ", ".join(valid_drivers)


def parse_timestamp(timestamp_str: str) -> Tuple[str, str, str, str]:
    """
    Parse ISO 8601 timestamp string into components.
    
    Returns:
        tuple: (date_str, year, month, time_only)
            - date_str: "YYYY-MM-DD"
            - year: "YYYY"
            - month: "MM"
            - time_only: "HH:MM:SS"
    """
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        year = dt.strftime("%Y")
        month = dt.strftime("%m")
        time_only = dt.strftime("%H:%M:%S")
        return date_str, year, month, time_only
    except (ValueError, TypeError) as e:
        # Fallback: use raw values with placeholders
        print(f"{RED}⚠️  Failed to parse timestamp '{timestamp_str}': {e}{RESET}")
        now = datetime.now()
        return (
            now.strftime("%Y-%m-%d"),
            now.strftime("%Y"),
            now.strftime("%m"),
            now.strftime("%H:%M:%S")
        )


def extract_session_state(uuid_value: str) -> Tuple[Optional[str], Optional[int], Optional[bool]]:
    """
    Extract base UUID, state number, and direction from UUID string.
    
    Examples:
        "going1-550e8400-e29b-41d4-a716-446655440000" → ("550e...", 1, True)
        "coming3-a1b2c3d4-..." → ("a1b2...", 3, False)
    
    Returns:
        tuple: (base_uuid, state_num, is_going) or (None, None, None) on parse failure
    """
    # Match pattern: going<NUM>-<uuid> or coming<NUM>-<uuid>
    match = re.match(r'^(going|coming)(\d+)-([0-9a-fA-F\-]{36})$', uuid_value)
    if not match:
        return None, None, None
    
    direction = match.group(1)
    state_num = int(match.group(2))
    base_uuid = match.group(3)
    is_going = (direction == "going")
    
    return base_uuid, state_num, is_going


def find_existing_sessions(csv_path: str, date_str: str, plate: str, driver: str) -> List[Tuple[str, str]]:
    """
    Find all existing sessions for the same (date, plate, driver) combination.
    
    Returns:
        List of (uuid_value, start_time) tuples sorted chronologically
    """
    if not os.path.exists(csv_path):
        return []
    
    existing_sessions = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Match on date, normalized plate, and normalized driver
                if (row.get('date') == date_str and
                    row.get('plate_number') == plate and
                    row.get('driver') == driver):
                    existing_sessions.append((row['uuid'], row['start']))
        
        # Sort by start time chronologically
        existing_sessions.sort(key=lambda x: x[1])
        return existing_sessions
        
    except Exception as e:
        print(f"{RED}⚠️  Error reading CSV {csv_path}: {e}{RESET}")
        return []


def determine_next_state(existing_sessions: List[Tuple[str, str]]) -> str:
    """
    Determine next session state based on existing sessions.
    
    State cycling logic:
        No existing → "going1-<new_uuid>"
        Last state = goingN → "comingN-<same_base_uuid>"
        Last state = comingN → "going{N+1}-<new_uuid>"
    
    Returns:
        Full UUID string with state prefix (e.g., "going1-550e...")
    """
    if not existing_sessions:
        # First session of the day → going1 with new UUID
        base_uuid = str(uuid.uuid4())
        return f"going1-{base_uuid}"
    
    # Get last session's UUID
    last_uuid = existing_sessions[-1][0]
    base_uuid, state_num, is_going = extract_session_state(last_uuid)
    
    if base_uuid is None or state_num is None:
        # Malformed UUID in CSV → treat as new session
        print(f"{ORANGE}⚠️  Malformed UUID '{last_uuid}' in CSV, generating new session{RESET}")
        base_uuid = str(uuid.uuid4())
        return f"going1-{base_uuid}"
    
    if is_going:
        # Last was goingN → next is comingN (same base UUID)
        return f"coming{state_num}-{base_uuid}"
    else:
        # Last was comingN → next is going{N+1} (new base UUID)
        new_base_uuid = str(uuid.uuid4())
        return f"going{state_num + 1}-{new_base_uuid}"


def format_component_list(components: Optional[List[str]]) -> str:
    """
    Format component list into semicolon-joined string.
    Handles None/empty lists gracefully.
    """
    if not components or not isinstance(components, list):
        return ""
    return "; ".join(components)


def get_csv_path(year: str, month: str) -> str:
    """
    Generate CSV path based on year and month.
    Ensures diagnostics directory exists.
    """
    os.makedirs(DIAGNOSTICS_PATH, exist_ok=True)
    filename = f"{year}_{month}_main_diagnostics.csv"
    return os.path.join(DIAGNOSTICS_PATH, filename)

def compute_session_status(session_uuid: str, diagnostics_ng: Optional[List[str]]) -> str:
    """
    Compute session status based on direction (from UUID) and broken components.
    
    Returns one of: going_ok, going_ng, coming_ok, coming_ng, going_na, coming_na
    
    Logic:
        - Direction extracted from UUID prefix (goingN/comingN)
        - Condition:
            * 'ok'  = no real broken items (empty/None/whitespace-only diagnostics_ng)
            * 'ng'  = ≥1 non-empty/non-whitespace item in diagnostics_ng
            * 'na'  = malformed UUID or unexpected input types
    """
    # Extract direction from UUID
    base_uuid, state_num, is_going = extract_session_state(session_uuid)
    
    if is_going is None:
        # Malformed UUID - cannot determine direction
        return "going_na"
    
    direction = "going" if is_going else "coming"
    
    # Determine condition from broken components
    if diagnostics_ng is None or not isinstance(diagnostics_ng, list):
        condition = "na"
    else:
        # Filter out empty/whitespace-only items
        real_broken = [item for item in diagnostics_ng if item and str(item).strip()]
        condition = "ng" if real_broken else "ok"
    
    return f"{direction}_{condition}"

def main(
    start_time: str,
    finish_time: str,
    plate_number: Optional[str],
    driver_names: Optional[List[str]],
    diagnostics_ng: Optional[List[str]],
    diagnostics_ok: Optional[List[str]]
) -> str:
    """
    Log a new session to the monthly diagnostics CSV.
    
    Args:
        start_time: ISO 8601 timestamp string (e.g., "2026-01-27 14:35:22")
        finish_time: ISO 8601 timestamp string
        plate_number: Raw plate number string (may be None)
        driver_names: List of driver names (may be None/empty)
        diagnostics_ng: List of broken components
        diagnostics_ok: List of working components
    
    Returns:
        tuple: (logging_timestamp, session_uuid)
            - logging_timestamp: ISO 8601 timestamp when session was logged (current time)
            - session_uuid: Generated session UUID string (e.g., "going1-550e...")
    """

    table_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Normalize inputs
    normalized_plate = normalize_plate(plate_number)
    normalized_driver = normalize_driver(driver_names)
    
    # Parse timestamps
    date_str, year, month, start_time_only = parse_timestamp(start_time)
    _, _, _, finish_time_only = parse_timestamp(finish_time)
    
    # Get CSV path
    csv_path = get_csv_path(year, month)
    
    # Find existing sessions for this (date, plate, driver) combo
    existing_sessions = find_existing_sessions(csv_path, date_str, normalized_plate, normalized_driver)
    
    # Determine next session state
    session_uuid = determine_next_state(existing_sessions)
    
    # Format components
    broken_components = format_component_list(diagnostics_ng)
    working_components = format_component_list(diagnostics_ok)
    
    # Compute status value
    status_value = compute_session_status(session_uuid, diagnostics_ng)

    # Prepare CSV row
    row_data = {
        'uuid': session_uuid,
        'date': date_str,
        'start': start_time_only,
        'finish': finish_time_only,
        'plate_number': normalized_plate,
        'driver': normalized_driver,
        'broken_component': broken_components,
        'working_component': working_components,
        'status': status_value
    }
    
    # Write to CSV (create if doesn't exist)
    file_exists = os.path.exists(csv_path)
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'uuid', 'date', 'start', 'finish', 
                'plate_number', 'driver', 
                'broken_component', 'working_component', 'status'
            ])
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            # Write row
            writer.writerow(row_data)
        
        print(f"{GREEN}✅ Session stored successfully into {os.path.basename(csv_path)}:{RESET}")
        print(f"   UUID: {session_uuid}")
        print(f"   Driver: {normalized_driver} | Plate: {normalized_plate}")
        print(f"   Time: {start_time_only} → {finish_time_only}")
        print(f"   Status: {status_value}")  # NEW - feedback to user
        return table_timestamp, session_uuid
        
    except Exception as e:
        print(f"{RED}❌ Failed to write to CSV {csv_path}: {e}{RESET}")
        return table_timestamp, f"ERROR-{session_uuid}"


# --- Standalone Testing ---
if __name__ == "__main__":
    # print(f"{BLUE}🚛 Session Tracker - Test Fleet Operations Test{RESET}\n")
    
    # # SCENARIO 1: Perfect Round Trip (Morning departure → Evening return)
    # print(f"{CYAN}🚛 SCENARIO 1: Perfect Round Trip (Truck ABC123 - Alvin){RESET}")
    # print(f"   Morning departure → Evening return with no issues\n")
    
    # test_start = "2026-02-02 06:30:15"
    # test_finish = "2026-02-02 06:33:40"
    # test_plate = "ABC 123"
    # test_driver = ["Alvin Hernandez"]
    # test_broken = []
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number ABC123"
    # ]
    
    # print("Logging morning departure (going1)...")
    # timestamp1, uuid1 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid1}\n")
    
    # test_start = "2026-02-02 18:45:22"
    # test_finish = "2026-02-02 18:48:55"
    # test_plate = "ABC 123"
    # test_driver = ["Alvin Hernandez"]
    # test_broken = []
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number ABC123"
    # ]
    
    # print("Logging evening return (coming1)...")
    # timestamp2, uuid2 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid2}\n")
    
    # # SCENARIO 2: Pre-existing Issue Persists (Broken lights stay broken)
    # print(f"{CYAN}🚛 SCENARIO 2: Pre-existing Issue (Truck XYZ789 - Maria){RESET}")
    # print(f"   Known broken lights persist through entire shift (no magic repair!)\n")
    
    # test_start = "2026-02-02 07:15:10"
    # test_finish = "2026-02-02 07:18:30"
    # test_plate = "XYZ 789"
    # test_driver = ["Maria Garcia"]
    # test_broken = ["❌ front lights: missing/broken"]
    # test_working = [
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number XYZ789"
    # ]
    
    # print("Logging morning departure with broken lights (going1)...")
    # timestamp3, uuid3 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid3}\n")
    
    # test_start = "2026-02-02 19:20:05"
    # test_finish = "2026-02-02 19:23:40"
    # test_plate = "XYZ 789"
    # test_driver = ["Maria Garcia"]
    # test_broken = ["❌ front lights: missing/broken"]  # STILL BROKEN - realistic!
    # test_working = [
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number XYZ789"
    # ]
    
    # print("Logging evening return - lights STILL broken (coming1)...")
    # timestamp4, uuid4 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid4}\n")
    
    # # SCENARIO 3: Degradation During Shift (Clean plate → dirty plate)
    # print(f"{CYAN}🚛 SCENARIO 3: Degradation During Shift (Truck DEF456 - Carlos){RESET}")
    # print(f"   Clean plate at departure → dirty plate on return (road grime accumulation)\n")
    
    # test_start = "2026-02-02 06:50:20"
    # test_finish = "2026-02-02 06:53:45"
    # test_plate = "DEF 456"
    # test_driver = ["Carlos Rodriguez"]
    # test_broken = []
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number DEF456"
    # ]
    
    # print("Logging clean departure (going1)...")
    # timestamp5, uuid5 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid5}\n")
    
    # test_start = "2026-02-02 17:55:30"
    # test_finish = "2026-02-02 17:59:10"
    # test_plate = "DEF 456"
    # test_driver = ["Carlos Rodriguez"]
    # test_broken = [
    #     "⚠️ plate number: visible but dirty/obscured",  # Degraded from clean
    #     "⚠️ wiper: streaking/worn"                      # Degraded from ok
    # ]
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok"
    # ]
    
    # print("Logging return with dirty plate/worn wipers (coming1)...")
    # timestamp6, uuid6 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid6}\n")
    
    # # SCENARIO 4: New Damage During Shift (Mirror cracks)
    # print(f"{CYAN}🚛 SCENARIO 4: New Damage During Shift (Truck GHI789 - Alvin){RESET}")
    # print(f"   Perfect mirror at departure → cracked mirror on return (accident damage)\n")
    
    # test_start = "2026-02-02 20:10:15"
    # test_finish = "2026-02-02 20:13:40"
    # test_plate = "GHI 789"
    # test_driver = ["Alvin Hernandez"]
    # test_broken = []
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ front mirror: ok",      # Perfect at departure
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number GHI789"
    # ]
    
    # print("Logging second departure (going1 - different truck)...")
    # timestamp7, uuid7 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid7}\n")
    
    # test_start = "2026-02-02 23:45:10"
    # test_finish = "2026-02-02 23:48:35"
    # test_plate = "GHI 789"
    # test_driver = ["Alvin Hernandez"]
    # test_broken = ["⚠️ front mirror: cracked/damaged"]  # NEW DAMAGE - realistic!
    # test_working = [
    #     "✅ front lights: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number GHI789"
    # ]
    
    # print("Logging return with cracked mirror (coming1)...")
    # timestamp8, uuid8 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid8}\n")
    
    # # SCENARIO 5: Shift Handoff (Different driver, same persistent issue)
    # print(f"{CYAN}🚛 SCENARIO 5: Shift Handoff (Truck XYZ789 - Maria → David){RESET}")
    # print(f"   Broken lights persist through driver change (no depot repair during shift)\n")
    
    # test_start = "2026-02-02 21:30:05"
    # test_finish = "2026-02-02 21:33:25"
    # test_plate = "XYZ 789"
    # test_driver = ["David Chen"]  # Different driver
    # test_broken = ["❌ front lights: missing/broken"]  # STILL BROKEN - realistic!
    # test_working = [
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number XYZ789"
    # ]
    
    # print("Logging night shift departure with same broken lights (going1)...")
    # timestamp9, uuid9 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid9}\n")
    
    # # SCENARIO 6: Next Day After Depot Repair (Overnight fix)
    # print(f"{CYAN}🚛 SCENARIO 6: Next Day After Depot Repair (Truck XYZ789 - Maria){RESET}")
    # print(f"   Broken lights FIXED overnight at depot (only realistic improvement point)\n")
    
    # test_start = "2026-02-03 06:45:10"
    # test_finish = "2026-02-03 06:48:35"
    # test_plate = "XYZ 789"
    # test_driver = ["Maria Garcia"]
    # test_broken = []
    # test_working = [
    #     "✅ front lights: ok",  # FIXED OVERNIGHT - realistic!
    #     "✅ front mirror: ok",
    #     "✅ top mirror: ok",
    #     "✅ wiper: ok",
    #     "✅ plate number: visible with number XYZ789"
    # ]
    
    # print("Logging next day departure with repaired lights (going1)...")
    # timestamp10, uuid10 = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    # print(f"Generated UUID: {uuid10}\n")
    
    # print(f"{BLUE}✅ REALISTIC FLEET OPERATIONS TEST COMPLETED{RESET}")

# --- This is the actual truck status tracker that will be used in orchestrator ---
    print(f"{BLUE}🚛 Session Tracker - Actual Fleet Operations{RESET}\n")
    session_timestamp, session_uuid = main(test_start, test_finish, test_plate, test_driver, test_broken, test_working)
    if session_uuid is None:
        print(f"{RED}❌ Session logging failed.{RESET}")
    else:
        print(f"\n{GREEN}✅ Session logged with UUID: {session_uuid}{RESET}")
        print(f"{BLUE}✨ Open diagnostics/2026_01_main_diagnostics.csv to review realistic fleet data{RESET}")