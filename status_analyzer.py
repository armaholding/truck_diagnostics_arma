import csv
import json
import os
import re
import sys
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import DIAGNOSTICS_PATH

# --- Configuration ---
COMPARE_COLUMN = "compare_going_coming"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_session_state(uuid_value: str) -> Tuple[Optional[str], Optional[int], Optional[bool]]:
    """
    Extract base UUID, state number, and direction from UUID string.
    
    Examples:
        "going1-550e8400-e29b-41d4-a716-446655440000" → ("550e...", 1, True)
        "coming3-a1b2c3d4-..." → ("a1b2...", 3, False)
    
    Returns:
        tuple: (base_uuid, state_num, is_going) or (None, None, None) on parse failure
    """
    match = re.match(r'^(going|coming)(\d+)-([0-9a-fA-F\-]{36})$', uuid_value)
    if not match:
        return None, None, None
    
    direction = match.group(1)
    state_num = int(match.group(2))
    base_uuid = match.group(3)
    is_going = (direction == "going")
    
    return base_uuid, state_num, is_going

def parse_broken_components(broken_str: str) -> List[str]:
    """
    Parse broken_component string into normalized list of items.
    Handles empty strings and whitespace normalization per item.
    
    Example:
        "⚠️ plate: dirty; ❌ wiper: broken" → ["⚠️ plate: dirty", "❌ wiper: broken"]
    """
    if not broken_str or broken_str.strip() == "":
        return []
    
    # Split by semicolon, strip each item
    items = [item.strip() for item in broken_str.split(";")]
    # Filter out empty items after stripping
    return [item for item in items if item]


def determine_comparison(
    going_status: str,
    coming_status: str,
    going_broken: List[str],
    coming_broken: List[str]
) -> Tuple[str, List[str], List[str], str]:
    """
    Determine comparison result based on status and broken components.
    
    Returns:
        tuple: (compare_status, broken_parts, identified_parts, compare_note)
    """
    # Normalize statuses to base form (remove direction prefix)
    going_base = going_status.split("_")[1] if "_" in going_status else going_status
    coming_base = coming_status.split("_")[1] if "_" in coming_status else coming_status
    
    # Handle na statuses per requirements
    if going_base == "na" and coming_base == "ok":
        # going_na → coming_ok → treat as ok_ok
        return "ok_ok", [], [], "ok"
    
    if coming_base == "na":
        # going_ok/going_na → coming_na → ok_na
        return "ok_na", [], [], "diagnosis failed"
    
    if going_base == "na" and coming_base == "ng":
        # going_na → coming_ng → treat as ok_ng
        identified = coming_broken  # All coming items are "new" since going was unknown
        return "ok_ng", coming_broken, identified, ""  # Note will be filled by user
    
    # Standard cases (both ok/ng)
    if going_base == "ok" and coming_base == "ok":
        return "ok_ok", [], [], "ok"
    
    if going_base == "ok" and coming_base == "ng":
        identified = coming_broken  # All coming items are newly broken
        return "ok_ng", coming_broken, identified, ""  # Note will be filled by user
    
    if going_base == "ng" and coming_base == "ng":
        # Check if broken components are identical
        if sorted(going_broken) == sorted(coming_broken):
            # Exact match - same broken parts persisted
            return "ng_ng", going_broken, [], "broken from previous"
        else:
            # Different broken parts - identify new damage
            # identified_parts = items in coming but NOT in going
            identified = [item for item in coming_broken if item not in going_broken]
            # broken_parts = union of all broken items
            broken_union = list(set(going_broken + coming_broken))
            return "ng_nng", broken_union, identified, ""  # Note will be filled by user
    
    # Fallback (should not occur with valid statuses)
    return "unknown", [], [], f"unhandled status combo: {going_status}/{coming_status}"


def prompt_user_reason(plate: str, driver: str, identified_parts: List[str]) -> str:
    """
    Prompt user for reason of degradation/damage for identified parts.
    Returns user input or default if non-interactive.
    """
    logger.info(f"\n🚨 DEGRADATION DETECTED for {plate} ({driver})")
    logger.info(f"   Newly identified broken components:")
    for i, part in enumerate(identified_parts, 1):
        logger.info(f"   {i}. {part}")
    
    # Check if running in interactive terminal
    if not sys.stdin.isatty():
        logger.warning("⚠️  Non-interactive mode - skipping user prompt. Use default note.")
        return "reason not provided (non-interactive mode)"
    
    reason = input("   ➡️  What caused this degradation/damage? (press Enter to skip): ").strip()
    if not reason:
        reason = "no reason provided"
    return reason


def load_existing_json(json_path: str) -> Dict:
    """Load existing JSON analysis or return empty dict."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️  Failed to load {json_path}: {e}. Starting fresh.")
    return {}


def save_json_analysis(json_path: str, analysis_data: Dict):
    """Save analysis data to JSON file with pretty formatting."""
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Analysis saved to {json_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save {json_path}: {e}")


def process_csv_file(csv_path: str) -> bool:
    """
    Process a single monthly CSV file:
    1. Find unprocessed going/coming pairs
    2. Compare statuses and components
    3. Update JSON analysis file
    4. Mark pairs as compared in CSV
    
    Returns:
        True if any pairs were processed, False otherwise
    """
    year_month = os.path.basename(csv_path).replace("_main_diagnostics.csv", "")
    json_path = os.path.join(DIAGNOSTICS_PATH, f"{year_month}_compare_status.json")
    
    # Load existing analysis for this month
    monthly_analysis = load_existing_json(json_path)
    
    # Read CSV rows
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as e:
        logger.error(f"❌ Failed to read {csv_path}: {e}")
        return False
    
    # Add compare_going_coming column if missing
    if COMPARE_COLUMN not in fieldnames:
        fieldnames = list(fieldnames) + [COMPARE_COLUMN]
        for row in rows:
            row[COMPARE_COLUMN] = ""
        logger.info(f"➕ Added new column '{COMPARE_COLUMN}' to {os.path.basename(csv_path)}")
    
    # Group rows by (date, base_uuid, trip_number) for pairing
    pairs: Dict[Tuple[str, str, int], Dict[str, Dict]] = {}
    
    for row in rows:
        # Skip already compared rows
        if str(row.get(COMPARE_COLUMN, "")).strip() == "compared":
            continue
        
        uuid_val = row.get('uuid', '')
        base_uuid, trip_num, is_going = extract_session_state(uuid_val)
        
        if base_uuid is None or trip_num is None:
            continue  # Skip malformed UUIDs
        
        date_val = row.get('date', '')
        key = (date_val, base_uuid, trip_num)
        
        if key not in pairs:
            pairs[key] = {}
        
        direction_key = 'going' if is_going else 'coming'
        pairs[key][direction_key] = row
    
    # Process complete pairs (both going and coming exist)
    processed_count = 0
    rows_to_update = []
    
    for (date_val, base_uuid, trip_num), pair in pairs.items():
        if 'going' not in pair or 'coming' not in pair:
            continue  # Skip incomplete pairs
        
        going_row = pair['going']
        coming_row = pair['coming']
        
        # Extract statuses
        going_status = going_row.get('status', 'going_na')
        coming_status = coming_row.get('status', 'coming_na')
        
        # Parse broken components
        going_broken = parse_broken_components(going_row.get('broken_component', ''))
        coming_broken = parse_broken_components(coming_row.get('broken_component', ''))
        
        # Determine comparison result
        compare_status, broken_parts, identified_parts, compare_note = determine_comparison(
            going_status, coming_status, going_broken, coming_broken
        )
        
        # Handle interactive cases requiring user input
        if compare_status in ("ok_ng", "ng_nng") and identified_parts:
            compare_note = prompt_user_reason(
                going_row.get('plate_number', 'UNKNOWN'),
                going_row.get('driver', 'UNKNOWN'),
                identified_parts
            )
        
        # Build analysis record
        analysis_record = {
            "plate_number": going_row.get('plate_number', 'UNKNOWN'),
            "driver": going_row.get('driver', 'UNKNOWN'),
            "trip_number": trip_num,
            "compare_status": compare_status,
            "broken_parts": broken_parts,
            "identified_parts": identified_parts,
            "compare_note": compare_note
        }
        
        # Update monthly analysis (append mode)
        if date_val not in monthly_analysis:
            monthly_analysis[date_val] = {}
        
        # Only add if not already present (prevent duplicates)
        if base_uuid not in monthly_analysis[date_val]:
            monthly_analysis[date_val][base_uuid] = analysis_record
            processed_count += 1
            
            # Mark both rows for CSV update
            going_row[COMPARE_COLUMN] = "compared"
            coming_row[COMPARE_COLUMN] = "compared"
            rows_to_update.extend([going_row, coming_row])
            
            logger.info(f"✅ Analyzed pair {base_uuid} (trip #{trip_num}) on {date_val}: {compare_status}")
    
    if processed_count == 0:
        logger.info(f"ℹ️  No new unprocessed pairs found in {os.path.basename(csv_path)}")
        return False
    
    # Save updated JSON analysis
    save_json_analysis(json_path, monthly_analysis)
    
    # Create backup before modifying CSV
    backup_path = csv_path + ".bak"
    shutil.copy2(csv_path, backup_path)
    logger.info(f"💾 Created backup: {os.path.basename(backup_path)}")
    
    # Write updated CSV with marked pairs
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)  # All rows (updated + unchanged)
        logger.info(f"✅ Updated {os.path.basename(csv_path)} with {processed_count} compared pairs")
    except Exception as e:
        logger.error(f"❌ Failed to update CSV {csv_path}: {e}")
        logger.info(f"   Restoring from backup...")
        shutil.copy2(backup_path, csv_path)
        return False
    
    return True


def discover_monthly_csvs() -> List[str]:
    """Discover all monthly diagnostics CSV files in diagnostics directory."""
    if not os.path.exists(DIAGNOSTICS_PATH):
        logger.error(f"❌ Diagnostics directory '{DIAGNOSTICS_PATH}' not found")
        return []
    
    csv_files = []
    for fname in os.listdir(DIAGNOSTICS_PATH):
        if fname.endswith("_main_diagnostics.csv"):
            csv_files.append(os.path.join(DIAGNOSTICS_PATH, fname))
    
    csv_files.sort()  # Process chronologically
    return csv_files


def main():
    """
    Main entry point for diagnostics analysis.
    Returns:
        tuple: (analyzer_timestamp, files_with_new_pairs)
            - analyzer_timestamp: ISO 8601 timestamp when analysis ran
            - files_with_new_pairs: Number of CSV files that had new unprocessed pairs analyzed (0 if none)
    """
    logger.info("🚛 Fleet Diagnostics Analyzer - Action_1: Going/Coming Comparison")
    logger.info("=" * 50)
    
    analyzer_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_files = discover_monthly_csvs()
    
    if not csv_files:
        logger.warning("⚠️  No monthly diagnostics CSV files found in 'diagnostics/' directory")
        logger.info(f"   Expected pattern: YYYY_MM_main_diagnostics.csv")
        return analyzer_timestamp, 0
    
    logger.info(f"📁 Found {len(csv_files)} monthly CSV file(s) to process:\n")
    for i, csv_path in enumerate(csv_files, 1):
        logger.info(f"   {i}. {os.path.basename(csv_path)}")
    
    logger.info("\n" + "=" * 50)
    
    files_with_new_pairs = 0

    for csv_path in csv_files:
        logger.info(f"\n📄 Processing: {os.path.basename(csv_path)}")
        logger.info("-" * 50)
        if process_csv_file(csv_path):
            files_with_new_pairs += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"✅ Analysis complete. Processed {files_with_new_pairs} file(s) with new pairs.")
    logger.info(f"   JSON outputs saved as: YYYY_MM_compare_status.json in '{DIAGNOSTICS_PATH}/'")
    logger.info("=" * 50)

    return analyzer_timestamp, files_with_new_pairs

if __name__ == "__main__":
    analyzer_timestamp, files_processed = main()