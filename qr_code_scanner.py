import cv2
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE = os.path.join(BASE_DIR, 'names.txt')
QRCODE_PREFIX = "arma_driver: "

def is_name_in_database(name_to_check):
    """Check if name exists in names.txt (case-insensitive). Return original name if found."""
    if not os.path.exists(NAMES_FILE):
        print(f"❌ Database file '{NAMES_FILE}' not found!")
        return None

    target_lower = name_to_check.strip().lower()
    try:
        with open(NAMES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                original_name = line.strip()
                if original_name and original_name.lower() == target_lower:
                    return original_name
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return None
    return None

def extract_name_from_scanned(scanned_value: str) -> str:
    """
    Extract the actual name from a scanned QR code value.
    Assumes the value starts with QRCODE_PREFIX.
    """
    return scanned_value[len(QRCODE_PREFIX):]
    
def scan_one_qr():
    """Scan exactly one QR code and return its content, or None if canceled."""
    print("📸 Opening camera... Show a QR code.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access camera.")
        return None

    qr_decoder = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("QR Scanner - Show QR Code (Press 'q' to cancel)", frame)

        # Decode QR
        decoded_info, points, _ = qr_decoder.detectAndDecode(frame)

        if decoded_info:
            # Draw bounding box
            if points is not None:
                pts = points[0].astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                cv2.imshow("QR Scanner - Show QR Code (Press 'q' to cancel)", frame)
                cv2.waitKey(300)

            cap.release()
            cv2.destroyAllWindows()
            return decoded_info.strip()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    """
    Run the QR scanner interactively and collect all matched names.
    Returns a list of successfully validated names (original casing).
    """
    print("👋 Welcome to the QR Code Scanner!")
    matched_names = []
    
    while True:
        user_input = input("\n🔍 Do you want to scan a QR code? (y/n): ").strip().lower()
        
        if user_input in ('n', 'no'):
            print("👋 Goodbye!")
            break
        elif user_input in ('y', 'yes'):
            scanned = scan_one_qr()
            if scanned is None:
                print("⚠️  Scan canceled or failed.")
            else:
                try:
                    # Extract name by removing prefix if present
                    name_to_check = extract_name_from_scanned(scanned)
                    matched_name = is_name_in_database(name_to_check)
                    if matched_name is not None:
                        print(f"✅ {matched_name} is in the database")
                        matched_names.append(matched_name)
                    else:
                        print(f"❌ '{scanned}' is NOT in the database (checked as: '{name_to_check}')")
                except IndexError:
                    # Handles case where scanned_value is shorter than prefix
                    print(f"❌ Invalid QR format: missing prefix '{QRCODE_PREFIX}'")
        else:
            print("⚠️  Please enter 'y' for yes or 'n' for no.")

    return matched_names

if __name__ == "__main__":
    match_name_list = main()
    print(match_name_list)