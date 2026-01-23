import cv2
import os
import base64
import logging
import time

# Configure logging (console output with level prefix)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE = os.path.join(BASE_DIR, 'names.txt')
QRCODE_PREFIX = "arma_driver: "

def is_name_in_database(name_to_check):
    """Check if name exists in names.txt (case-insensitive). Returns original casing if found."""
    if not os.path.exists(NAMES_FILE):
        logging.error(f"Database file not found: {NAMES_FILE}")
        return None

    target_lower = name_to_check.strip().lower()
    try:
        with open(NAMES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                original_name = line.strip()
                if original_name and original_name.lower() == target_lower:
                    return original_name
        return None
    except Exception as e:
        logging.error(f"Error reading database: {e}")
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

def scan_one_qr():
    """Capture a single QR code from the default camera with live preview."""
    print("📸 Opening camera... Show a QR code.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not access camera")
        return None

    qr_decoder = cv2.QRCodeDetector()
    window_name = "QR Scanner - Show QR Code (Press 'q' to cancel)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab camera frame")
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
    """Interactive QR scanner: validate scanned codes against local name database."""
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
                    name_to_check = extract_name_from_scanned(scanned)
                    matched_name = is_name_in_database(name_to_check)
                    if matched_name is not None:
                        print(f"✅ {matched_name} is in the database")
                        matched_names.append(matched_name)
                    else:
                        print(f"❌ Scanned name not in database: '{name_to_check}'")
                except IndexError as e:
                    print(f"❌ Invalid QR format: {e}")
        else:
            print("⚠️  Please enter 'y' for yes or 'n' for no.")

    return matched_names

if __name__ == "__main__":
    print("👋 Welcome to the QR Code Scanner!")
    match_name_list = main()
    print("\n✅ Matched names:", match_name_list)