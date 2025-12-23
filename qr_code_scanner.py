import cv2
from pyzbar import pyzbar
import os
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE = os.path.join(BASE_DIR, 'names.txt')

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

def scan_one_qr():
    """Scan exactly one QR code and return its content, or None if canceled."""
    print("📸 Opening camera... Show a QR code.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("QR Scanner - Show QR Code (Press 'q' to cancel)", frame)

        # Decode QR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)

        if decoded_objects:
            obj = decoded_objects[0]
            scanned_value = obj.data.decode('utf-8').strip()
            
            # Briefly highlight detection
            pts = [(point.x, point.y) for point in obj.polygon]
            if len(pts) == 4:
                cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 0), 3)
                cv2.imshow("QR Scanner - Show QR Code (Press 'q' to cancel)", frame)
                cv2.waitKey(300)  # Show green box for 0.3 sec

            cap.release()
            cv2.destroyAllWindows()
            return scanned_value

        # Allow manual cancel
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    print("👋 Welcome to the QR Code Scanner!")
    
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
                matched_name = is_name_in_database(scanned)
                if matched_name is not None:
                    print(f"✅ {matched_name} is in the database")
                else:
                    print(f"❌ '{scanned}' is NOT in the database")
        else:
            print("⚠️  Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()