import cv2

# for i in range(5):
#     cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#     if cap.isOpened():
#         print(f"✅ Camera found at index {i}")
#         cap.release()
#     else:
#         print(f"❌ No camera at index {i}")

import numpy as np

# Test if GUI functions work
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imshow("Test Window", img)
print("If you see a black window, GUI works!")
cv2.waitKey(2000)  # Wait 2 sec
cv2.destroyAllWindows()

# Test camera access
cap = cv2.VideoCapture(0) # Try different indices (0, 1, 2, etc.) if the default doesn't work

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# If successful, proceed with reading frames in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('Camera Feed', frame)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
