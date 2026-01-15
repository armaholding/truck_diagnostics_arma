import os
import cv2
import numpy as np
import sys

# --- Configuration ---
DATASET_ROOT = "truck_parts-7"
CLASS_NAMES = [
    'carrier', 'lift', 'light_back', 'light_front', 'mirror',
    'mirror_top', 'plate_number', 'stand', 'truck_back', 'truck_front', 'wiper'
]

def parse_label_line(line):
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
        return None, []
    class_id = int(parts[0])
    coords = parts[1:]
    if len(coords) % 2 != 0:
        return None, []
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return class_id, points

def compute_bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    return x_center, y_center, width, height

def denormalize_bbox(x_center, y_center, width, height, img_w, img_h):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2

def draw_bounding_boxes(image, label_path, class_names):
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return image

    img_h, img_w = image.shape[:2]
    annotated_img = image.copy()

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        class_id, points = parse_label_line(line)
        if class_id is None or class_id >= len(class_names):
            continue

        if len(points) == 2:
            x_center, y_center = points[0]
            width, height = points[1]
        else:
            x_center, y_center, width, height = compute_bbox_from_points(points)

        if width <= 0 or height <= 0:
            continue

        x1, y1, x2, y2 = denormalize_bbox(x_center, y_center, width, height, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 255, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        cv2.putText(
            annotated_img, label,
            (x1, y1 - 10 if y1 > 20 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return annotated_img

def safe_imshow(window_name, image):
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    except cv2.error as e:
        if "The function is not implemented" in str(e):
            fallback_path = "debug_view.jpg"
            cv2.imwrite(fallback_path, image)
            print(f"\n⚠️  OpenCV GUI not available. Annotated image saved to: {fallback_path}")
            print("View the image externally, then press Enter to continue...")
            input()
            return False
        else:
            raise

def main():
    while True:
        split = input("Enter split to visualize (train/valid/test): ").strip().lower()
        if split in ['train', 'valid', 'test']:
            break
        print("Invalid input. Please enter 'train', 'valid', or 'test'.")

    images_dir = os.path.join(DATASET_ROOT, split, "images")
    labels_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    image_files.sort()
    print(f"\nFound {len(image_files)} images in '{split}' split.")
    print("Press Enter to view each image. Type 'q' + Enter at any prompt to quit.\n")

    for idx, img_file in enumerate(image_files):
        print(f"\n--- [{idx+1}/{len(image_files)}] {img_file} ---")
        user_input = input("Press Enter to view, or 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("Exiting.")
            break

        # If user pressed Enter (empty string), proceed
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        annotated_image = draw_bounding_boxes(image, label_path, CLASS_NAMES)
        safe_imshow("Annotated Image - Press any key to close", annotated_image)

    print("\nVisualization completed.")

if __name__ == "__main__":
    main()