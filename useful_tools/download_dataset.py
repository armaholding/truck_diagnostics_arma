# from roboflow import Roboflow
# from dotenv import load_dotenv
# import os
# import supervision as sv

# load_dotenv()
# # Access the API key from environment variables
# roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

# rf = Roboflow(api_key=roboflow_api_key)
# project = rf.workspace("ml-project-ymokd").project("truck_parts")
# version = project.version(10)
# dataset = version.download("yolov11")

# print("✅ Dataset downloaded. Converting all annotations to clean bounding boxes...")

# # Fix labels by forcing mask interpretation and re-exporting as YOLO
# for subset in ["train", "valid", "test"]:
#     images_dir = os.path.join(dataset.location, subset, "images")
#     labels_dir = os.path.join(dataset.location, subset, "labels")
#     data_yaml = os.path.join(dataset.location, "data.yaml")
    
#     if not os.path.exists(images_dir):
#         continue  # skip if split doesn't exist
    
#     print(f"Processing {subset} set...")
    
#     # Load dataset with force_masks=True → treats all as polygons/masks
#     ds = sv.DetectionDataset.from_yolo(
#         images_directory_path=images_dir,
#         annotations_directory_path=labels_dir,
#         data_yaml_path=data_yaml,
#         force_masks=True  # ← THIS IS KEY
#     )
    
#     # Re-export as YOLO format — now with clean bounding boxes
#     ds.as_yolo(
#         annotations_directory_path=labels_dir,
#         # overwrite=True
#     )

# print("🎉 Labels standardized. Ready for training!")


import os
import glob
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

# Step 1: Download dataset in yolov8 format
rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("ml-project-ymokd").project("truck_parts")
version = project.version(10)
dataset = version.download("yolov11")

print("✅ Dataset downloaded. Cleaning labels...")

def polygon_to_bbox(coords):
    """Convert normalized polygon coords [x1,y1,x2,y2,...] to [xc,yc,w,h]"""
    xs = coords[::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    return xc, yc, w, h

# Process all splits
for subset in ["train", "valid", "test"]:
    labels_dir = os.path.join(dataset.location, subset, "labels")
    if not os.path.exists(labels_dir):
        continue
        
    print(f"Processing {subset}...")
    label_paths = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    for lp in label_paths:
        new_lines = []
        with open(lp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))  # handle "9.0" or "9"
                    rest = list(map(float, parts[1:]))
                    
                    if len(rest) == 4:
                        # Already a valid bbox
                        xc, yc, w, h = rest
                        # Clamp to [0,1]
                        xc = max(0, min(1, xc))
                        yc = max(0, min(1, yc))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))
                        new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        
                    elif len(rest) >= 6 and len(rest) % 2 == 0:
                        # Polygon: convert to bbox
                        xc, yc, w, h = polygon_to_bbox(rest)
                        xc = max(0, min(1, xc))
                        yc = max(0, min(1, yc))
                        w = max(0, min(1, w))
                        h = max(0, min(1, h))
                        new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    else:
                        print(f"⚠️ Skipping invalid line in {lp}: {line.strip()}")
                except Exception as e:
                    print(f"⚠️ Error parsing line in {lp}: {e}")
        
        # Overwrite with clean labels
        with open(lp, 'w') as f:
            f.write("\n".join(new_lines))

print("🎉 All labels cleaned! Now safe for YOLO detection training.")