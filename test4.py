import cv2
import numpy as np
from ultralytics import YOLO
from math import tan, radians, cos, pi
import csv
import os
from scipy.spatial.transform import Rotation as R
from extract_gps import get_lat_lon, get_exif_data

# === SETTINGS ===
image_path = "photo_real/IMG_0125.jpg"
output_dir = "output_tiles"

# Extract GPS from image EXIF
exif_data = get_exif_data(image_path)
if exif_data:
    drone_lat, drone_lon = get_lat_lon(exif_data)
else:
    print("No GPS data found in EXIF. Using default coordinates.")
#drone_lat = 41.1953210
#drone_lon = 29.2436957
altitude = 20.0  # in meters
yaw_deg = 0.0   # direction camera is facing (0 = North, 90 = East)

hfov_deg = 78
rows, cols = 8, 12  # 8x12 grid

os.makedirs(output_dir, exist_ok=True)

# === CONSTANTS ===
EARTH_RADIUS = 6378137

def pixel_to_gps(x, y, x_offset, y_offset, image_width, image_height, yaw_deg, pitch_deg, roll_deg):
    # Combine offsets for full image position
    x_full = x + x_offset
    y_full = y + y_offset

    hfov = radians(hfov_deg)
    vfov = hfov * (image_height / image_width)

    fx = image_width / (2 * tan(hfov / 2))
    fy = image_height / (2 * tan(vfov / 2))
    cx = image_width / 2
    cy = image_height / 2

    # Intrinsic
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    K_inv = np.linalg.inv(K)

    # Pixel vector in camera coordinates
    pixel = np.array([x_full, y_full, 1.0])
    ray = K_inv @ pixel
    ray = ray / np.linalg.norm(ray)

    # Full rotation matrix (yaw, pitch, roll)
    R_matrix = R.from_euler('ZYX', [radians(yaw_deg), radians(pitch_deg), radians(roll_deg)]).as_matrix()
    ray[0] = -ray[0]
    ray_world = R_matrix @ ray

    # Project to ground plane
    if ray_world[2] == 0:
        return drone_lat, drone_lon  # No valid intersection

    scale = -altitude / ray_world[2]  # Negate if Z points down
    ground_point = ray_world * scale

    offset_east = ground_point[0]
    offset_north = ground_point[1]

    dlat = offset_north / EARTH_RADIUS * (180 / pi)
    dlon = offset_east / (EARTH_RADIUS * cos(pi * drone_lat / 180)) * (180 / pi)

    return drone_lat + dlat, drone_lon + dlon

# === LOAD IMAGE ===
full_image = cv2.imread(image_path)
img_height, img_width = full_image.shape[:2]
tile_w = img_width // cols
tile_h = img_height // rows

# === LOAD YOLO MODEL ===
model = YOLO("yolo11x.pt")  # Replace with your model if needed

# === OUTPUT CSV ===
csv_file = open("detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Class", "Confidence", "Latitude", "Longitude", "TileRow", "TileCol"])

# === PROCESS TILES ===
for row in range(rows):
    for col in range(cols):
        x_start = col * tile_w
        y_start = row * tile_h
        tile = full_image[y_start:y_start+tile_h, x_start:x_start+tile_w]

        results = model(tile, conf=0.5, iou=0.5, verbose=False)
        boxes = results[0].boxes
        names = model.names

        annotated = tile.copy()

        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            cls_name = names[cls_id]
            conf = boxes.conf[i].item()
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # Center of bounding box in tile
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Calculate GPS from full image coordinates
            yaw_rad = radians(yaw_deg)
            lat, lon = pixel_to_gps(
                cx, cy, 
                x_offset=x_start, 
                y_offset=y_start, 
                image_width=img_width,
                image_height=img_height,
                yaw_deg=yaw_deg,
                pitch_deg=0.0,  # or actual pitch
                roll_deg=0.0    # or actual roll
            )

            # Annotate
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save to CSV
            csv_writer.writerow([cls_name, f"{conf:.2f}", lat, lon, row, col])

        cv2.imwrite(f"{output_dir}/tile_{row}_{col}.jpg", annotated)

csv_file.close()
print("Processing completed. Detections saved.")
