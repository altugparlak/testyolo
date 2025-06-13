import cv2
import numpy as np
import os
import time
import csv
import threading
import queue
import glob
from ultralytics import YOLO
from math import tan, radians, cos, pi
from scipy.spatial.transform import Rotation as R
from extract_gps import get_lat_lon, get_exif_data

# ==== CONFIGURATION ====
watch_folder = "photo_real_test"
output_csv = "detections.csv"
output_dir = "output_tiles"
model = YOLO("yolo12x.pt")

CONFIDENCE_THRESHOLD = 0.3
ROWS, COLS = 8, 12
HFOV_DEG = 78
ALTITUDE = 20.0
YAW_DEG = 0.0

EARTH_RADIUS = 6378137
processed_files = set()
image_queue = queue.Queue()

os.makedirs(output_dir, exist_ok=True)

CLASS_NAMES = {
    "person", "car", "motorcycle", "airplane", "bus", "boat",
    "stop sign", "snowboard", "umbrella", "sports ball", "baseball bat",
    "bed", "tennis racket", "suitcase", "skis"
}

# ==== HELPER FUNCTIONS ====

def pixel_to_gps(x, y, x_offset, y_offset, image_width, image_height, yaw_deg, pitch_deg, roll_deg):
    # Combine offsets for full image position
    x_full = x + x_offset
    y_full = y + y_offset

    hfov = radians(HFOV_DEG)
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
        return start_lat, start_lon  # No valid intersection

    scale = -ALTITUDE / ray_world[2]  # Negate if Z points down
    ground_point = ray_world * scale

    offset_east = ground_point[0]
    offset_north = ground_point[1]

    dlat = offset_north / EARTH_RADIUS * (180 / pi)
    dlon = offset_east / (EARTH_RADIUS * cos(pi * start_lat / 180)) * (180 / pi)

    return start_lat + dlat, start_lon + dlon

def process_image(path):
    global start_lat, start_lon

    print(f"\n[INFO] Reading image: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Failed to load image.")

    print("[INFO] Extracting GPS data...")
    exif = get_exif_data(path)
    start_lat, start_lon = get_lat_lon(exif)
    print(f"[INFO] GPS Origin: ({start_lat:.6f}, {start_lon:.6f})")

    h, w, _ = img.shape
    tile_h, tile_w = h // ROWS, w // COLS
    results = []

    print("[INFO] Splitting image into tiles and detecting objects...")
    for row in range(ROWS):
        for col in range(COLS):
            tile = img[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w]
            det = model(tile, verbose=False)[0]

            for box in det.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                class_name = model.names[int(cls)]

                #if conf < CONFIDENCE_THRESHOLD or class_name not in CLASS_NAMES:
                 #   continue
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                lat, lon = pixel_to_gps(cx, cy, col * tile_w, row * tile_h, w, h, YAW_DEG, 0, 0)

                results.append([class_name, round(conf, 2), lat, lon, row, col])

    print(f"[INFO] Detected {len(results)} objects.")
    return results

# ==== WORKER THREAD ====

def image_worker():
    while True:
        path = image_queue.get()
        if path is None:
            print("[INFO] Worker thread received stop signal.")
            break

        try:
            print(f"[INFO] Processing file from queue: {os.path.basename(path)}")
            results = process_image(path)

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                for row in results:
                    writer.writerow(row)
                f.flush()
            print(f"[SUCCESS] Results written to {output_csv} for {os.path.basename(path)}")

        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")

        image_queue.task_done()

# ==== SETUP CSV HEADER ====
if not os.path.exists(output_csv):
    print(f"[INFO] Creating output CSV: {output_csv}")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Confidence", "Latitude", "Longitude", "TileRow", "TileCol"])

# ==== START WORKER THREAD ====
threading.Thread(target=image_worker, daemon=True).start()

print(f"[READY] Watching folder: {watch_folder}")

# ==== MAIN MONITOR LOOP ====
try:
    while True:
        all_images = sorted(glob.glob(os.path.join(watch_folder, "*.jpg")))
        new_images = [f for f in all_images if f not in processed_files]

        for img_path in new_images:
            print(f"[NEW] Found new image: {img_path}")
            processed_files.add(img_path)
            image_queue.put(img_path)

        time.sleep(1)

except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    image_queue.put(None)
    image_queue.join()
    print("[DONE] All images processed. Exiting.")
