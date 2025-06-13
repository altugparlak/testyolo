import tkinter as tk
from tkintermapview import TkinterMapView
import csv
import os
import time
import threading

# === CONFIGURATION ===
CSV_FILE = "detections.csv"
CENTER_LAT = 41.1956
CENTER_LON = 29.2437
ZOOM = 17
UPDATE_INTERVAL = 1000  # milliseconds

# === STATE ===
last_mod_time = 0
existing_markers = []

# === GUI SETUP ===
root = tk.Tk()
root.title("UAV Detections Map")
root.geometry("1000x800")

map_widget = TkinterMapView(root, width=1000, height=800, corner_radius=0)
map_widget.pack(fill="both", expand=True)
map_widget.set_tile_server(
    "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22
)

map_widget.set_position(CENTER_LAT, CENTER_LON)
map_widget.set_zoom(ZOOM)

# === FUNCTIONS ===

def load_detections():
    detections = []
    try:
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row["Latitude"])
                    lon = float(row["Longitude"])
                    cls = row["Class"]
                    conf = float(row["Confidence"])
                    tile_row = row["TileRow"]
                    tile_col = row["TileCol"]
                    label = f"{cls} ({conf:.2f}) [Tile {tile_row},{tile_col}]"
                    detections.append((lat, lon, label))
                except Exception as e:
                    print(f"[WARN] Skipping invalid row: {e}")
    except FileNotFoundError:
        print(f"[WARN] CSV file not found: {CSV_FILE}")
    return detections

def update_map():
    global last_mod_time, existing_markers

    try:
        mod_time = os.path.getmtime(CSV_FILE)
    except FileNotFoundError:
        root.after(UPDATE_INTERVAL, update_map)
        return

    if mod_time != last_mod_time:
        print("[INFO] CSV file updated. Reloading...")
        last_mod_time = mod_time

        # Remove old markers
        for marker in existing_markers:
            marker.delete()
        existing_markers.clear()

        # Add new markers
        new_detections = load_detections()
        for lat, lon, label in new_detections:
            marker = map_widget.set_marker(lat, lon, text=label)
            existing_markers.append(marker)

    root.after(UPDATE_INTERVAL, update_map)

# === START LOOP ===
print("[INFO] Starting real-time CSV watcher...")
update_map()
root.mainloop()
