import tkinter as tk
from tkintermapview import TkinterMapView
import csv

# === CONFIGURATION ===
CSV_FILE = "detections.csv"
CENTER_LAT = 41.1956  # You can auto-compute from data if needed
CENTER_LON = 29.2437
ZOOM = 17

# === LOAD CSV DETECTIONS ===
detections = []
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
            print(f"Skipping row due to error: {e}")

# === TKINTER GUI SETUP ===
root = tk.Tk()
root.title("UAV Detections Map")
root.geometry("1000x800")

map_widget = TkinterMapView(root, width=1000, height=800, corner_radius=0)
map_widget.pack(fill="both", expand=True)
map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)  # google satellite

map_widget.set_position(CENTER_LAT, CENTER_LON)
map_widget.set_zoom(ZOOM)

# === ADD MARKERS ===
for lat, lon, label in detections:
    map_widget.set_marker(lat, lon, text=label)

root.mainloop()
