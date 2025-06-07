import cv2
import os
from ultralytics import YOLO

# Load the image
image_path = 'photo/photo8.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Divide into 2 rows and 3 columns
rows, cols = 8, 12
section_height = height // rows
section_width = width // cols

# Load the model
model = YOLO("yolo11l.pt")  # or your custom model path

# Create output directory
os.makedirs("results", exist_ok=True)

# Loop through each section
for i in range(rows):
    for j in range(cols):
        x_start = j * section_width
        y_start = i * section_height
        x_end = x_start + section_width
        y_end = y_start + section_height

        # Crop the section
        section = image[y_start:y_end, x_start:x_end]

        # Run YOLO on the section
        results = model(section)

        # Save the result image (with bounding boxes)
        result_image = results[0].plot()
        output_path = f"results/section_{i}_{j}.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Saved: {output_path}")
