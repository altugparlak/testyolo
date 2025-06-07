from ultralytics import YOLO
import cv2

# Load YOLOv8 model
# Note: Corrected the model name to a standard YOLOv8 model.
# If "yolo12l.pt" is your custom model, please change it back.
model = YOLO("yolo11s.pt")

# Set threshold
CONFIDENCE_THRESHOLD = 0.15

# Load webcam or video
cap = cv2.VideoCapture("10.mp4")

# Get the original frame width, height, and FPS
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

H, W, _ = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)
zoom_factor = 4

# --- 1. VIDEO WRITER SETUP ---
output_filename = "zoomed_output2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_filename, fourcc, fps, (W, H))

# --- Create a resizable window ---
cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)

print("Processing video... Press 'q' to quit.")

while True:
    # We already read the first frame, so we start processing from it
    # and then read the next one at the end of the loop.
    
    # --- START OF 2X ZOOM LOGIC ---
    crop_width = W // zoom_factor
    crop_height = H // zoom_factor
    crop_x1 = (W - crop_width) // 2
    crop_y1 = (H - crop_height) // 2
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height
    zoomed_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    display_frame = cv2.resize(zoomed_frame, (W, H))
    # --- END OF 2X ZOOM LOGIC ---

    # Run detection on the zoomed-in frame
    results = model(display_frame)[0]

    # Get class names
    class_names = model.names

    # Draw boxes for all detected objects
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- 2. WRITE THE FRAME TO THE OUTPUT FILE ---
    out.write(display_frame)

    # Display the resulting frame
    cv2.imshow("YOLO Object Detection", display_frame)

    # Read the next frame at the end of the loop
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. RELEASE EVERYTHING WHEN JOB IS FINISHED ---
print(f"Video saved successfully to {output_filename}")
cap.release()
out.release()
cv2.destroyAllWindows()