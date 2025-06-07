from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11l.pt")

video_path = "C:/Users/alpar/OneDrive/Masaüstü/yolotest/10.mp4"
# Run inference on a video file
results = model.predict(source="photo/photo2.jpg", save=True, conf=0.5, show=True)
