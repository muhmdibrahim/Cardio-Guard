from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.

# Save the model for future use
model.save('best.pt')
