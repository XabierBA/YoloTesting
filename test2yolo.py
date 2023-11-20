from ultralytics import YOLO

model = YOLO("yolov8m.pt")

res = model.track(source = "test.mp4", show = True)
