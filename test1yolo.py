from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

res = model.predict("test2.jpeg")[0]

print(res.boxes)
res = res.plot(line_width=1)

res = res[:, :, :: -1]

res = Image.fromarray(res)

res.save("output1.jpg")

path = cv2.imread("output1.jpg")

cv2.imshow("Result", path)

cv2.waitKey(0)

cv2.destroyAllWindows()