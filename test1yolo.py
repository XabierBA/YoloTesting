from ultralytics import YOLO
from PIL import Image
import cv2

#MODELING START
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

#PREDICTIONS
res = model.predict("bus.jpg")[0]

#IMAGE EDIT AND SAVE
print(res.boxes)
res = res.plot(line_width=3)
res = res[:, :, :: -1]
res = Image.fromarray(res)
res.save("output1.jpg")

#IMAGE SHOWING CODE
path = cv2.imread("output1.jpg")
cv2.imshow("Result", path)
cv2.waitKey(0)
cv2.destroyAllWindows()