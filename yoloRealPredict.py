from ultralytics import YOLO
from PIL import Image
import cv2

#MODELING START
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

#PREDICTIONS
res = model.predict(source="0",show=True)


# View results
for r in res:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image


# #IMAGE SHOWING CODE
cv2.imshow("Result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()