from ultralytics import YOLO
from PIL import Image
import cv2

#MODELING START
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

#PREDICTIONS
res = model.track(source="0",show=True)

# El programa marca las manos de las personas en la imagen poniendo puntos en cada falange
for r in res:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    for box in r.boxes:  # iterate through detected objects
        if box.cls == "hand":  # check if the detected object is a hands
            keypoints = box.keypoints  # get keypoints for the hand
            for point in keypoints:
                x, y = int(point[0]), int(point[1])
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                cv2.circle(im_array, (x, y), 5, (0, 255, 0), -1)  # draw points on each phalanx    im.show()  # show image
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image

# #IMAGE SHOWING CODEsult", res)
cv2.imshow("Result", res)


cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()