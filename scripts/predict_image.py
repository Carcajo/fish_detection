# import cv2
# from ultralytics import YOLO
#
# model = YOLO('model/best.pt')
# img_path = 'rory.jpg'
# img = cv2.imread(img_path)
#
# results = model(img)
# annotated_img = results[0].plot()
# cv2.imwrite('path_to_output_image.jpg', annotated_img)

from ultralytics import YOLO

model = YOLO('model/best2.pt')
image_path = 'photo.jpg'

results = model.predict(source=image_path, conf=0.1, save=True)

for result in results:
    result.show()
