from ultralytics import YOLO

model = YOLO('model/best2.pt')
image_path = 'photo.jpg'

results = model.predict(source=image_path, conf=0.1, save=True)

for result in results:
    result.show()
