from ultralytics import YOLO

model = YOLO('model/yolov8n.pt')

model.train(data='/home/maxim/PycharmProjects/alfa_vision/dataset.yaml', epochs=100, imgsz=640, batch=8, name='deepfish_model')

model.save('model/best2.pt')
