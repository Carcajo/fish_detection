import json
import os
from pathlib import Path

coco_json_path = 'dataset/images/train/_annotations.coco.json'
output_dir = 'dataset/labels'
img_dir = 'dataset/photos'
    # Преобразование координат в формат YOLO

os.makedirs(output_dir, exist_ok=True)

with open(coco_json_path) as f:
    coco_data = json.load(f)


def convert_bbox_coco_to_yolo(image_width, image_height, bbox):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height
    ]


images = {image['id']: image for image in coco_data['images']}
categories = {category['id']: category for category in coco_data['categories']}


for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']

    image_info = images[image_id]
    image_width = image_info['width']
    image_height = image_info['height']
    image_file_name = image_info['file_name']

    yolo_bbox = convert_bbox_coco_to_yolo(image_width, image_height, bbox)

    label_file_name = Path(image_file_name).stem + '.txt'
    label_file_path = os.path.join(output_dir, label_file_name)

    with open(label_file_path, 'a') as f:
        f.write(f"{category_id - 1} " + " ".join(map(str, yolo_bbox)) + '\n')
