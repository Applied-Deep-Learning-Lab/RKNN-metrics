import os
import json
from PIL import Image


def is_image_file(filename):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    return filename.lower().endswith(image_extensions)

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    
    return [x_min, y_min, width, height]

def convert_yolo_to_coco(dir, output_json, categories):
    images          = []
    annotations     = []
    annotation_id   = 1
    img_dir         = os.path.join(dir, "images")
    label_dir       = os.path.join(dir, "labels")
    for img_filename in os.listdir(img_dir):
        if is_image_file(img_filename):
            img_path = os.path.join(img_dir, img_filename)
            root, _ = os.path.splitext(img_filename)
            label_path = os.path.join(label_dir, root+".txt")

            img = Image.open(img_path)
            img_width, img_height = img.size
            image_id = len(images) + 1

            images.append({
                "id": image_id,
                "file_name": img_filename,
                "width": img_width,
                "height": img_height
            })

            if os.path.exists(label_path):
                with open(label_path, 'r') as label_file:
                    for line in label_file.readlines():
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        
                        bbox = yolo_to_coco_bbox([x_center, y_center, width, height], img_width, img_height)
                        area = bbox[2] * bbox[3]  # width * height

                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": bbox,
                            "area": area,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as outfile:
        json.dump(coco_format, outfile, indent=4)
