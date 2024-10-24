import os
import time
import cv2
import argparse
import numpy as np
from pycocotools.coco import COCO
from utils.coco_utils import COCO_test_helper, coco_eval_with_json
from postprocess.airockchip_yolov8_postprocess import post_process
from utils.ultralytics2coco import convert_yolo_to_coco
from config import config

def draw(image, boxes, scores, classes, CLASSES):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(b) for b in box]
        print(f"{CLASSES[str(cl)]} @ ({top} {left} {right} {bottom}) {score:.3f}")
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            f'{CLASSES[str(cl)]} {score:.2f}',
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path)
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        raise ValueError(f"{model_path} is not a valid model file (expected .pt, .torchscript, .rknn, or .onnx)")
    print(f'Model "{model_path}" is a {platform} model, starting validation')
    return model, platform

def img_check(path):
    img_types = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(path.lower().endswith(ext) for ext in img_types)

def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference Script')
    # Basic parameters
    parser.add_argument('--model_path', type=str, required=True, help='Model path (.pt, .rknn, .onnx file)')
    parser.add_argument('--img_show', action='store_true', default=False, help='Show the result')
    parser.add_argument('--img_save', action='store_true', default=False, help='Save the result')
    
    # Data parameters
    parser.add_argument('--anno_json', type=str, default='path/to/annotations/instances_val2017.json', help='COCO annotation path')
    parser.add_argument('--img_folder', type=str, default='path/to/images', help='Image folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='Enable COCO mAP test')
    
    # New parameter for Ultralytics annotations
    parser.add_argument('--ultralytics_dir', type=str, help='Ultralytics annotations directory')
    parser.add_argument('--coco_output', type=str, help='Output path for converted COCO annotations')
    
    args = parser.parse_args()
    total_latency = 0.0
    inference_count = 0
    categories = [
        {"id": int(k), "name": v} for k, v in config["CLASSES"].items()
    ]
    # Convert Ultralytics annotations to COCO if provided
    if args.ultralytics_dir and args.coco_output:
        convert_yolo_to_coco(args.ultralytics_dir, args.coco_output, categories)
        args.anno_json = args.coco_output
        print(f'Converted Ultralytics annotations to COCO format at {args.coco_output}')
    
    # Initialize model
    model, platform = setup_model(args)
    
    # Prepare image list
    img_list = sorted([f for f in os.listdir(args.img_folder) if img_check(f)])
    co_helper = COCO_test_helper(enable_letter_box=True)
    IMG_SIZE = config["IMG_SIZE"]
    CLASSES = config["CLASSES"]
    
    # Load COCO annotations if mAP test is enabled
    if args.coco_map_test:
        coco = COCO(args.anno_json)
        img_ids = coco.getImgIds()
        images = coco.loadImgs(img_ids)
        filename_to_id = {image['file_name']: image['id'] for image in images}
    
    # Run inference
    for idx, img_name in enumerate(img_list):
        start_time = time.time()
        print(f'Processing image {idx + 1}/{len(img_list)}: {img_name}', end='\r')
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_name}")
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f"Failed to read image: {img_name}")
            continue

        # Preprocess image
        pad_color = (0, 0, 0)
        img = co_helper.letter_box(img_src.copy(), new_shape=IMG_SIZE, pad_color=pad_color)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare input data
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
            input_data = input_data[np.newaxis, :]
        else:
            input_data = img[np.newaxis, :]

        # Run model
        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs, IMG_SIZE)
        end_time = time.time()
        total_latency += (end_time - start_time)
        inference_count += 1

        # Draw and save results if requested
        if args.img_show or args.img_save:
            print(f'\n\nImage: {img_name}')
            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes, CLASSES)

            if args.img_save:
                os.makedirs('./result', exist_ok=True)
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print(f'Detection result saved to {result_path}')

            if args.img_show:
                cv2.imshow("Detection Result", img_p)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Record detections for COCO mAP calculation
        if args.coco_map_test and boxes is not None:
            image_id = filename_to_id.get(img_name)
            if image_id is None:
                print(f"Image ID not found for image: {img_name}")
                continue
            for i in range(boxes.shape[0]):
                co_helper.add_single_record(
                    image_id=image_id,
                    category_id=int(classes[i]),
                    bbox=boxes[i].tolist(),
                    score=round(float(scores[i]), 5)
                )

    # Calculate mAP if requested
    if args.coco_map_test:
        pred_json = f"{os.path.splitext(os.path.basename(args.model_path))[0]}_{platform}.json"
        pred_json_path = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json_path)
        coco_eval_with_json(args.anno_json, pred_json_path)
        if inference_count > 0:
            avg_latency = total_latency / inference_count
            print(f'Average latency per image: {avg_latency:.3f} seconds')

if __name__ == "__main__":
    main()
    