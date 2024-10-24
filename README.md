# RKNN metrics

This repository allows you to evaluate results using COCO metrics.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Inference](#running-inference)
  - [Converting Annotations](#converting-annotations)
- [Modules](#modules)
- [Example](#example)

## Features

- **Multi-Model Support**: Compatible with RKNN (`.rknn`), and ONNX (`.onnx`) models.
- **Annotation Conversion**: Convert Ultralytics-style (`.txt`) annotations to COCO format.
- **Configurable Parameters**: Easily adjust thresholds, image sizes, and class labels via `config.json`.
- **COCO Evaluation**: Compute COCO mAP metrics using `pycocotools`.
- **Modular Design**: Organized into separate modules for preprocessing, postprocessing, and annotation conversion for easier maintenance and extension.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Applied-Deep-Learning-Lab/rknn-metrics.git
   cd rknn-metrics
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

All configurable parameters are stored in `config.json`. Modify this file to adjust object detection thresholds, image sizes, and class labels.

### `config.json`

```json
{
    "OBJ_THRESH": 0.3,
    "NMS_THRESH": 0.5,
    "IMG_SIZE": [960, 960],
    "CLASSES": {
        "0": "bike",
        "1": "bus",
        "2": "car",
        "3": "construction equipment",
        "4": "emergency",
        "5": "motorbike",
        "6": "personal mobility",
        "7": "quadbike",
        "8": "truck"
    }
}
```

- **OBJ_THRESH**: Object confidence threshold.
- **NMS_THRESH**: Non-Maximum Suppression threshold.
- **IMG_SIZE**: Target image size for inference (width, height).
- **CLASSES**: Mapping of class IDs to class names.

## Usage

### Running Inference

Use `main.py` to perform object detection inference on a set of images. The script supports displaying and saving results, as well as evaluating using COCO mAP metrics.

#### Command-Line Arguments

- `--model_path` (str, required): Path to the model file (`.rknn`, `.onnx`).
- `--img_show` (flag): Display detection results onscreen.
- `--img_save` (flag): Save detection results to disk (default: `./result`).
- `--anno_json` (str): Path to COCO annotation file (default: `path/to/annotations/instances_val2017.json`).
- `--img_folder` (str): Path to the folder containing images (default: `path/to/images`).
- `--coco_map_test` (flag): Enable COCO mAP evaluation.
- `--ultralytics_dir` (str): Path to Ultralytics `.txt` annotations directory.
- `--coco_output` (str): Output path for converted COCO annotations.

#### Example Command

```bash
python main.py \
    --model_path models/yolov8.rknn \
    --img_folder data/valid/images \
    --img_show \
    --img_save \
    --coco_map_test \
    --anno_json data/valid/instances_val2017.json
```

### Converting Annotations

To convert Ultralytics-style `.txt` annotations to COCO format, use the `--ultralytics_dir` and `--coco_output` arguments.

#### Example Command

```bash
python main.py \
    --model_path models/yolov8.rknn \
    --img_folder data/valid/images \
    --ultralytics_dir data/ \
    --coco_output data/valid/coco_annotations.json \
    --coco_map_test
```

This command will:

1. Convert Ultralytics `.txt` annotations in `data/valid/labels` to COCO format and save to `data/valid/coco_annotations.json`.
2. Use the converted annotations for COCO mAP evaluation.

## Modules

### `airockchip_yolov8_postprocess.py`

Processes raw model outputs into final detection boxes, classes, and scores specifically for modified YOLOv8 by AIRockchip.

- **Functions**:
  - `filter_boxes(boxes, box_confidences, box_class_probs)`: Filters boxes based on object confidence threshold.
  - `nms_boxes(boxes, scores)`: Applies NMS to suppress overlapping boxes.
  - `dfl(position: np.ndarray)`: Decodes position data.
  - `box_process(position, IMG_SIZE)`: Converts model output to bounding boxes.
  - `post_process(input_data, IMG_SIZE)`: Aggregates and filters detections.

### `annotation_converter.py`

Converts Ultralytics-style `.txt` annotations to COCO format.

- **Functions**:
  - `ultralytics_to_coco(ultralytics_dir, coco_output_path, categories)`: Performs the conversion.

### `config.py`

Loads configuration parameters from `config.json`.

- **Variables**:
  - `config`: Dictionary containing configuration parameters.

### `main.py`

The main script to perform inference, handle annotation conversion, and evaluate results.

- **Key Steps**:
  1. Parse command-line arguments.
  2. Convert annotations if Ultralytics format is provided.
  3. Initialize the specified model.
  4. Perform inference on each image.
  5. Optionally display and/or save results.
  6. Record detections for COCO mAP evaluation.
  7. Compute and display mAP and average latency.

## Example

1. **Prepare Data**

   - Place your images in `data/valid/images`.
   - If using Ultralytics annotations, place `.txt` files in `data/valid/ultralytics`.

2. **Configure `config.json`**

   Adjust thresholds, image sizes, and class labels as needed.

3. **Run Inference and Evaluation**

   ```bash
   python main.py \
       --model_path models/yolov8.rknn \
       --img_folder data/valid/images \
       --ultralytics_dir data/valid/ultralytics \
       --coco_output data/valid/coco_annotations.json \
       --img_save \
       --coco_map_test
   ```

4. **View Results**

   - Saved images with detections will be in the `./result` directory.
   - COCO evaluation metrics will be printed in the console.
