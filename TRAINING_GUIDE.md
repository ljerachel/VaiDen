# Dental Cavities Detection - YOLOv5 Training Guide

This guide will help you train a YOLOv5 **object detection** model on your dental cavities dataset.

> **Note:** This guide uses `train.py` (object detection with bounding boxes), not `classify/train.py` (image classification). Since your annotations contain bounding box coordinates, you need object detection training.

## Prerequisites

1. Make sure you have all required dependencies installed:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your dataset structure is as follows:
   ```
   Dataset/
   ├── Images/
   │   ├── no_retractors/
   │   ├── pilot/
   │   └── retractors/
   └── Annotations/
       └── Darknet_YOLO/
           ├── no_retractors/
           ├── pilot/
           └── retractors/
   ```

## Step 1: Prepare Your Dataset

Run the dataset preparation script to organize your images and annotations into train/val splits:

```bash
python prepare_dental_dataset.py
```

This script will:

- Find all image-label pairs in your dataset
- Detect the number of classes from your annotations
- Split the dataset into train (80%) and validation (20%) sets
- Copy files to `datasets/dental_cavities/` directory
- Update the YAML configuration file with detected classes

**Output structure:**

```
datasets/
└── dental_cavities/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## Step 2: Review Configuration

Check and update `data/dental_cavities.yaml` if needed:

- Review class names (the script auto-detects them, but you may want to rename them)
- Verify the number of classes matches your annotations

## Step 3: Start Training

**Important:** Use the root `train.py` (object detection), NOT `classify/train.py` (image classification).

- Your annotations are in YOLO format with bounding boxes → use `train.py` (object detection)
- `classify/train.py` is for image classification (single class per image), not object detection

### Basic Training (Recommended for first run)

**For Baseline/Quick Test (10-20 epochs):**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 10 --batch-size 8
```

**For Full Training (100+ epochs):**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 16
```

### Training Options

**Model sizes:**

- `yolov5n.pt` - Nano (fastest, smallest)
- `yolov5s.pt` - Small (recommended for starting)
- `yolov5m.pt` - Medium
- `yolov5l.pt` - Large
- `yolov5x.pt` - Extra Large (best accuracy, slowest)

**Common training parameters:**

```bash
# Basic training
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100

# With custom batch size (adjust based on your GPU memory)
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 8

# With rectangular training (faster, slightly less accurate)
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --rect

# Multi-scale training (vary image size during training)
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --multi-scale

# Resume training from last checkpoint
python train.py --data data/dental_cavities.yaml --resume

# Train from scratch (no pretrained weights)
python train.py --data data/dental_cavities.yaml --weights '' --cfg models/yolov5s.yaml --img 640 --epochs 100
```

### Advanced Options

**Custom project name:**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --project runs/train --name dental_cavities_exp1
```

**Freeze backbone layers (faster training, less accuracy):**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --freeze 10
```

**Use different optimizer:**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --optimizer AdamW
```

## Step 4: Monitor Training

Training results will be saved to `runs/train/exp/` (or your custom project/name):

- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `val_batch0_labels.jpg` - Validation batch visualization

## Step 5: Validate Your Model

After training, validate the best model:

```bash
python val.py --data data/dental_cavities.yaml --weights runs/train/exp/weights/best.pt --img 640
```

## Step 6: Test Detection

Test your trained model on new images:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/test/images --img 640 --conf 0.25
```

## Troubleshooting

### Out of Memory Errors

**Quick Fix - Use AutoBatch (Recommended):**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size -1
```

**Manual Fixes:**

- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Reduce image size: `--img 416` or `--img 320`
- Use a smaller model: `yolov5n.pt` instead of `yolov5s.pt`
- Use CPU: `--device cpu` (if GPU memory insufficient)
- Reduce workers: `--workers 2`

**Safe Starting Configuration:**

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 416 --epochs 100 --batch-size 4 --workers 4
```

See `FIX_MEMORY_ERROR.md` for detailed memory troubleshooting guide.

### Poor Results

- Train for more epochs: `--epochs 200` or `--epochs 300`
- Use data augmentation (enabled by default)
- Try a larger model: `yolov5m.pt` or `yolov5l.pt`
- Check your annotations for quality and consistency

### Dataset Issues

- Ensure all images have corresponding label files
- Verify label format is correct (YOLO format: `class_id x_center y_center width height`)
- Check that class IDs in labels match the class names in your YAML file

## Tips for Better Results

1. **Data Quality**: Ensure your annotations are accurate and consistent
2. **Data Augmentation**: Already enabled by default, but you can adjust in hyperparameters
3. **Learning Rate**: Default is usually good, but you can tune it
4. **Early Stopping**: Training will stop if no improvement for 100 epochs (adjustable with `--patience`)
5. **Image Size**: Larger images (640+) generally give better accuracy but slower training

## Next Steps

After training:

1. Evaluate on validation set
2. Test on real-world images
3. Fine-tune hyperparameters if needed
4. Export model for deployment: `python export.py --weights runs/train/exp/weights/best.pt`

## Additional Resources

- YOLOv5 Documentation: https://docs.ultralytics.com/yolov5/
- Training Custom Data Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
- GitHub Repository: https://github.com/ultralytics/yolov5
