# Quick Baseline Training Guide

For getting a baseline performance model quickly, use fewer epochs. This is perfect for:

- Testing if your dataset is set up correctly
- Getting initial results quickly
- Checking if training works on your system
- Understanding model performance before committing to long training

## Quick Baseline Commands

### Minimal Baseline (Fastest - ~5-10 minutes)

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 416 --epochs 5 --batch-size 4
```

### Standard Baseline (Recommended - ~15-30 minutes)

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 10 --batch-size 4
```

### Better Baseline (More accurate - ~30-60 minutes)

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 20 --batch-size 4
```

## Why Fewer Epochs for Baseline?

1. **Faster iteration**: Test your setup in minutes, not hours
2. **Lower memory usage**: Fewer epochs = less cumulative memory pressure
3. **Quick validation**: See if your dataset and config are correct
4. **Early stopping**: If model isn't learning, you'll know quickly

## Epoch Guidelines

- **5 epochs**: Ultra-quick test, just to see if training runs
- **10 epochs**: Good baseline, shows if model is learning
- **20 epochs**: Better baseline, more reliable metrics
- **50 epochs**: Decent model, good for initial deployment
- **100+ epochs**: Full training, best performance

## After Baseline Training

Once you have a baseline:

1. **Check the results** in `runs/train/exp/`:
   - `results.png` - Training curves
   - `confusion_matrix.png` - How well it's detecting
   - `val_batch0_labels.jpg` - Visual predictions

2. **If results look good**, train longer:

   ```bash
   python train.py --data data/dental_cavities.yaml --weights runs/train/exp/weights/last.pt --epochs 100 --batch-size 4
   ```

3. **If results are poor**, check:
   - Dataset quality
   - Annotation accuracy
   - Class balance
   - Image quality

## Memory-Safe Baseline Command

If you're still having memory issues, use this:

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 320 --epochs 10 --batch-size 2 --workers 2
```

This uses minimal resources and will complete quickly.
