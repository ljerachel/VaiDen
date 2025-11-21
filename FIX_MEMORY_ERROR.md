# Fixing Out of Memory (OOM) Errors in YOLOv5 Training

## Error Explanation

The `RuntimeError: [enforce fail at alloc_cpu.c` error occurs when your system runs out of memory (RAM or VRAM) during training. This happens when:

- Batch size is too large
- Image size is too large
- Model is too large for available memory
- System doesn't have enough RAM/VRAM

## Quick Fixes (Try in Order)

### 1. Reduce Batch Size (Easiest Fix)

Start with a very small batch size and increase gradually:

```bash
# Try batch size 4 first
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 4

# If that works, try 8
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 8
```

### 2. Use AutoBatch (Recommended)

Let YOLOv5 automatically find the optimal batch size:

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size -1
```

The `-1` tells YOLOv5 to automatically determine the best batch size for your system.

### 3. Reduce Image Size

Smaller images use less memory:

```bash
# Try 416 instead of 640
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 100 --batch-size 8

# Or even smaller
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 320 --epochs 100 --batch-size 8
```

### 4. Use a Smaller Model

Switch to a smaller model that uses less memory:

```bash
# Use nano model (smallest)
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 640 --epochs 100 --batch-size 8

# Or small model with smaller batch
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 100 --batch-size 4
```

### 5. Use CPU Training (If GPU Memory is Insufficient)

If you're using GPU and it's running out of memory, try CPU (slower but uses RAM):

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 100 --batch-size 4 --device cpu
```

### 6. Reduce Workers

Fewer data loading workers can help:

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 100 --batch-size 4 --workers 2
```

### 7. Disable Image Caching

If you're using `--cache ram`, try disabling it:

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --epochs 100 --batch-size 4 --cache
```

## Recommended Starting Command

For most systems, start with this safe configuration:

```bash
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 416 --epochs 100 --batch-size 4 --workers 4
```

Then gradually increase:

- If successful, try `--batch-size 8`
- If still successful, try `--img 512`
- If still successful, try `yolov5s.pt` instead of `yolov5n.pt`

## Memory Usage Guidelines

**For CPU Training (RAM):**

- yolov5n: ~2-4 GB RAM
- yolov5s: ~4-8 GB RAM
- yolov5m: ~8-16 GB RAM

**For GPU Training (VRAM):**

- yolov5n: ~2-4 GB VRAM
- yolov5s: ~4-6 GB VRAM
- yolov5m: ~6-10 GB VRAM

**Batch Size Impact:**

- Each batch size increase roughly doubles memory usage
- Image size 640 uses ~2.5x more memory than 416

## Check Your System Memory

**Windows:**

```powershell
# Check RAM
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory

# Check GPU VRAM (if you have NVIDIA GPU)
nvidia-smi
```

**Python:**

```python
import psutil

print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
```

## Advanced Solutions

### Gradient Accumulation (Simulate Larger Batch)

If you need a larger effective batch size but don't have memory:

```python
# This requires modifying train.py or using a custom training script
# Effective batch size = batch_size * accumulate
```

### Mixed Precision Training

Already enabled by default, but you can verify AMP is working:

```bash
# AMP should be automatically enabled if supported
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --batch-size 4
```

## Troubleshooting Steps

1. **Start with the smallest configuration:**

   ```bash
   python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 320 --batch-size 2 --device cpu
   ```

2. **If that works, gradually increase:**
   - Increase batch size: `--batch-size 4`
   - Increase image size: `--img 416`
   - Switch to GPU: `--device 0` (if you have GPU)
   - Use larger model: `--weights yolov5s.pt`

3. **Monitor memory usage** while training to see where it fails

## Example: Progressive Training Setup

```bash
# Step 1: Test with minimal settings
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 320 --batch-size 2 --epochs 1

# Step 2: If successful, increase batch size
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 320 --batch-size 4 --epochs 1

# Step 3: If successful, increase image size
python train.py --data data/dental_cavities.yaml --weights yolov5n.pt --img 416 --batch-size 4 --epochs 1

# Step 4: If successful, use larger model
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --batch-size 4 --epochs 1

# Step 5: Full training with optimal settings
python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 416 --batch-size 4 --epochs 100
```
