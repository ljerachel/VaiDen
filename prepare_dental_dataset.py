# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Prepare dental cavities dataset for YOLOv5 training.
This script organizes images and annotations into train/val splits.
"""

import random
import shutil
from pathlib import Path

import yaml as yaml_lib

# Configuration
DATASET_ROOT = Path("Dataset")
IMAGES_DIR = DATASET_ROOT / "Images"
ANNOTATIONS_DIR = DATASET_ROOT / "Annotations" / "Darknet_YOLO"
OUTPUT_DIR = Path("datasets") / "dental_cavities"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SEED = 42

# Image formats supported by YOLOv5
IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dng", ".webp", ".mpo"}


def find_all_images_and_labels():
    """Find all image files and their corresponding label files."""
    image_label_pairs = []

    # Walk through all image directories
    for category_dir in IMAGES_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        for view_dir in category_dir.iterdir():
            if not view_dir.is_dir():
                continue

            # Find all images in this directory
            for img_file in view_dir.iterdir():
                if img_file.suffix.lower() in IMG_FORMATS:
                    # Find corresponding label file
                    rel_path = img_file.relative_to(IMAGES_DIR)
                    # Construct label path
                    label_file = ANNOTATIONS_DIR / rel_path.with_suffix(".txt")

                    if label_file.exists():
                        image_label_pairs.append((img_file, label_file))
                    else:
                        print(f"Warning: Label file not found for {img_file}")

    return image_label_pairs


def get_class_names_from_labels(label_files):
    """Extract unique class IDs from label files."""
    class_ids = set()
    for label_file in label_files:
        try:
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except Exception as e:
            print(f"Error reading {label_file}: {e}")

    return sorted(class_ids)


def create_dataset_structure():
    """Create the output directory structure."""
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)


def copy_files(image_label_pairs, split_name, indices):
    """Copy image and label files to the appropriate split directory."""
    for idx in indices:
        img_file, label_file = image_label_pairs[idx]

        # Copy image
        img_dest = OUTPUT_DIR / "images" / split_name / img_file.name
        shutil.copy2(img_file, img_dest)

        # Copy label
        label_dest = OUTPUT_DIR / "labels" / split_name / label_file.name
        shutil.copy2(label_file, label_dest)


def main():
    """Main function to prepare the dataset."""
    print("=" * 60)
    print("Dental Cavities Dataset Preparation for YOLOv5")
    print("=" * 60)

    # Check if directories exist
    if not IMAGES_DIR.exists():
        print(f"Error: Images directory not found: {IMAGES_DIR}")
        return

    if not ANNOTATIONS_DIR.exists():
        print(f"Error: Annotations directory not found: {ANNOTATIONS_DIR}")
        return

    # Find all image-label pairs
    print("\n1. Finding all images and labels...")
    image_label_pairs = find_all_images_and_labels()
    print(f"   Found {len(image_label_pairs)} image-label pairs")

    if len(image_label_pairs) == 0:
        print("Error: No image-label pairs found!")
        return

    # Get class names
    print("\n2. Extracting class information...")
    label_files = [pair[1] for pair in image_label_pairs]
    class_ids = get_class_names_from_labels(label_files)
    print(f"   Found {len(class_ids)} classes: {class_ids}")

    # Create output structure
    print("\n3. Creating output directory structure...")
    create_dataset_structure()

    # Split dataset
    print("\n4. Splitting dataset...")
    random.seed(SEED)
    n_total = len(image_label_pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = n_total - n_train

    indices = list(range(n_total))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"   Train: {n_train} images ({TRAIN_RATIO * 100:.1f}%)")
    print(f"   Val:   {n_val} images ({VAL_RATIO * 100:.1f}%)")

    # Copy files
    print("\n5. Copying files...")
    copy_files(image_label_pairs, "train", train_indices)
    print("   ✓ Training files copied")
    copy_files(image_label_pairs, "val", val_indices)
    print("   ✓ Validation files copied")

    # Create class names mapping
    print("\n6. Generating class names...")
    # Map class IDs to sequential indices (YOLOv5 requires sequential class IDs starting from 0)
    # Default class names - you can modify these based on your actual classes
    class_names = {}
    for idx, class_id in enumerate(class_ids):
        if class_id == 0:
            class_names[idx] = "cavity"
        elif class_id == 1:
            class_names[idx] = "cavity_type_1"  # Modify based on your actual class names
        else:
            class_names[idx] = f"cavity_type_{class_id}"

    print(f"   Class mapping: {class_names}")
    print("   Note: Class IDs in annotations will be remapped to sequential indices (0, 1, 2, ...)")

    # Update YAML file with detected classes
    print("\n7. Updating YAML configuration file...")
    yaml_path = Path("data/dental_cavities.yaml")
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_data = yaml_lib.safe_load(f)

        # Update class names and number of classes
        yaml_data["names"] = class_names
        yaml_data["nc"] = len(class_names)

        with open(yaml_path, "w") as f:
            yaml_lib.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"   ✓ Updated {yaml_path} with {len(class_names)} classes")
    else:
        print(f"   ⚠ Warning: {yaml_path} not found. Please create it manually.")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review and update the class names in data/dental_cavities.yaml if needed")
    print(
        "2. Run training with: python train.py --data data/dental_cavities.yaml --weights yolov5s.pt --img 640 --epochs 100"
    )
    print("=" * 60)

    return class_names, len(class_ids)


if __name__ == "__main__":
    class_names, num_classes = main()
