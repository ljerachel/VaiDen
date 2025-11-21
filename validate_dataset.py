# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate dental cavities dataset for YOLOv5 training.
This script checks for common issues before training.
"""

from collections import Counter
from pathlib import Path

import yaml

# Load YAML config
yaml_path = Path("data/dental_cavities.yaml")
print("=" * 60)
print("Dataset Validation")
print("=" * 60)

# 1. Check YAML file
print("\n1. Checking YAML configuration...")
try:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    required_keys = ["path", "train", "val", "names", "nc"]
    for key in required_keys:
        if key not in data:
            print(f"   ❌ Missing key: {key}")
        else:
            print(f"   ✓ Found key: {key}")

    print(f"   Classes defined: {data.get('nc')}")
    print(f"   Class names: {data.get('names')}")
except Exception as e:
    print(f"   ❌ Error loading YAML: {e}")
    exit(1)

# 2. Check dataset paths
print("\n2. Checking dataset paths...")
path = Path(data["path"])
if not path.is_absolute():
    path = (Path.cwd() / path).resolve()

print(f"   Dataset root: {path}")
print(f"   Exists: {path.exists()}")

train_path = path / data["train"]
val_path = path / data["val"]

print(f"\n   Train path: {train_path}")
print(f"   Exists: {train_path.exists()}")

print(f"\n   Val path: {val_path}")
print(f"   Exists: {val_path.exists()}")

# 3. Check images and labels
print("\n3. Checking images and labels...")
train_images = list((path / "images" / "train").glob("*.*"))
train_labels = list((path / "labels" / "train").glob("*.txt"))
val_images = list((path / "images" / "val").glob("*.*"))
val_labels = list((path / "labels" / "val").glob("*.txt"))

print(f"   Train images: {len(train_images)}")
print(f"   Train labels: {len(train_labels)}")
print(f"   Val images: {len(val_images)}")
print(f"   Val labels: {len(val_labels)}")

# 4. Check for matching image-label pairs
print("\n4. Checking image-label pairs...")
train_image_names = {img.stem for img in train_images}
train_label_names = {label.stem for label in train_labels}
missing_labels = train_image_names - train_label_names
missing_images = train_label_names - train_image_names

if missing_labels:
    print(f"   ⚠️  {len(missing_labels)} images without labels (showing first 5):")
    for name in list(missing_labels)[:5]:
        print(f"      - {name}")
if missing_images:
    print(f"   ⚠️  {len(missing_images)} labels without images (showing first 5):")
    for name in list(missing_images)[:5]:
        print(f"      - {name}")
if not missing_labels and not missing_images:
    print("   ✓ All images have corresponding labels")

# 5. Check class IDs in labels
print("\n5. Checking class IDs in labels...")
class_ids_found = Counter()
total_labels = 0
errors = []

for label_file in list(train_labels)[:100] + list(val_labels)[:50]:  # Sample
    try:
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_ids_found[class_id] += 1
                        total_labels += 1

                        # Check if coordinates are normalized
                        coords = [float(x) for x in parts[1:5]]
                        if any(c < 0 or c > 1 for c in coords):
                            errors.append(f"{label_file.name}: coordinates out of range")
                    else:
                        errors.append(f"{label_file.name}: invalid format")
    except Exception as e:
        errors.append(f"{label_file.name}: {e}")

print(f"   Class IDs found: {dict(class_ids_found)}")
print(f"   Total labels checked: {total_labels}")

# Check if class IDs match YAML
yaml_class_ids = set(data["names"].keys())
found_class_ids = set(class_ids_found.keys())
missing_in_yaml = found_class_ids - yaml_class_ids
missing_in_labels = yaml_class_ids - found_class_ids

if missing_in_yaml:
    print(f"   ❌ Class IDs in labels not in YAML: {missing_in_yaml}")
if missing_in_labels:
    print(f"   ⚠️  Class IDs in YAML not found in labels: {missing_in_labels}")
if not missing_in_yaml and not missing_in_labels:
    print("   ✓ All class IDs match YAML configuration")

if errors:
    print(f"\n   ⚠️  Found {len(errors)} errors (showing first 5):")
    for error in errors[:5]:
        print(f"      - {error}")

# 6. Test dataset loading
print("\n6. Testing dataset loading with YOLOv5...")
try:
    from utils.general import check_dataset

    result = check_dataset(data, autodownload=False)
    print("   ✓ Dataset check passed!")
    print(f"   Train path resolved: {result.get('train')}")
    print(f"   Val path resolved: {result.get('val')}")
except Exception as e:
    print(f"   ❌ Dataset check failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Validation complete!")
print("=" * 60)
