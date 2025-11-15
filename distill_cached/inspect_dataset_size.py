"""
Quick script to inspect the original image size of tsbpp/fall2025_deeplearning dataset

Run this script to check the original image dimensions:
    python inspect_dataset_size.py
"""

from datasets import load_dataset
from PIL import Image

print("Loading dataset...")
try:
    dataset = load_dataset("tsbpp/fall2025_deeplearning", split="pretrain")
    print("Using 'pretrain' split")
except ValueError:
    try:
        dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
        print("Using 'train' split")
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        exit(1)

print(f"Dataset loaded: {len(dataset):,} images\n")

# Sample images to check sizes
print("Inspecting image sizes (sampling 20 images)...")
sizes = []
for i in range(min(20, len(dataset))):
    try:
        item = dataset[i]
        if isinstance(item, dict):
            image = item["image"]
        else:
            image = item
        
        if isinstance(image, Image.Image):
            size = image.size  # (width, height)
            sizes.append(size)
            if i < 5:  # Print first 5
                print(f"  Image {i}: {size[0]}x{size[1]}")
        else:
            print(f"  Image {i}: Unexpected type - {type(image)}")
    except Exception as e:
        print(f"  Image {i}: Error - {e}")

if sizes:
    # Get unique sizes and their counts
    from collections import Counter
    size_counts = Counter(sizes)
    
    print(f"\n{'='*50}")
    print("Image Size Summary:")
    print(f"{'='*50}")
    for size, count in size_counts.most_common():
        print(f"  {size[0]}x{size[1]}: {count} images ({count*100/len(sizes):.1f}%)")
    
    if len(size_counts) == 1:
        size = list(size_counts.keys())[0]
        print(f"\n✓ All sampled images are: {size[0]}x{size[1]}")
        print(f"  Original image size: {size[0]}x{size[1]} pixels")
    else:
        most_common = size_counts.most_common(1)[0]
        print(f"\n⚠️  Multiple image sizes detected")
        print(f"  Most common: {most_common[0][0]}x{most_common[0][1]} ({most_common[1]} images)")
else:
    print("\n⚠️  Could not determine image sizes")
