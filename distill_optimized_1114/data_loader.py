"""
Data loading module for DINO training.

IMPORTANT: If you have a file named 'datasets.py' in your project directory,
it will conflict with the HuggingFace 'datasets' package. 
Please rename or delete any local 'datasets.py' file.
"""

import sys
from pathlib import Path

# Check if there's a conflicting local datasets.py file
current_dir = Path(__file__).parent.absolute()
local_datasets_file = current_dir / 'datasets.py'

if local_datasets_file.exists():
    raise ImportError(
        f"ERROR: Found a local file 'datasets.py' at {local_datasets_file}\n"
        f"This conflicts with the HuggingFace 'datasets' package.\n"
        f"Please rename or delete this file. Suggested name: 'datasets_old.py' or 'datasets_backup.py'"
    )

# Import HuggingFace datasets package
try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        f"Failed to import 'load_dataset' from HuggingFace datasets package.\n"
        f"Make sure you have installed it: pip install datasets\n"
        f"Original error: {e}"
    )

from torch.utils.data import Dataset
import torch

def _inspect_dataset_splits(dataset_name):
    """Helper function to inspect available splits in a dataset"""
    try:
        dataset_dict = load_dataset(dataset_name, split=None)
        if isinstance(dataset_dict, dict):
            return list(dataset_dict.keys())
        else:
            # If it's a DatasetDict, get the keys
            return list(dataset_dict.keys()) if hasattr(dataset_dict, 'keys') else ["train"]
    except:
        return ["train"]  # Default fallback

class PretrainDataset(Dataset):
    """Unlabeled pretraining dataset (~500k images)"""
    def __init__(self, transform=None):
        print("Loading pretraining dataset...")
        # Try different possible split names
        try:
            # First try 'pretrain' split
            self.dataset = load_dataset("tsbpp/fall2025_deeplearning", split="pretrain")
        except ValueError:
            try:
                # If that fails, try 'train' split (which might be the unlabeled data)
                self.dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
                print("Note: Using 'train' split for pretraining (assuming it's unlabeled)")
            except ValueError as e:
                # If both fail, show available splits
                available_splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                raise ValueError(
                    f"Could not find pretraining split. Available splits: {available_splits}\n"
                    f"Please check the dataset structure. Original error: {e}"
                )
        
        # Set format for faster access (if supported)
        try:
            self.dataset = self.dataset.with_format("pil")  # Keep as PIL for transforms
        except:
            pass  # If not supported, continue without
        
        self.transform = transform
        print(f"Loaded {len(self.dataset)} pretraining images")
    
    def __getitem__(self, idx):
        # Optimized access - try direct indexing first
        try:
            item = self.dataset[idx]
            # Handle both dict and direct image access
            if isinstance(item, dict):
                image = item["image"]
            else:
                image = item
        except:
            # Fallback to slower method if needed
            image = self.dataset[idx]["image"]
        
        if self.transform:
            views = self.transform(image)  # Multi-crop returns list
        else:
            views = image
        return views
    
    def __len__(self):
        return len(self.dataset)


class EvalDataset(Dataset):
    """Labeled evaluation dataset"""
    def __init__(self, split="train", transform=None):
        print(f"Loading evaluation dataset ({split})...")
        # Try different possible split names
        try:
            # First try eval_public/{split}
            self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                       split=f"eval_public/{split}")
        except ValueError:
            try:
                # If that fails, try loading the full dataset and accessing a subset
                full_dataset = load_dataset("tsbpp/fall2025_deeplearning")
                if isinstance(full_dataset, dict):
                    # If it's a dict, try to find eval_public key
                    if "eval_public" in full_dataset:
                        self.dataset = full_dataset["eval_public"][split]
                    else:
                        # Fallback: use the split directly if it exists
                        self.dataset = full_dataset[split]
                else:
                    raise ValueError("Unexpected dataset structure")
            except (ValueError, KeyError) as e:
                # If all fails, show available splits
                available_splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                raise ValueError(
                    f"Could not find evaluation split 'eval_public/{split}'. "
                    f"Available splits: {available_splits}\n"
                    f"Please check the dataset structure. Original error: {e}"
                )
        self.transform = transform
        print(f"Loaded {len(self.dataset)} {split} images")
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)

