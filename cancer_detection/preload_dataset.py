import os
import medmnist
from dataset import DATA_DIR, get_dataset_class

def preload():
    print(f"Pre-downloading DermaMNIST dataset to: {DATA_DIR}")
    # Initialize dataset splits to trigger download to the custom directory
    DatasetClass = get_dataset_class("dermamnist")
    DatasetClass(split="train", download=True, root=DATA_DIR)
    DatasetClass(split="val", download=True, root=DATA_DIR)
    DatasetClass(split="test", download=True, root=DATA_DIR)
    print("DermaMNIST preloaded successfully!")

if __name__ == "__main__":
    preload()
