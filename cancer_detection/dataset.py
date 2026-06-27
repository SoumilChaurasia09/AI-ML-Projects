import io
import base64
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, Evaluator
from PIL import Image

def get_dataset_class(dataset_name: str):
    """
    Returns the dataset class from the medmnist module based on its key name.
    """
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "breastmnist":
        return getattr(medmnist, "BreastMNIST")
    elif dataset_name == "dermamnist":
        return getattr(medmnist, "DermaMNIST")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'breastmnist' or 'dermamnist'.")

def get_dataset_metadata(dataset_name: str):
    """
    Retrieves metadata dictionary for the specified MedMNIST dataset.
    """
    info = INFO[dataset_name.lower().strip()]
    return {
        "name": info.get("python_class", dataset_name.capitalize()),
        "description": info["description"],
        "n_channels": info["n_channels"],
        "classes": info["label"],  # dict of class index -> name
        "n_samples": info["n_samples"],
        "task": info["task"]
    }

def get_dataloaders(dataset_name: str, batch_size: int = 32, augment: bool = True):
    """
    Downloads and loads the specified dataset, returning train, val, and test DataLoaders.
    """
    dataset_name = dataset_name.lower().strip()
    info = INFO[dataset_name]
    n_channels = info["n_channels"]
    
    # Define normalization based on channels
    mean = [0.5] * n_channels
    std = [0.5] * n_channels
    
    # Transformations
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip() if dataset_name == "breastmnist" else transforms.Compose([]),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    DatasetClass = get_dataset_class(dataset_name)
    
    # Load splits
    train_dataset = DatasetClass(split="train", transform=train_transform, download=True)
    val_dataset = DatasetClass(split="val", transform=eval_transform, download=True)
    test_dataset = DatasetClass(split="test", transform=eval_transform, download=True)
    
    # Create DataLoaders
    # Note: drop_last=True helps avoid batch-size of 1 for BatchNorm stability if train size % batch_size == 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_random_samples(dataset_name: str, num_samples: int = 8):
    """
    Loads the dataset and returns a list of random samples containing the base64-encoded image
    and the corresponding class label.
    """
    dataset_name = dataset_name.lower().strip()
    DatasetClass = get_dataset_class(dataset_name)
    
    # Load dataset without normalized tensors to easily save raw PIL images
    dataset = DatasetClass(split="train", download=True)
    info = INFO[dataset_name]
    labels_map = info["label"]
    
    total_imgs = len(dataset)
    indices = random.sample(range(total_imgs), min(num_samples, total_imgs))
    
    samples = []
    for idx in indices:
        # MedMNIST dataset returns (PIL Image, numpy array label)
        img, label = dataset[idx]
        
        # Convert label array to scalar int
        label_idx = int(label[0])
        label_name = labels_map[str(label_idx)]
        
        # Save image to bytes buffer as PNG
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        samples.append({
            "image": f"data:image/png;base64,{img_str}",
            "label_id": label_idx,
            "label_name": label_name
        })
        
    return samples

if __name__ == "__main__":
    # Quick verification
    meta = get_dataset_metadata("breastmnist")
    print("BreastMNIST Meta:", meta)
    samples = get_random_samples("breastmnist", num_samples=2)
    print(f"Retrieved {len(samples)} samples. Image encoded length: {len(samples[0]['image'])}")
