import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import tarfile
import urllib.request

# Import our custom CNN architecture
from model import SimpleCNN

# CIFAR-10 class labels
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class SyntheticCIFAR10(Dataset):
    """
    A synthetic CIFAR-10 dataset fallback.
    
    If both the real dataset and the S3 mirror cannot be downloaded due to network issues,
    this class generates random CIFAR-like images so that the code can still be verified.
    """
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.num_samples = 1000 if train else 200
        self.data = np.random.randint(0, 256, size=(self.num_samples, 32, 32, 3), dtype=np.uint8)
        self.targets = np.random.randint(0, 10, size=(self.num_samples,)).tolist()
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def download_and_extract_s3(dest_dir="./data"):
    """Downloads and extracts CIFAR-10 from a fast Amazon S3 mirror."""
    s3_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    archive_path = os.path.join(dest_dir, "cifar10.tgz")
    extracted_folder = os.path.join(dest_dir, "cifar10")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.exists(extracted_folder):
        print(f"Downloading CIFAR-10 from fast Amazon S3 mirror: {s3_url}...")
        # Custom progress callback for urlretrieve
        pbar = tqdm(unit='B', unit_scale=True, desc="Downloading S3 Mirror")
        def progress_callback(blocks, block_size, total_size):
            pbar.total = total_size
            pbar.update(blocks * block_size - pbar.n)
            
        urllib.request.urlretrieve(s3_url, archive_path, reporthook=progress_callback)
        pbar.close()
        
        print("Extracting CIFAR-10 S3 mirror archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
        
        # Clean up the archive file to save space
        if os.path.exists(archive_path):
            os.remove(archive_path)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch over the dataset."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        current_loss = running_loss / total
        current_acc = 100.0 * correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Evaluates the model on the validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def plot_curves(history, output_path):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Train a Custom CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--synthetic", action="store_true", help="Force using synthetic CIFAR-10 dataset (skips downloading)")
    parser.add_argument("--s3-mirror", action="store_true", help="Force using the fast Amazon S3 mirror instead of Toronto")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("Note: Training on CPU may be slow. Consider running with fewer epochs (e.g. --epochs 2) to test.")

    # Data Augmentation and Data Loaders
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    train_set = None
    val_set = None
    
    # Load dataset
    if args.synthetic:
        print("Using Synthetic CIFAR-10 dataset (forced by user).")
        train_set = SyntheticCIFAR10(train=True, transform=train_transform)
        val_set = SyntheticCIFAR10(train=False, transform=val_transform)
    else:
        # Try S3 mirror
        if args.s3_mirror:
            try:
                download_and_extract_s3(dest_dir="./data")
                train_set = torchvision.datasets.ImageFolder(root='./data/cifar10/train', transform=train_transform)
                val_set = torchvision.datasets.ImageFolder(root='./data/cifar10/test', transform=val_transform)
                print("Loaded CIFAR-10 from Amazon S3 mirror successfully.")
            except Exception as e:
                print(f"S3 mirror load failed: {e}. Falling back to torchvision download...")
        
        # Try standard download
        if train_set is None:
            try:
                print("Attempting to load real CIFAR-10 via torchvision (will download from Toronto if not present)...")
                train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
                val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
                print("Real CIFAR-10 loaded successfully via torchvision.")
            except Exception as e:
                print(f"Torchvision download failed: {e}.")
                
                # Auto-fallback to S3 mirror if torchvision fails
                if not args.s3_mirror:
                    try:
                        print("Attempting auto-fallback to Amazon S3 mirror...")
                        download_and_extract_s3(dest_dir="./data")
                        train_set = torchvision.datasets.ImageFolder(root='./data/cifar10/train', transform=train_transform)
                        val_set = torchvision.datasets.ImageFolder(root='./data/cifar10/test', transform=val_transform)
                        print("Loaded CIFAR-10 from Amazon S3 mirror successfully.")
                    except Exception as s3_err:
                        print(f"S3 auto-fallback failed: {s3_err}")
        
        # Final fallback to synthetic data if all else fails
        if train_set is None:
            print("\n" + "="*80)
            print("WARNING: All real dataset downloads failed (network restricted or server unreachable).")
            print("Falling back to a Synthetic CIFAR-10 dataset so the training script can complete.")
            print("="*80 + "\n")
            train_set = SyntheticCIFAR10(train=True, transform=train_transform)
            val_set = SyntheticCIFAR10(train=False, transform=val_transform)
            
    # Set up Data Loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model, Optimizer, and Scheduler
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training Loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    checkpoint_name = "cifar10_cnn.pth"
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  --> Saved new best model checkpoint! (Val Acc: {best_val_acc:.2f}%)")
            
    print("\nTraining complete!")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    
    # 4. Save training plot
    plot_curves(history, "training_curves.png")

if __name__ == "__main__":
    main()
