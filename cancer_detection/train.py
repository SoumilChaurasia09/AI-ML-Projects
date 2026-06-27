import os
import io
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server safety
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from medmnist import INFO, Evaluator

from model import CancerCNN
from dataset import get_dataloaders, get_dataset_metadata

# Directory to save trained model weights
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def train_model(dataset_name: str, 
                epochs: int = 10, 
                learning_rate: float = 0.001, 
                batch_size: int = 32, 
                optimizer_name: str = "Adam", 
                weight_decay: float = 0.0,
                augment: bool = True,
                callback=None):
    """
    Trains the CancerCNN on the specified dataset.
    Reports progress using the callback function: callback(epoch, metrics_dict, status_msg)
    """
    dataset_name = dataset_name.lower().strip()
    meta = get_dataset_metadata(dataset_name)
    in_channels = meta["n_channels"]
    num_classes = len(meta["classes"])
    class_names = [meta["classes"][str(i)] for i in range(num_classes)]
    
    # 1. Load Data
    if callback:
        callback(0, None, "Loading datasets and preparing data loaders...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        augment=augment
    )
    
    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CancerCNN(in_channels=in_channels, num_classes=num_classes, pretrained=True).to(device)
    
    # 3. Loss & Optimizer (with Class Weighting to handle imbalance)
    # MedMNIST labels are (N, 1) vectors, we need to squeeze them to (N,) for CrossEntropyLoss
    try:
        # Calculate class weights from training dataset labels
        labels_arr = train_loader.dataset.labels.flatten().astype(int)
        class_counts = np.bincount(labels_arr)
        total_samples = len(labels_arr)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        if callback:
            callback(0, None, f"Calculated training class weights: {class_weights.tolist()}")
    except Exception as e:
        criterion = nn.CrossEntropyLoss()
        if callback:
            callback(0, None, f"Warning: could not calculate class weights ({e}). Using standard cross-entropy loss.")
            
    # Set up optimizer (Default: AdamW)
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay if weight_decay > 0 else 1e-2)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        
    # Learning rate scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
    best_val_acc = 0.0
    best_model_path = os.path.join(WEIGHTS_DIR, f"best_{dataset_name}.pth")
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            # MedMNIST label is long tensor of shape [batch_size, 1]
            labels = labels.squeeze(1).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze(1).long().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # Save best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            
        # Step the Cosine Annealing scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        status_msg = f"Epoch {epoch}/{epochs} (LR: {current_lr:.6f}) - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        
        if callback:
            metrics = {
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "history": history
            }
            callback(epoch, metrics, status_msg)
            
    # 5. Load best weights and evaluate on clean Test Split
    if callback:
        callback(epochs, None, f"Training complete. Loading best weights ({best_val_acc:.4f} Val Acc) for test evaluation...")
        
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    test_acc = test_correct / test_total
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 6. Compute Detailed Metrics
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    # 7. Generate plots
    # Plot A: Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    cm_buf = io.BytesIO()
    plt.savefig(cm_buf, format='png', dpi=100)
    plt.close()
    cm_base64 = base64.b64encode(cm_buf.getvalue()).decode('utf-8')
    
    # Plot B: ROC Curve
    plt.figure(figsize=(6, 5))
    if num_classes == 2:
        # Binary ROC (Malignant class is index 1)
        fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkviolet', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.title('Receiver Operating Characteristic (ROC)')
    else:
        # Multi-class ROC
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve((all_targets == i).astype(int), all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f'{class_names[i][:15]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.title('ROC Curves (Per Class)')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    
    roc_buf = io.BytesIO()
    plt.savefig(roc_buf, format='png', dpi=100)
    plt.close()
    roc_base64 = base64.b64encode(roc_buf.getvalue()).decode('utf-8')
    
    results = {
        "test_acc": test_acc,
        "classification_report": report,
        "confusion_matrix_img": f"data:image/png;base64,{cm_base64}",
        "roc_curve_img": f"data:image/png;base64,{roc_base64}"
    }
    
    # Save results to a persistent JSON file so they are loaded automatically on server restart
    try:
        import json
        results_path = os.path.join(WEIGHTS_DIR, f"results_{dataset_name}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        print(f"Warning: could not save training results to JSON ({e})")
    
    if callback:
        callback(epochs, {
            "epoch": epochs,
            "epochs": epochs,
            "train_loss": history["train_loss"][-1],
            "train_acc": history["train_acc"][-1],
            "val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][-1],
            "history": history,
            "results": results
        }, "Training and evaluation completed successfully!")
        
    return results
