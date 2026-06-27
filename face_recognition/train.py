import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FaceDetectionDataset, FaceRecognitionTripletDataset
from model import FaceDetectorCNN, FaceEmbeddingCNN

# Status file path for live tracking
STATUS_FILE = "training_status.json"
HISTORY_FILE = "training_history.json"

def update_status(status_dict):
    """Writes the current training status to a JSON file for the API and frontend."""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print(f"Error updating status file: {e}")

def load_history():
    """Loads existing training history or returns an empty dict."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"detector": [], "recognizer": []}

def save_history(history_dict):
    """Saves training history to a JSON file."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        print(f"Error saving history file: {e}")

def train_detector(epochs=8, batch_size=32, lr=0.001):
    """Trains the custom Face Detection CNN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Face Detector on {device}...")
    
    # Initialize status
    status = {
        "stage": "detector",
        "current_epoch": 0,
        "total_epochs": epochs,
        "progress": 0.0,
        "loss": 0.0,
        "cls_loss": 0.0,
        "bbox_loss": 0.0,
        "running": True,
        "message": "Initializing Detector training..."
    }
    update_status(status)
    
    # Load dataset
    dataset = FaceDetectionDataset(num_samples=800, has_face_ratio=0.8)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FaceDetectorCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    cls_criterion = nn.BCEWithLogitsLoss()
    bbox_criterion = nn.SmoothL1Loss()
    
    history = load_history()
    history["detector"] = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_bbox_loss = 0.0
        
        for batch_idx, (imgs, is_face, bboxes) in enumerate(train_loader):
            imgs = imgs.to(device)
            is_face = is_face.to(device).unsqueeze(1) # shape [B, 1]
            bboxes = bboxes.to(device) # shape [B, 4]
            
            optimizer.zero_grad()
            cls_logits, pred_bboxes = model(imgs)
            
            # Classification Loss
            loss_cls = cls_criterion(cls_logits, is_face)
            
            # Bounding Box Loss (only compute where is_face == 1)
            face_mask = (is_face == 1.0).squeeze()
            if face_mask.sum() > 0:
                loss_bbox = bbox_criterion(pred_bboxes[face_mask], bboxes[face_mask])
            else:
                loss_bbox = torch.tensor(0.0).to(device)
                
            # Combined Loss
            loss = loss_cls + 2.0 * loss_bbox
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_cls_loss += loss_cls.item()
            epoch_bbox_loss += loss_bbox.item()
            
            # Progress calculation
            prog = ((epoch * len(train_loader)) + batch_idx + 1) / (epochs * len(train_loader)) * 100.0
            status.update({
                "current_epoch": epoch + 1,
                "progress": round(prog, 1),
                "loss": round(loss.item(), 4),
                "cls_loss": round(loss_cls.item(), 4),
                "bbox_loss": round(loss_bbox.item(), 4),
                "message": f"Detector: Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}"
            })
            update_status(status)
            
        avg_loss = epoch_loss / len(train_loader)
        avg_cls = epoch_cls_loss / len(train_loader)
        avg_bbox = epoch_bbox_loss / len(train_loader)
        
        history["detector"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cls_loss": avg_cls,
            "bbox_loss": avg_bbox
        })
        save_history(history)
        
    # Save model weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/detector.pth")
    print("Detector weights saved to weights/detector.pth")
    
    status.update({
        "message": "Detector training completed!",
        "progress": 100.0
    })
    update_status(status)
    return model

def train_recognizer(epochs=10, batch_size=32, lr=0.001):
    """Trains the custom Face Embedding CNN using Siamese Triplet Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Face Embedding Model on {device}...")
    
    # Initialize status
    status = {
        "stage": "recognizer",
        "current_epoch": 0,
        "total_epochs": epochs,
        "progress": 0.0,
        "loss": 0.0,
        "running": True,
        "message": "Initializing Embedding Model training..."
    }
    update_status(status)
    
    # Load dataset (15 identities, 12 images each)
    dataset = FaceRecognitionTripletDataset(num_identities=15, samples_per_identity=12)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FaceEmbeddingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
    
    history = load_history()
    history["recognizer"] = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            
            a_embed = model(anchors)
            p_embed = model(positives)
            n_embed = model(negatives)
            
            loss = triplet_loss_fn(a_embed, p_embed, n_embed)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Progress calculation
            prog = ((epoch * len(train_loader)) + batch_idx + 1) / (epochs * len(train_loader)) * 100.0
            status.update({
                "current_epoch": epoch + 1,
                "progress": round(prog, 1),
                "loss": round(loss.item(), 4),
                "message": f"Recognizer: Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}"
            })
            update_status(status)
            
        avg_loss = epoch_loss / len(train_loader)
        
        history["recognizer"].append({
            "epoch": epoch + 1,
            "loss": avg_loss
        })
        save_history(history)
        
    # Save model weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/embedder.pth")
    print("Recognizer weights saved to weights/embedder.pth")
    
    status.update({
        "message": "Recognizer training completed!",
        "progress": 100.0,
        "running": False
    })
    update_status(status)
    return model

def train_all():
    """Sequentially trains both models and saves status."""
    try:
        train_detector(epochs=8)
        # Add a short delay
        time.sleep(1)
        train_recognizer(epochs=10)
    except Exception as e:
        status = {
            "running": False,
            "error": str(e),
            "message": f"Training failed: {e}"
        }
        update_status(status)
        raise e

if __name__ == "__main__":
    train_all()
