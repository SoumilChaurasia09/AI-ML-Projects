import os
import threading
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dataset import get_dataset_metadata, get_random_samples
from train import train_model, WEIGHTS_DIR
from model import CancerCNN

app = FastAPI(title="Cancer Detection CNN Dashboard")

# Serve static files from the static directory
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)

# Global states to track training thread progress
training_state = {
    "active": False,
    "dataset_name": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "metrics": None,
    "logs": [],
    "error": None
}

# Lock for safe multithreaded logging updates
state_lock = threading.Lock()

class TrainParams(BaseModel):
    dataset_name: str = Field("dermamnist", description="Dataset name, default dermamnist")
    epochs: int = Field(10, ge=1, le=50)
    learning_rate: float = Field(0.001, gt=0.0)
    batch_size: int = Field(32, ge=8, le=128)
    optimizer_name: str = Field("AdamW", pattern="^(AdamW|Adam|SGD)$")
    weight_decay: float = Field(0.0, ge=0.0)
    augment: bool = Field(True)

def training_worker(params: TrainParams):
    global training_state
    
    def log_callback(epoch, metrics, status_msg):
        with state_lock:
            training_state["current_epoch"] = epoch
            training_state["metrics"] = metrics
            training_state["logs"].append(status_msg)
            # Keep only the last 100 log lines to avoid bloating memory
            if len(training_state["logs"]) > 100:
                training_state["logs"].pop(0)

    try:
        train_model(
            dataset_name=params.dataset_name,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            batch_size=params.batch_size,
            optimizer_name=params.optimizer_name,
            weight_decay=params.weight_decay,
            augment=params.augment,
            callback=log_callback
        )
    except Exception as e:
        with state_lock:
            training_state["error"] = str(e)
            training_state["logs"].append(f"ERROR: {str(e)}")
    finally:
        with state_lock:
            training_state["active"] = False

# --- REST Endpoints ---

@app.get("/")
def read_root():
    """Serves the main dashboard page."""
    html_file = os.path.join(static_path, "index.html")
    if os.path.exists(html_file):
        response = FileResponse(html_file)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return {"message": "Server is running, but index.html was not found in static folder."}

@app.get("/api/dataset/info")
def get_dataset_info(dataset_name: str = "dermamnist"):
    """Returns dataset metadata and a few random sample images encoded in base64."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name != "dermamnist":
        raise HTTPException(status_code=400, detail="Only Skin Cancer (DermaMNIST) dataset is supported.")
    try:
        metadata = get_dataset_metadata(dataset_name)
        samples = get_random_samples(dataset_name, num_samples=8)
        return {
            "metadata": metadata,
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/train/start")
def start_training(params: TrainParams, background_tasks: BackgroundTasks):
    """Triggers model training in a background thread."""
    global training_state
    
    if params.dataset_name != "dermamnist":
        raise HTTPException(status_code=400, detail="Only Skin Cancer (DermaMNIST) training is supported.")
        
    with state_lock:
        if training_state["active"]:
            raise HTTPException(status_code=400, detail="Training is already in progress.")
        
        # Reset training state
        training_state["active"] = True
        training_state["dataset_name"] = params.dataset_name
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = params.epochs
        training_state["metrics"] = None
        training_state["logs"] = ["Initializing background training thread..."]
        training_state["error"] = None
        
    background_tasks.add_task(training_worker, params)
    return {"status": "started", "message": f"Training initiated for {params.dataset_name}."}

@app.get("/api/train/status")
def get_training_status():
    """Returns the live training metrics and logs."""
    with state_lock:
        return training_state

@app.get("/api/model/results")
def get_model_results(dataset_name: str = "dermamnist"):
    """Returns whether a model is trained, and loads cached test evaluation metrics if they exist."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name != "dermamnist":
        raise HTTPException(status_code=400, detail="Only Skin Cancer (DermaMNIST) results are supported.")
    model_path = os.path.join(WEIGHTS_DIR, f"best_{dataset_name}.pth")
    has_model = os.path.exists(model_path)
    
    results = None
    if has_model:
        results_path = os.path.join(WEIGHTS_DIR, f"results_{dataset_name}.json")
        if os.path.exists(results_path):
            try:
                import json
                with open(results_path, 'r') as f:
                    results = json.load(f)
            except Exception as e:
                pass
                
    return {
        "has_model": has_model,
        "results": results
    }

@app.post("/api/predict")
async def predict_image(
    dataset_name: str = Form("dermamnist"),
    file: UploadFile = File(...)
):
    """Loads a trained model and returns predictions for the uploaded image file."""
    # 1. Check if model exists
    dataset_name = dataset_name.lower().strip()
    if dataset_name != "dermamnist":
        raise HTTPException(
            status_code=400, 
            detail="Only Skin Cancer (DermaMNIST) predictions are supported."
        )
    model_path = os.path.join(WEIGHTS_DIR, f"best_{dataset_name}.pth")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Model for {dataset_name} has not been trained yet. Please train it first!"
        )
        
    # 2. Retrieve metadata for shape/classes
    try:
        meta = get_dataset_metadata(dataset_name)
        in_channels = meta["n_channels"]
        classes = meta["classes"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset name: {str(e)}")
        
    # 3. Read and preprocess uploaded image
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        
        # Match input channel requirements
        if in_channels == 1:
            pil_img = pil_img.convert("L")
        else:
            pil_img = pil_img.convert("RGB")
            
        # Resize to 28x28 (MedMNIST resolution)
        pil_img = pil_img.resize((28, 28))
        
        # Convert to numpy array and normalize [0, 1] -> [-1, 1]
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        mean = 0.5
        std = 0.5
        img_np = (img_np - mean) / std
        
        # Transpose HWC -> CHW if RGB
        if in_channels == 3:
            img_np = img_np.transpose(2, 0, 1)
        else:
            img_np = np.expand_dims(img_np, axis=0)  # Add channel dimension
            
        # Add batch dimension and convert to tensor
        input_tensor = torch.tensor(img_np).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
        
    # 4. Initialize and load CNN
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CancerCNN(in_channels=in_channels, num_classes=len(classes)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        # 5. Format response
        predictions = []
        for idx, prob in enumerate(probabilities):
            predictions.append({
                "class_id": idx,
                "class_name": classes[str(idx)],
                "probability": float(prob)
            })
            
        # Sort predictions by probability descending
        predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        
        return {
            "dataset": dataset_name,
            "predicted_class": predictions[0]["class_name"],
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount static files (must be at the bottom so it doesn't shadow the api endpoints)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
