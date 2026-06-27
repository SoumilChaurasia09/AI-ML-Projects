import os
import json
import threading
import torch
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from model import FaceDetectorCNN, FaceEmbeddingCNN
from dataset import SyntheticFaceGenerator
from train import train_all

import webbrowser

app = FastAPI(title="Face Detection & Recognition CNN API")

@app.on_event("startup")
def open_browser():
    import time
    def _open():
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:8004")
        except Exception as e:
            print(f"Failed to auto-open browser: {e}")
    threading.Thread(target=_open, daemon=True).start()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and database
detector_model = None
embedder_model = None
database_path = "registered_faces.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Attempts to load the trained weights for detector and embedder."""
    global detector_model, embedder_model
    
    det_path = "weights/detector.pth"
    emb_path = "weights/embedder.pth"
    
    if os.path.exists(det_path) and os.path.exists(emb_path):
        try:
            # Load Detector
            detector_model = FaceDetectorCNN().to(device)
            detector_model.load_state_dict(torch.load(det_path, map_location=device))
            detector_model.eval()
            
            # Load Embedder
            embedder_model = FaceEmbeddingCNN().to(device)
            embedder_model.load_state_dict(torch.load(emb_path, map_location=device))
            embedder_model.eval()
            
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            detector_model = None
            embedder_model = None
            return False
    else:
        print("Model weights not found. Training required.")
        return False

# Attempt to load models on startup
models_loaded = load_models()

def load_database():
    """Loads registered faces from JSON file."""
    if os.path.exists(database_path):
        try:
            with open(database_path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_database(db):
    """Saves registered faces to JSON file."""
    with open(database_path, "w") as f:
        json.dump(db, f, indent=2)

def preseed_database_if_needed():
    """Pre-seeds the database with some synthetic identities so the user has data to play with."""
    db = load_database()
    if len(db) > 0 or not load_models():
        return
        
    print("Pre-seeding database with synthetic faces...")
    generator = SyntheticFaceGenerator(crop_size=64)
    preseed_names = ["Sophia", "Jackson", "Emma", "Aiden", "Olivia"]
    
    for idx, name in enumerate(preseed_names):
        # Generate a face image for this identity ID (e.g. 500 + idx)
        identity_id = 500 + idx
        _, _, face_crop = generator.generate_image(identity_id=identity_id, has_face=True)
        
        # Convert face crop to tensor
        crop_arr = np.array(face_crop).astype(np.float32) / 255.0
        crop_tensor = torch.tensor(crop_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = embedder_model(crop_tensor).squeeze().cpu().numpy().tolist()
            
        db.append({
            "name": name,
            "identity_id": identity_id,
            "embedding": embedding
        })
    save_database(db)
    print("Database pre-seeded with 5 synthetic faces!")

# Run preseed after a short delay or after models are verified
if models_loaded:
    preseed_database_if_needed()


@app.get("/api/status")
def get_status():
    global models_loaded
    if not models_loaded:
        models_loaded = load_models()
        if models_loaded:
            preseed_database_if_needed()
            
    db = load_database()
    
    # Load training status
    training_status = {"running": False, "message": "Idle"}
    if os.path.exists("training_status.json"):
        try:
            with open("training_status.json", "r") as f:
                training_status = json.load(f)
        except Exception:
            pass
            
    # Load training history
    history = {"detector": [], "recognizer": []}
    if os.path.exists("training_history.json"):
        try:
            with open("training_history.json", "r") as f:
                history = json.load(f)
        except Exception:
            pass

    return {
        "models_trained": models_loaded,
        "device": str(device),
        "registered_count": len(db),
        "registered_names": [face["name"] for face in db],
        "training_status": training_status,
        "history": history
    }


# Background training execution thread
training_thread = None

def run_training_wrapper():
    global models_loaded
    try:
        train_all()
        models_loaded = load_models()
        preseed_database_if_needed()
    except Exception as e:
        print(f"Background training failed: {e}")

@app.post("/api/train")
def trigger_training():
    global training_thread
    
    # Check if already running
    if os.path.exists("training_status.json"):
        try:
            with open("training_status.json", "r") as f:
                status = json.load(f)
                if status.get("running", False):
                    return {"status": "already_running", "message": "Training is already in progress."}
        except Exception:
            pass
            
    training_thread = threading.Thread(target=run_training_wrapper)
    training_thread.start()
    return {"status": "started", "message": "Model training started in the background."}


@app.get("/api/generate-sample")
def generate_sample(identity_id: int = None, has_face: bool = True):
    """
    Generates a synthetic face image and returns it as JSON with base64/RGB arrays,
    specifically for demonstration purposes on the frontend.
    """
    generator = SyntheticFaceGenerator()
    img, bbox, _ = generator.generate_image(identity_id=identity_id, has_face=has_face)
    
    # Save PIL image to base64 string
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    import base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "image": f"data:image/jpeg;base64,{img_str}",
        "bbox": list(bbox),
        "has_face": has_face,
        "identity_id": identity_id
    }


def preprocess_image(image_bytes: bytes, target_size=128):
    """Helper to convert uploaded image bytes to PIL and PyTorch Tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size
    
    # Resize for CNN
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return img, img_tensor, original_size


@app.post("/api/detect")
async def detect_face(file: UploadFile = File(...)):
    global detector_model
    if detector_model is None:
        if not load_models():
            return JSONResponse(status_code=400, content={"error": "Models not trained. Please train the models first."})
            
    try:
        contents = await file.read()
        pil_img, tensor, (orig_w, orig_h) = preprocess_image(contents, target_size=128)
        
        with torch.no_grad():
            cls_logits, bbox = detector_model(tensor)
            prob = torch.sigmoid(cls_logits).item()
            bbox = bbox.squeeze().cpu().numpy().tolist()
            
        has_face = prob >= 0.5
        
        # Bounding box is [xmin, ymin, xmax, ymax] in normalized [0, 1] coordinates
        # Map to original image size
        scaled_bbox = []
        if has_face:
            scaled_bbox = [
                max(0, int(bbox[0] * orig_w)),
                max(0, int(bbox[1] * orig_h)),
                min(orig_w, int(bbox[2] * orig_w)),
                min(orig_h, int(bbox[3] * orig_h))
            ]
            
        return {
            "face_detected": has_face,
            "confidence": prob,
            "bbox": scaled_bbox,
            "normalized_bbox": bbox
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/register")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    global detector_model, embedder_model
    if detector_model is None or embedder_model is None:
        if not load_models():
            return JSONResponse(status_code=400, content={"error": "Models not trained. Please train the models first."})
            
    try:
        contents = await file.read()
        pil_img, tensor, _ = preprocess_image(contents, target_size=128)
        
        # 1. Detect face
        with torch.no_grad():
            cls_logits, bbox = detector_model(tensor)
            prob = torch.sigmoid(cls_logits).item()
            bbox = bbox.squeeze().cpu().numpy().tolist()
            
        if prob < 0.5:
            return JSONResponse(status_code=400, content={"error": "No face detected in the image. Registration aborted."})
            
        # 2. Crop face from the original image (with 10% margin)
        orig_w, orig_h = pil_img.size
        xmin, ymin, xmax, ymax = bbox
        
        margin_x = (xmax - xmin) * 0.1
        margin_y = (ymax - ymin) * 0.1
        
        crop_xmin = max(0, int((xmin - margin_x) * orig_w))
        crop_ymin = max(0, int((ymin - margin_y) * orig_h))
        crop_xmax = min(orig_w, int((xmax + margin_x) * orig_w))
        crop_ymax = min(orig_h, int((ymax + margin_y) * orig_h))
        
        face_crop = pil_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        face_crop = face_crop.resize((64, 64), Image.Resampling.BILINEAR)
        
        # 3. Generate face embedding
        crop_arr = np.array(face_crop).astype(np.float32) / 255.0
        crop_tensor = torch.tensor(crop_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = embedder_model(crop_tensor).squeeze().cpu().numpy().tolist()
            
        # 4. Save to DB
        db = load_database()
        
        # Overwrite if name already exists, else append
        existing = next((item for item in db if item["name"].lower() == name.lower()), None)
        if existing:
            existing["embedding"] = embedding
        else:
            db.append({
                "name": name,
                "embedding": embedding
            })
            
        save_database(db)
        return {"status": "success", "message": f"Successfully registered face for '{name}'."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/api/recognize")
async def recognize_face(file: UploadFile = File(...)):
    global detector_model, embedder_model
    if detector_model is None or embedder_model is None:
        if not load_models():
            return JSONResponse(status_code=400, content={"error": "Models not trained. Please train the models first."})
            
    try:
        contents = await file.read()
        pil_img, tensor, (orig_w, orig_h) = preprocess_image(contents, target_size=128)
        
        # 1. Detect face
        with torch.no_grad():
            cls_logits, bbox = detector_model(tensor)
            prob = torch.sigmoid(cls_logits).item()
            bbox = bbox.squeeze().cpu().numpy().tolist()
            
        if prob < 0.5:
            return {
                "face_detected": False,
                "name": "No face detected",
                "distance": 1.0,
                "bbox": []
            }
            
        # 2. Crop face (with 10% margin)
        xmin, ymin, xmax, ymax = bbox
        margin_x = (xmax - xmin) * 0.1
        margin_y = (ymax - ymin) * 0.1
        
        crop_xmin = max(0, int((xmin - margin_x) * orig_w))
        crop_ymin = max(0, int((ymin - margin_y) * orig_h))
        crop_xmax = min(orig_w, int((xmax + margin_x) * orig_w))
        crop_ymax = min(orig_h, int((ymax + margin_y) * orig_h))
        
        face_crop = pil_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        face_crop = face_crop.resize((64, 64), Image.Resampling.BILINEAR)
        
        # 3. Extract embedding
        crop_arr = np.array(face_crop).astype(np.float32) / 255.0
        crop_tensor = torch.tensor(crop_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = embedder_model(crop_tensor).squeeze().cpu().numpy()
            
        # 4. Compare with DB
        db = load_database()
        if len(db) == 0:
            return {
                "face_detected": True,
                "name": "Database empty",
                "distance": 1.0,
                "bbox": [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            }
            
        best_name = "Unknown"
        min_dist = float("inf")
        
        for record in db:
            ref_emb = np.array(record["embedding"])
            
            # Euclidean distance between L2-normalized embeddings is directly related to cosine similarity
            # Distance is between 0 and 2.0 (since they are L2 normalized)
            dist = np.linalg.norm(embedding - ref_emb)
            
            if dist < min_dist:
                min_dist = dist
                best_name = record["name"]
                
        # Set recognition threshold. A threshold of 0.6 is typical for L2-normalized triplet embeddings.
        threshold = 0.65
        match_name = best_name if min_dist < threshold else "Unknown"
        
        return {
            "face_detected": True,
            "name": match_name,
            "distance": float(min_dist),
            "bbox": [crop_xmin, crop_ymin, crop_xmax, crop_ymax],
            "all_distances": [{r["name"]: float(np.linalg.norm(embedding - np.array(r["embedding"])))} for r in db]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


# Serve Web Frontend (if static directory exists)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
