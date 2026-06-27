# Face Detection and Recognition System

This is a complete, self-contained Python web application that does two things:
1. **Face Detection**: Finds the location of a face in an image and draws a bounding box around it.
2. **Face Recognition**: Identifies who the face belongs to by comparing it against registered faces.

The project uses custom Convolutional Neural Networks (CNNs) built from scratch in PyTorch, with a FastAPI backend and a clean, responsive web interface that supports live webcam detection.

---

## How It Works

This system divides face recognition into three steps:

1. **Detection**: The image is fed into a custom `FaceDetectorCNN`. The model determines if a face exists and predicts the exact coordinates of the bounding box `[xmin, ymin, xmax, ymax]`.
2. **Cropping & Resizing**: The detected face is cropped and resized to 64x64 pixels.
3. **Embedding (Siamese Network)**: The cropped face is passed through `FaceEmbeddingCNN` to generate a 128-dimensional embedding vector. We compare this vector against our database of registered faces using Euclidean distance. If the distance is below a set threshold, it's a match!

Because we use embeddings, **you can register new faces dynamically without having to retrain the neural networks.**

---

## Folder Structure

* `static/` - Frontend files (HTML layout, CSS styles, and JavaScript for webcam and canvas drawing)
* `main.py` - FastAPI backend server
* `model.py` - Custom PyTorch CNN architectures (Detector and Embedder)
* `dataset.py` - Synthetic face generator for training (no massive datasets to download)
* `train.py` - Python script to train the networks
* `test_pipeline.py` - A quick sanity check script to verify everything compiles and runs
* `requirements.txt` - Python dependencies list

---

## How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Verify everything is working
Run the test script to make sure PyTorch and other components are working fine:
```bash
python test_pipeline.py
```

### 3. Start the Web Server
Launch the FastAPI server:
```bash
python main.py
```

### 4. Open the Web App
Open your browser and navigate to:
**http://127.0.0.1:8004**

*(The server is configured to try and open this link automatically in your default browser on startup).*

---

## Features

- **Live Webcam Recognition**: Click "Start Camera" in the sandbox tab to try real-time face tracking.
- **Dynamic Register**: Type a name, upload a photo (or generate a synthetic face), and register it instantly.
- **No Large Downloads**: Generates and trains on synthetic face drawings locally, so you don't need to download gigabytes of dataset files to run the project.
