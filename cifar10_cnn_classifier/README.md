# CIFAR-10 Image Classifier (PyTorch + Flask)

A VGG-style CNN to classify images into the 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

I built this with a lightweight Flask web interface, allowing you to drag & drop images to get predictions and probability bars in real-time.

## Project Structure
- `model.py`: The CNN network structure (3 blocks of Conv/BatchNorm/ReLU/MaxPool/Dropout, followed by fully connected layers).
- `train.py`: Training script. Includes data augmentation (random crop, flips, rotations) and automatically falls back to an Amazon S3 mirror if torchvision's download server is slow.
- `predict.py`: CLI script to run predictions on a single image.
- `app.py`: Simple Flask server to run the browser interface.
- `templates/index.html`: The HTML/CSS/JS frontend for uploading files and displaying prediction bars.
- `test_images/`: A few sample images (`cat.png`, `car.png`, `airplane.png`) to test the model.

## Setup and Installation

Install the dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If you have an Nvidia GPU, make sure to install PyTorch with CUDA support to speed up training).*

## Running the Project

### 1. Train the model
To download the dataset and train the model, run:
```bash
# Trains for 15 epochs (reaches ~80-85% accuracy)
python train.py --s3-mirror --epochs 15
```
It will save the best weights to `cifar10_cnn.pth` and generate a `training_curves.png` chart.

### 2. Run the Web App
To start the browser UI:
```bash
python app.py
```
Then open `http://127.0.0.1:5000/` in your browser. You can upload any image and see what the CNN predicts.

### 3. Predict via Command Line
To run a quick prediction on a single file:
```bash
python predict.py test_images/cat.png --show
```
