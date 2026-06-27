import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

from model import SimpleCNN

app = Flask(__name__)

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
MODEL_PATH = "cifar10_cnn.pth"

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded trained model weights from '{MODEL_PATH}'.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Warning: Model checkpoint '{MODEL_PATH}' not found. "
          f"Please run 'train.py' to train the model first! Predictions will use random weights.")
    model.to(device)
    model.eval()

# Preprocessing transforms
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)
preprocess_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Accepts an uploaded image file, runs inference, and returns JSON results."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename selected'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        tensor = preprocess_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
            probs_list = probabilities.cpu().tolist()

        return jsonify({
            'class_index': predicted_idx,
            'predicted_class': CLASSES[predicted_idx],
            'confidence': confidence,
            'probabilities': probs_list
        })

    except Exception as e:
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Automatically open the default web browser after a brief 1-second delay
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000/")

    # Open only in the primary process (avoids opening twice during reloader restarts)
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1.0, open_browser).start()

    print("Starting local Web Server on http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True)
