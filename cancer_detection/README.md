# OncoVision: Skin Lesion Diagnostic Classifier

OncoVision is a web-based clinical classification system for skin cancer detection. The application utilizes a customized ResNet-18 model optimized for low-resolution dermoscopy scans to run real-time inference and output probability scores across 7 distinct skin lesion classes.

## Features
- **Real-Time Classification**: Instant inference on uploaded clinical images.
- **Detailed Evaluation Metrics**: Displays test accuracy, classification reports, confusion matrices, and ROC curves on startup.
- **Premium User Interface**: Modern glassmorphism UI designed for easy diagnostics.
- **Modular REST API**: Powered by FastAPI for model predictions and metrics delivery.

## Model Architecture & Transfer Learning
The classification backend implements a custom **ResNet-18** network adapted for 28x28 inputs:
- Replaces the standard ResNet-18 7x7 stride 2 convolution and maxpool with a 3x3 stride 1 convolution, preventing aggressive downsampling and preserving spatial resolution for low-res dermoscopic images.
- Implements transfer learning by loading ImageNet-pretrained weights for core feature-extraction layers.
- Fine-tuned on the **DermaMNIST** dataset.

## Supported Classes
The model evaluates input scans against 7 skin lesion categories:
1. Actinic Keratoses and Intraepithelial Carcinoma (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis-like Lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic Nevi (nv)
7. Vascular Lesions (vasc)

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision
- FastAPI & Uvicorn
- NumPy, Scikit-learn, Matplotlib, Seaborn, MedMNIST

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/oncovision.git
   cd oncovision
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server
Start the local FastAPI development server:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

## API Endpoints
Interactive OpenAPI documentation is available at `/docs` (Swagger UI):
- `POST /api/predict`: Accepts a skin lesion image file and returns sorted probability percentages.
- `GET /api/model/results`: Returns cached classification reports, Confusion Matrix plots, and ROC curve plots.
