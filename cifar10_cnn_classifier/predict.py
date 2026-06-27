import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import SimpleCNN

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def preprocess_image(image_path):
    """Loads and preprocesses an image for inference."""
    try:
        # Open image using Pillow
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None
        
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    # Apply transforms and add batch dimension (batch_size = 1)
    tensor = transform(image).unsqueeze(0)
    return image, tensor

def predict(model, tensor, device):
    """Runs the image tensor through the CNN and returns the prediction."""
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        outputs = model(tensor)
        
        # Calculate class probabilities using softmax
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get the class with the highest probability
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        
    return predicted_idx.item(), confidence.item(), probabilities.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Classify an image using the trained CIFAR-10 CNN")
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    parser.add_argument("--model-path", type=str, default="cifar10_cnn.pth", help="Path to trained model weights")
    parser.add_argument("--show", action="store_true", help="Display the image with prediction using Matplotlib")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at '{args.model_path}'. Please train the model first!")
        return

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at '{args.image_path}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        print(f"Loaded trained model weights from '{args.model_path}'.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    pil_img, tensor_img = preprocess_image(args.image_path)
    if tensor_img is None:
        return

    pred_class_idx, confidence, probs = predict(model, tensor_img, device)
    pred_label = CLASSES[pred_class_idx]
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {pred_label.upper()}")
    print(f"Confidence Score: {confidence * 100:.2f}%")
    print("\nProbability breakdown:")
    for idx, prob in enumerate(probs):
        print(f"  {CLASSES[idx]:<12}: {prob * 100:6.2f}%")
        
    # 6. Save or show results with visual overlay
    if args.show:
        plt.figure(figsize=(5, 5))
        plt.imshow(pil_img)
        plt.title(f"Prediction: {pred_label} ({confidence * 100:.1f}%)")
        plt.axis('off')
        
        # Save a copy next to the image for viewing
        base, ext = os.path.splitext(args.image_path)
        output_plot_path = f"{base}_prediction{ext}"
        plt.savefig(output_plot_path)
        print(f"\nOverlay prediction plot saved to '{output_plot_path}'.")
        plt.show()

if __name__ == "__main__":
    main()
