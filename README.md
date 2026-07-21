# AI & Machine Learning Projects Portfolio

This repository contains a diverse collection of computer vision, deep learning, reinforcement learning, tabular classification, and retrieval-augmented generation (RAG) projects.

## Projects Included

1. **[Skin Cancer Diagnostic Classifier](./cancer_detection)**
   - A FastAPI web application using customized ResNet-18 Transfer Learning to run diagnostic classification on clinical dermoscopy scans across 7 disease categories.
   
2. **[CIFAR-10 Image Classifier](./cifar10_cnn_classifier)**
   - A Convolutional Neural Network (CNN) trained to classify general object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) on the CIFAR-10 dataset.
   
3. **[Face Recognition System](./face_recognition)**
   - A deep learning face detection and landmark-matching application leveraging facial feature representation.

4. **[Adult Census Income Classifier](./adult-census-income-classification)**
   - A machine learning pipeline and interactive FastAPI web dashboard utilizing a hyperparameter-tuned XGBoost model to predict individual income levels (above/below $50K/year) from demographic census profiles.

5. **[Cart-Pole DQN Reinforcement Learning](./cartpole-rl-training)**
   - An interactive DQN agent training dashboard visualizing the classic Cart-Pole control problem at 60 FPS in Farama Gymnasium with PyTorch.

6. **[CineMinds AI Movie Recommendation Platform](./cine-match)**
   - A Netflix-style movie recommendation engine combining content filtering, interactive cosine similarity vector space visualizers, and a conversational RAG co-pilot.

7. **[Apollo L.E.M. Planetary Flight & RAG Guidance](./lunar-landing-rag)**
   - An Apollo lunar module planetary flight simulator (Moon, Mars, Earth, Asteroid Bennu) coupled side-by-side with a vector-space visualizer and RAG-based flight guide.

8. **[University Admissions RAG Chatbot Sandbox](./university-rag-chatbot)**
   - An interactive single-page app displaying a client-side RAG Admissions chatbot for VIT Bhopal University with voice synthesis and speech-to-text dictation.

9. **[Used Car Value Predictor](./car-value-predictor)**
   - An interactive web dashboard utilizing a Random Forest Regressor trained on the CarDekho V3 dataset to estimate used car market resale values directly from 10 technical specifications (mileage, engine cc, power, seats, age, fuel, transmission).

---

## Getting Started

To explore or run any of the individual projects, navigate into the project directory and follow the instructions in the project-specific `README.md` file.

### Portfolio Structure
```text
Vision-AI-Portfolio/
├── .gitignore
├── README.md
├── adult-census-income-classification/
├── cancer_detection/
├── cartpole-rl-training/
├── cifar10_cnn_classifier/
├── car-value-predictor/
├── cine-match/
├── face_recognition/
├── lunar-landing-rag/
└── university-rag-chatbot/
```
