# 🛰️ Satellite Image Classification using Deep Learning

A Deep Learning-based land cover classification system that classifies satellite images into different land-use categories using transfer learning. This project evaluates and compares the performance of **ResNet50**, **DenseNet121**, and **EfficientNetB0** architectures on the **EuroSAT dataset**.

---

## 📌 Project Overview

Satellite image classification plays an important role in remote sensing applications such as urban planning, agriculture monitoring, forest management, and environmental analysis.

This project develops an end-to-end image classification pipeline using transfer learning techniques to accurately classify satellite imagery into multiple land-cover categories.

---

## 🎯 Objectives

- Build an accurate satellite image classification model.
- Compare the performance of multiple transfer learning architectures.
- Apply data preprocessing and augmentation techniques.
- Evaluate models using standard classification metrics.

---

## 📂 Dataset

**Dataset Used:** EuroSAT Dataset (Subset)

### Classes
- 🌲 Forest
- 🏭 Industrial
- 🏠 Residential
- 🌊 River

**Total Images Used:** ~1000

### Data Split
- Training Set: **70%**
- Validation Set: **20%**
- Test Set: **10%**

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Google Colab

---

## ⚙️ Project Workflow

### 1️⃣ Data Understanding
- Analyzed dataset distribution
- Verified image classes
- Explored dataset structure

### 2️⃣ Data Preprocessing
- Image resizing (64 × 64)
- Pixel normalization
- Data augmentation
  - Horizontal Flip
  - Rotation
  - Zoom

### 3️⃣ Model Development
Implemented and compared three transfer learning models:

- ResNet50
- DenseNet121
- EfficientNetB0

### 4️⃣ Model Training
- Trained on the training dataset
- Validated using the validation dataset
- Optimized using transfer learning

### 5️⃣ Model Evaluation
Evaluated models using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cohen's Kappa Score

### 6️⃣ Prediction
Predicted land-cover classes for unseen satellite images.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|---------:|----------:|-------:|---------:|
| ResNet50 | **95.68%** | **0.956** | **0.956** | **0.956** |
| EfficientNetB0 | 92.61% | 0.926 | 0.917 | 0.921 |
| DenseNet121 | 92.26% | 0.922 | 0.913 | 0.917 |

> Replace the remaining values with your actual results.


## 🚀 Future Improvements

- Increase dataset size
- Train on all EuroSAT classes
- Deploy using Streamlit
- Integrate Grad-CAM for model explainability
- Optimize model for real-time inference

---

## 👩‍💻 Author

**Vidhi Sharma**

B.Tech – Artificial Intelligence & Machine Learning

GitHub: https://github.com/Vidhisharma-17

LinkedIn: https://linkedin.com/in/vidhi-sharma-23559030a
