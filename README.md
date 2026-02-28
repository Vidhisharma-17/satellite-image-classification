# 🚀 Satellite Image Classification using Deep Learning

## 📌 Overview

This project focuses on classifying satellite images into different land-use categories using a Convolutional Neural Network (CNN). The model is trained on a subset of the EuroSAT dataset and is capable of identifying terrain types such as Forest, Residential, Industrial, and River.

The objective of this project is to demonstrate the application of Artificial Intelligence and Computer Vision in remote sensing and geospatial analysis.

## 📂 Dataset

* **Dataset Used:** EuroSAT Dataset (subset)
* **Classes:**

  * Forest
  * Industrial
  * Residential
  * River
* **Total Images Used:** ~1000
* **Data Split:**

  * Training Set: 70%
  * Validation Set: 20%
  * Test Set: 10%


## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV


## ⚙️ Project Pipeline

The project follows a complete Machine Learning lifecycle:

1. **Data Understanding**

   * Analyzed class distribution
   * Verified dataset structure

2. **Data Preprocessing**

   * Image resizing (64x64)
   * Normalization (pixel scaling)
   * Data augmentation (flip, rotation, zoom)

3. **Model Building**

   * Designed a CNN architecture
   * Used convolution, pooling, and dense layers

4. **Model Training**

   * Trained using training dataset
   * Validated performance using validation dataset

5. **Evaluation**

   * Tested model on unseen test data
   * Generated accuracy and loss graphs

6. **Prediction**

   * Predicted class for new satellite images

## 📊 Results

* **Training Accuracy:** ~84%
* **Validation Accuracy:** ~80%
* **Test Accuracy:** ~87%

The model demonstrates strong generalization capability on unseen data.

## 🔍 Sample Prediction

The trained model can classify new satellite images into land-use categories with good accuracy.

Example:
Predicted Class: Forest
Confidence: 0.91

## 🌍 Applications

* Land-use and land-cover classification
* Environmental monitoring
* Urban planning
* Disaster detection and management
* Remote sensing analysis



## 🔮 Future Work

* Implement transfer learning using models like ResNet or MobileNet
* Increase dataset size for better generalization
* Deploy model as a web application using Streamlit
* Improve accuracy with hyperparameter tuning



## 📁 Project Structure


Satellite-Image-Classification/
│
├── dataset/              # Dataset (not uploaded to GitHub)
├── satellite_pipeline.py # Main training script
├── advanced_model.py     # Transfer learning model (optional)
├── satellite_model.keras # Saved trained model
├── accuracy_graph.png    # Model performance graph
├── requirements.txt      # Dependencies
└── README.md             # Project documentation


## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/satellite-image-classification.git
```

2. Install dependencies:


pip install -r requirements.txt


3. Run the model:


python satellite_pipeline.py

 👩‍💻 Author

**Vidhi Sharma**
B.Tech Artificial Intelligence & Machine Learning


## ⭐ Acknowledgement

* EuroSAT Dataset
* TensorFlow Documentation


## 📌 Note

This project is developed for educational purposes and demonstrates the use of deep learning techniques in satellite image classification.
