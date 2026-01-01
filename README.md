# 🧠 Brain Tumor Detection using Deep Learning

## 📌 Project Overview
This project implements an **end-to-end deep learning pipeline** to classify **brain MRI images** into multiple tumor categories using **Convolutional Neural Networks (CNNs)** and **transfer learning**.

The system is designed to assist in the **early detection and classification of brain tumors**, which is a critical task in medical imaging. The project focuses not only on accuracy but also on **proper evaluation, reproducibility, and professional project structure**.

---

## 🎯 Problem Statement
Manual analysis of brain MRI scans is:
- Time-consuming
- Subjective
- Dependent on expert availability  

The goal of this project is to build an automated system that can classify MRI images into the following categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **Normal (No Tumor)**

---

## 📂 Dataset
- **Source**: Public MRI brain tumor dataset (Kaggle)
- **Type**: Brain MRI images
- **Classes**: 4
- **Format**: JPG / PNG images

### Original Dataset Structure

Data/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── normal/


Each folder represents a class label.

---

## 🗂️ Project Structure
The project follows an **industry-standard machine learning structure**:

Brain-Tumor-Detection/
│
├── Data/ # Original dataset
│
├── data/
│ └── final/
│ ├── train/ # Training data
│ └── val/ # Validation data
│
├── src/
│ ├── split_data.py # Train/validation split
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│ └── predict.py # Prediction & CSV generation
│
├── models/
│ └── brain_tumor_model.h5 # Trained model
│
├── results/
│ ├── confusion_matrix.png
│ ├── classification_report.txt
│ └── val_predictions.csv
│
├── notebooks/
│ └── Brain_Tumor_Detection.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Model Architecture**: MobileNetV2 (Transfer Learning)
- **Libraries**:
  - NumPy
  - Pandas
  - OpenCV
  - Scikit-learn
  - split-folders
  - Matplotlib

---

## 🧪 Methodology

### 1️⃣ Data Preparation
- Dataset organized into class-wise folders
- Data split into:
  - **80% Training**
  - **20% Validation**
- Splitting performed using `split-folders`

### 2️⃣ Preprocessing
- Images resized to **224 × 224**
- Pixel values normalized to `[0, 1]`
- Ensures uniform input to the CNN

### 3️⃣ Model Architecture
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dense (ReLU)
  - Dropout (regularization)
  - Dense (Softmax for 4 classes)

Transfer learning allows the model to leverage pre-learned visual features while adapting to medical images.

---

## 🏋️ Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 10 (baseline)
- Training performed on CPU (local) / GPU (recommended)

The trained model is saved for reuse:


---

## 📊 Model Evaluation
Evaluation is performed using:
- **Confusion Matrix**
- **Classification Report**
  - Precision
  - Recall
  - F1-score

These metrics provide a better understanding than accuracy alone, especially for medical datasets.

### Observations
- Strong performance for **glioma** and **pituitary tumors**
- Some confusion between **glioma and meningioma**, which is a known challenge due to visual similarity in MRI scans

---

## 🔍 Model Testing
The trained model is tested on:
- Individual MRI images
- Entire validation dataset

Predictions for all validation images are saved in:
    results/val_predictions.csv