# 🩺 Pneumonia Detection AI

An AI-powered chest X-ray classification application built using Machine Learning, Deep Learning, and Streamlit.

## 📌 Project Overview

Pneumonia Detection AI is an academic project designed to classify chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

The application provides an interactive dashboard where users can upload chest X-ray images, select an AI model, view predictions, and compare model performance.

## 🚀 Key Features

- Interactive dark-themed Streamlit dashboard
- Chest X-ray image upload and preview
- AI-based pneumonia prediction
- Six implemented Machine Learning and Deep Learning models
- Model performance comparison
- Prediction confidence where supported
- Interactive performance charts

## 🤖 Implemented Models

### Deep Learning
1. CNN
2. ResNet50
3. VGG16
4. EfficientNetB0

### Machine Learning
5. Support Vector Machine (SVM)
6. Random Forest

## Trained Model Download Links

Download the large trained models from Google Drive:

- **VGG16 Model:** https://drive.google.com/file/d/1yDJZ_qGSRg8RBYBS0d9BKpPcaUbop7Co/view?usp=drive_link
- **ResNet50 Model:** https://drive.google.com/file/d/10ye79JnS4ZbAQxcTFKZ95VKAv4NuGf64/view?usp=drive_link

After downloading, place the models in their respective folders:

- VGG16: `vgg16_lightweight_results/best_vgg16.keras`
- ResNet50: `resnet50_improved_results/best_resnet50_finetuned.keras`
  
## 📊 Model Performance

| Model | Accuracy |
|---|---:|
| CNN | 77.35% |
| ResNet50 | 62.62% |
| VGG16 | 78.48% |
| EfficientNetB0 | 86.89% |
| SVM | 76.70% |
| Random Forest | 81.55% |

*These are previously recorded test-set results.*

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- NumPy
- Pandas
- Pillow
- Joblib

## 📂 Project Structure

```text
Pneumonia_Detection_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── cnn_results/
├── resnet50_improved_results/
├── vgg16_lightweight_results/
├── efficientnetb0_corrected_results/
├── svm_results/
├── random_forest_results/
│
└── processed_data/
```

*Model folders and trained weights must be provided separately if they are not included in the repository.*

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project Directory

```bash
cd Pneumonia_Detection_Project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

## 🎯 Project Objective

The objective is to explore and compare different machine learning and deep learning approaches for chest X-ray image classification through an interactive application.

## ⚠️ Disclaimer

This project is developed for academic and research purposes only. Its predictions are experimental and must not be used as a substitute for professional medical diagnosis.

## 👨‍💻 Author

**Muhammad Tufail Ahmed**

Machine Learning | Deep Learning | Data Science | Python

---

⭐ If you find this project interesting, consider giving the repository a star.

## Dataset

Source: Kaggle - Paul Mooney

Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Classes:
- NORMAL
- PNEUMONIA

The dataset is not included in this repository.
Download it separately from Kaggle.
