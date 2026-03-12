# Emotion Detection Using EfficientNet

### Project Overview

This project implements a **Facial Emotion Detection System** using **Deep Learning and Transfer Learning**.

The model classifies human facial expressions into **7 different emotion categories** using a **fine-tuned EfficientNet architecture**.

The dataset used for training was sourced from **Kaggle**, and several preprocessing techniques such as **data augmentation and class balancing** were applied to improve the model’s performance and generalization ability.

The complete implementation is provided in a **Jupyter Notebook (.ipynb)** and demonstrates the full pipeline from **data preprocessing, model training, fine-tuning, and evaluation**.

---

## Emotion Classes

The model predicts the following **7 facial emotions**:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Dataset

The dataset used for this project is sourced from **Kaggle** and contains thousands of labeled facial expression images.

**Dataset Characteristics**

- Labeled facial emotion images
- 7 emotion classes
- Various lighting conditions and facial expressions
- Suitable for deep learning-based emotion classification

Due to GitHub file size limitations, the dataset is **not included in this repository**.

**Dataset Structure**

```
dataset/
   train/
      angry/
      disgust/
      fear/
      happy/
      sad/
      surprise/
      neutral/

   test/
```

---

## Data Preprocessing

Before training the model, several preprocessing steps were applied to improve performance and model robustness.

### Data Augmentation

To increase dataset diversity and prevent overfitting, the following augmentation techniques were used:

- Random Horizontal Flip
- Image Rotation
- Zoom / Scaling
- Brightness Adjustment
- Image Normalization

These transformations allow the model to learn more **generalized facial features**.

### Class Balancing

Emotion datasets often contain **imbalanced class distributions**.

To solve this:

- Data augmentation was applied to **underrepresented classes**
- Balanced sampling strategies were used during training

This ensures the model does not become biased toward dominant classes.

---

## Model Architecture

The model uses **EfficientNet**, a modern convolutional neural network architecture known for:

- High accuracy
- Efficient computation
- Strong feature extraction capability

### Transfer Learning Approach

Instead of training from scratch:

- A **pretrained EfficientNet model** was used
- The **final classification layer was replaced**
- Output layer modified to predict **7 emotion classes**
- The model was **fine-tuned on the emotion dataset**

Additional improvements include:

- Dropout for regularization
- Fine-tuning of classifier layers

---

## Training Details

Training configuration used in this project:

- Model: EfficientNet
- Training Method: Transfer Learning + Fine-tuning
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Framework: PyTorch / TensorFlow
- Implementation: Jupyter Notebook

The model achieved **90%+ classification accuracy** on the validation dataset.

---

## End-to-End Workflow

```
Facial Emotion Dataset
        ↓
Data Preprocessing
        ↓
Data Augmentation
        ↓
Class Balancing
        ↓
EfficientNet (Pretrained Model)
        ↓
Replace Final Classification Layer
        ↓
Fine-Tuning on Emotion Dataset
        ↓
Model Evaluation
        ↓
Emotion Prediction
```

---

## Objectives

- Build an accurate facial emotion detection system
- Apply **transfer learning using EfficientNet**
- Improve model generalization using **data augmentation**
- Handle **class imbalance** in emotion datasets
- Demonstrate a **complete deep learning training pipeline**
- Create a **real-world computer vision project**

---

## Installation & Setup

### 1. Prerequisites

- Python 3.9+
- Jupyter Notebook
- PyTorch / TensorFlow
- OpenCV
- NumPy
- Matplotlib

---

### 2. Clone the Repository

```
git clone https://github.com/Gopinath-chinnadurai/Emotion_Detection.git

cd Emotion_Detection
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Run the Notebook

```
jupyter notebook scripts/training_emotions.ipynb
```

Run the notebook cells sequentially to:

- Load the dataset
- Apply preprocessing
- Train the model
- Evaluate performance

---

## Model Performance

The trained model achieved:

- **90%+ Accuracy**
- Strong performance across multiple emotion categories
- Robust predictions due to augmentation and balanced training

---

## Applications

Emotion detection systems can be used in:

- Human–Computer Interaction
- Mental Health Monitoring
- Smart Surveillance Systems
- Customer Sentiment Analysis
- AI Assistants
- Smart Classrooms

---

## Future Improvements

Possible future improvements include:

- Real-time emotion detection using webcam
- Deployment using **Streamlit or Flask**
- Emotion detection from video streams
- Training with larger and more diverse datasets
- Deploying the model as an API

---

## Project Structure

```
Emotion_Detection/
│
├── scripts/
│   └── training_emotions.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Conclusion

This project demonstrates:

- Practical application of **Deep Learning in Computer Vision**
- Implementation of **Transfer Learning with EfficientNet**
- Use of **data augmentation and class balancing**
- Building a **complete model training pipeline**
- Real-world emotion detection system development

---

## Author

**Gopinath Chinnadurai**

GitHub:

https://github.com/Gopinath-chinnadurai
