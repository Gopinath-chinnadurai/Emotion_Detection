# Emotion Detection using EfficientNet

This project implements a **Deep Learning-based Emotion Detection System** that classifies human facial expressions into **seven different emotion categories** using **Transfer Learning with EfficientNet**.

The model is trained on a **facial emotion dataset sourced from Kaggle** and enhanced using **data augmentation and class balancing techniques** to improve performance and generalization. The final model achieves **over 90% accuracy** in detecting emotions.

The implementation is provided in a **Jupyter Notebook (.ipynb)** and demonstrates the full pipeline from **data preprocessing to model training and evaluation**.

---

## Project Overview

Human emotion recognition plays a crucial role in many **AI and Computer Vision applications**, such as:

- Human–Computer Interaction  
- Mental Health Monitoring  
- Smart Surveillance Systems  
- Customer Behavior Analysis  
- Emotion-aware AI Assistants  

This project builds a **robust deep learning model capable of identifying facial emotions from images** using a modern CNN architecture.

---

## Emotion Classes

The model classifies facial expressions into the following **7 emotion categories**:

1. Angry  
2. Disgust  
3. Fear  
4. Happy  
5. Sad  
6. Surprise  
7. Neutral  

---

## Dataset

The dataset used in this project is obtained from **Kaggle** and contains thousands of labeled facial emotion images.

### Dataset Characteristics

- Labeled facial emotion images  
- 7 emotion classes  
- Various lighting conditions and facial variations  
- Suitable for deep learning-based classification  

Due to size limitations, the dataset is **not included in this repository**.

### Dataset Structure

After downloading from Kaggle, organize the dataset as follows:


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


---

## Data Preprocessing

Several preprocessing techniques were applied before training the model.

### Data Augmentation

To improve generalization and prevent overfitting, the following transformations were used:

- Random Horizontal Flip
- Rotation
- Zoom / Scaling
- Brightness Adjustment
- Normalization

These techniques increase the diversity of training data and help the model learn more robust features.

---

### Class Balancing

Emotion datasets often suffer from **class imbalance**.

To address this issue:

- Data augmentation was applied to **underrepresented classes**
- Balanced sampling was used during training

This ensures the model learns **all emotion categories effectively** rather than being biased toward dominant classes.

---

## Model Architecture

The model uses **EfficientNet**, a state-of-the-art convolutional neural network architecture known for:

- High accuracy
- Efficient computation
- Better parameter utilization

### Transfer Learning

Instead of training from scratch:

- A **pretrained EfficientNet model** was used.
- The **final classification layer was replaced** to match the **7 emotion classes**.
- The model was then **fine-tuned on the emotion dataset**.

Additional improvements:

- Dropout added for regularization
- Fine-tuning of the classifier layer

---

## Training Details

Training configuration:

- **Model:** EfficientNet  
- **Transfer Learning:** Enabled  
- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** Adam  
- **Training Method:** Fine-tuning  
- **Accuracy Achieved:** **90%+**

The model was trained using **GPU acceleration** for faster convergence.

---

## Results

The trained model achieved **over 90% classification accuracy** on the validation dataset.

### Performance Highlights

- Accurate detection of multiple emotions
- Robust against facial expression variations
- Improved performance through augmentation and class balancing

---

## Project Structure

Emotion_Detection/
│
├── scripts/
│ └── training_emotions.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore


---

## Technologies Used

- Python
- Jupyter Notebook
- PyTorch / TensorFlow
- EfficientNet
- OpenCV
- NumPy
- Matplotlib

---

## How to Run the Project

### 1 Clone the Repository


git clone https://github.com/Gopinath-chinnadurai/Emotion_Detection.git


### 2 Install Dependencies


pip install -r requirements.txt


### 3 Open the Notebook


jupyter notebook scripts/training_emotions.ipynb


### 4 Run the Cells

Execute the notebook step-by-step to:

- Load dataset
- Apply preprocessing
- Train the model
- Evaluate the model

---

## Future Improvements

Possible improvements include:

- Real-time emotion detection using webcam
- Deployment using Streamlit or Flask
- Emotion detection from video streams
- Training with larger datasets for better generalization

---

## Applications

Emotion detection systems can be applied in:

- Smart classrooms
- Mental health monitoring tools
- Customer feedback analysis
- AI assistants
- Security and surveillance systems

---

## Author

**Gopinath Chinnadurai**

AI / ML Enthusiast | Python Developer  

GitHub: https://github.com/Gopinath-chinnadurai
