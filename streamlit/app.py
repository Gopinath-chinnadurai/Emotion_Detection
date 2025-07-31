import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tempfile

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = r"C:\Emotion_Detection\model\emotion_model_best.pth"
IMAGE_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(EMOTION_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

def detect_face(img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return img 

    x, y, w, h = faces[0]  
    cropped_face = cv_img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)

def predict_emotion(image: Image.Image) -> str:
    face = detect_face(image)
    tensor = preprocess(face).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return EMOTION_CLASSES[predicted.item()]

st.title(" Emotion Detection from Facial Image")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)  


    if st.button("Predict Emotion"):
        with st.spinner("Analyzing..."):
            prediction = predict_emotion(image)
            st.success(f" Predicted Emotion: **{prediction.upper()}**")
