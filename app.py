import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.cnn import EmotionCNN

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
COLOR_MAP = {
    'angry': 'red',
    'disgust': 'green',
    'fear': 'purple',
    'happy': 'orange',
    'neutral': 'gray',
    'sad': 'blue',
    'surprise': 'gold'
}

@st.cache_resource
def load_model():
    model = EmotionCNN(num_classes=len(EMOTION_CLASSES))
    model.load_state_dict(torch.load("models/emotion_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_emotion(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return EMOTION_CLASSES[predicted.item()]

st.title("Emotion Detection from Facial Expression")
st.write("Upload an image to detect facial emotion. Results will appear side-by-side.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    final_image = image_cv.copy()
    detected_emotion = None

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            face_img = image_cv[y:y+h, x:x+w]
            face_tensor = preprocess_image(face_img)
            emotion = predict_emotion(face_tensor)
            detected_emotion = emotion

            cv2.rectangle(final_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    col1, spacer, col2 = st.columns([1, 0.5, 1])
    with col1:
        st.image(image_np, caption="Uploaded Image", width=400)
    with col2:
        st.image(final_image_rgb, caption="Detected Face", width=400)
        if detected_emotion:
            color = COLOR_MAP[detected_emotion]
            st.markdown(
                f"""
                <div style="text-align:center; margin-top: 10px;">
                    <span style="font-size: 20px; font-weight: bold; color: {color};">
                        Detected Emotion: {detected_emotion.upper()}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
