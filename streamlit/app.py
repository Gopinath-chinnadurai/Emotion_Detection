import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# ── MUST be first Streamlit command ──────────────────────────────────────────
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")

# ── Config ───────────────────────────────────────────────────────────────────
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJI   = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']
IMAGE_SIZE      = 224
MODEL_PATH      = "emotion_model_best.pth"
GDRIVE_FILE_ID  = "1jzejwzxPLU1cSalunY0ylSP5HbDQMs2Y"

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model... please wait."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model could not be downloaded. Check your Google Drive sharing settings.")
        st.stop()

    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(EMOTION_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ── Face Detection ────────────────────────────────────────────────────────────
def detect_face(img: Image.Image):
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return img, False

    x, y, w, h = faces[0]
    cropped = cv_img[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), True

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_emotion(image: Image.Image):
    face, face_found = detect_face(image)
    tensor = preprocess(face).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        idx     = torch.argmax(probs).item()
    return EMOTION_CLASSES[idx], EMOTION_EMOJI[idx], probs, face_found

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("😊 Emotion Detection")
st.caption("Upload a face image — EfficientNet will detect the emotion using deep learning.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    if st.button("🔍 Predict Emotion", use_container_width=True):
        with st.spinner("Analyzing..."):
            emotion, emoji, probs, face_found = predict_emotion(image)

        if not face_found:
            st.warning("⚠️ No face detected — prediction made on full image.")

        with col2:
            st.subheader("Result")
            st.markdown(f"## {emoji} {emotion}")

        st.divider()
        st.subheader("Confidence Scores")
        for i, cls in enumerate(EMOTION_CLASSES):
            st.progress(float(probs[i]), text=f"{EMOTION_EMOJI[i]} {cls}: {probs[i]*100:.1f}%")