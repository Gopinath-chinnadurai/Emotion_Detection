import torch
import cv2
import numpy as np
from torchvision import transforms
from models.cnn import EmotionCNN

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

def load_model(model_path='models/emotion_cnn.pth'):
    model = EmotionCNN(num_classes=len(EMOTION_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_emotion(image, model):
    face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face_tensor = transform(face).unsqueeze(0) 
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
    return EMOTION_CLASSES[predicted.item()], confidence
