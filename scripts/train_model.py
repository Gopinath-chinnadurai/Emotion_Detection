import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.fer_dataset import FERDataset
from models.cnn import EmotionCNN

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FERDataset(root_dir='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_dataset.class_names)
print("Detected emotion classes:", train_dataset.class_names)
model = EmotionCNN(num_classes=num_classes).to(device)

labels = train_dataset.labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} finished, Loss: {running_loss / len(train_loader):.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/emotion_cnn.pth")
print("Model saved to models/emotion_cnn.pth")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Final Training Accuracy: {(correct / total) * 100:.2f}%")
