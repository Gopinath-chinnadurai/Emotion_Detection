import os
from PIL import Image
from torch.utils.data import Dataset

class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        for idx, emotion in enumerate(self.class_names):
            emotion_dir = os.path.join(root_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                self.image_paths.append(os.path.join(emotion_dir, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
