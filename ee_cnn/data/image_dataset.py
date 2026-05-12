import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, data, transform=None, transform_pos=None, transform_neg=None):
        """
        Args:
            data (list of tuples): List of (label, path_to_image)
            transform (callable, optional): Transform to be applied on an image
        """
        self.data = data
        self.transform = transform
        self.transform_pos = transform_pos
        self.transform_neg = transform_neg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        elif self.transform_pos is not None and label == 1:
            image = self.transform_pos(image)
        elif self.transform_neg is not None and label == 0:
            image = self.transform_neg(image)

        return image, torch.tensor(label, dtype=torch.long)
