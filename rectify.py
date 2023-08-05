import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


# Define Rectify Dataset class
class RectifyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = ((image - np.mean(image)) / np.std(image)).astype(np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask < 128] = 0.0
        mask[mask >= 128] = 255.0
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class TransformDataset(Dataset):
    def __init__(self, root_dir, weak_transform=None, strong_transform=None):
        self.root_dir = root_dir
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.list_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        origin_image = image

        if self.strong_transform:
            strong_image = self.strong_transform(origin_image)
            weak_image = self.weak_transform(image)
            strong_image = self.weak_transform(strong_image)
            weak_image = ((weak_image - torch.mean(weak_image)) / torch.std(weak_image))
            strong_image = ((strong_image - torch.mean(strong_image)) / torch.std(strong_image))

        return weak_image, strong_image
