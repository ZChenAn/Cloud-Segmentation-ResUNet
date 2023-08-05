import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_files = os.listdir(self.root_dir)
        self.images = os.listdir(root_dir)  # List all images

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.list_files[idx])
        images = np.array(Image.open(img_path).convert("RGB"))
        origin_images = images
        images = ((images - np.mean(images)) / np.std(images)).astype(np.float32)

        if self.transform:
            images = self.transform(images)
        return images, origin_images, self.images[idx]
