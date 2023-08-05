import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)  # Apply sigmoid to the output

        # Compute intersection and union for dice coefficient
        intersection = (output * target).sum(dim=(2, 3))
        union = output.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2 * intersection + self.eps) / (union + self.eps)  # Compute dice coefficient

        return 1 - dice.mean()  # Return dice loss


# Define Cloud Dataset class
class CloudDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # List all images
        self.masks = os.listdir(mask_dir)  # List all masks

    def __len__(self):
        return len(self.images)  # Return total number of images

    def __getitem__(self, index):
        # Define the path of image and mask
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))  # Open and convert the image to RGB
        origin_image = image
        # Normalize the image
        image = ((image - np.mean(image)) / np.std(image)).astype(np.float32)
        # Open and convert the mask to gray scale, and normalize it
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            image = self.transform(image)  # Apply transformations to image
            mask = self.transform(mask)  # Apply transformations to mask

        return image, mask, origin_image, self.images[index]  # Return image, mask, origin image and filenames


# Function to convert a tensor into one-hot format
def to_one_hot(tensor, num_classes):
    one_hot = torch.zeros(tensor.size(0), num_classes, tensor.size(2), tensor.size(3)).to(tensor.device)
    one_hot.scatter_(1, tensor.long(), 1)
    return one_hot


# Define IoU function
def iou_score(output, target):
    smooth = 1e-6
    iou = 0.0
    output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    for pred, true in zip(output, target):
        pred_ = pred > 0.5
        true_ = true > 0.5

        intersection = (pred_ & true_).sum()
        union = (pred_ | true_).sum()

        iou += (intersection + smooth) / (union + smooth)

    return iou, iou / output.shape[0]


# Define F-score function
def f_score(y_pred, y_true, threshold=0.5, beta=1):
    # Apply threshold to prediction
    y_pred = y_pred > threshold

    # Convert tensors to boolean type
    y_pred = y_pred.bool()
    y_true = y_true.bool()

    # Calculate True Positive (TP), False Positive (FP), and False Negative (FN)
    TP = (y_pred * y_true).sum().to(torch.float32)
    FP = (y_pred * ~y_true).sum().to(torch.float32)
    FN = (~y_pred * y_true).sum().to(torch.float32)

    # Calculate precision and recall
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # Calculate F-score
    f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-8)

    return f_score.item()


# Function to return a Convolution followed by ReLU
def conv_relu(in_channels, out_channels, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.ReLU(inplace=True),
    )


# Define ResUNet class
class ResUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Use pretrained ResNet50 model
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # Extract layers of ResNet50
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # First three layers
        self.layer0_1x1 = conv_relu(64, 64)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # Next two layers
        self.layer1_1x1 = conv_relu(256, 256)
        self.layer2 = self.base_layers[5]  # Layer 5
        self.layer2_1x1 = conv_relu(512, 512)
        self.layer3 = self.base_layers[6]  # Layer 6
        self.layer3_1x1 = conv_relu(1024, 1024)
        self.layer4 = self.base_layers[7]  # Layer 7
        self.layer4_1x1 = conv_relu(2048, 2048)

        # Define upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Define the convolutions for the upsampling part
        self.conv_up3 = conv_relu(1024 + 2048, 2048)
        self.conv_up2 = conv_relu(512 + 2048, 1024)
        self.conv_up1 = conv_relu(1024 + 256, 512)
        self.conv_up0 = conv_relu(512 + 64, 256)

        # Define the convolutions for the original size part
        self.conv_original_size0 = conv_relu(3, 16)
        self.conv_original_size1 = conv_relu(16, 32)
        self.conv_original_size3 = conv_relu(32, 64)
        self.conv_original_size2 = conv_relu(64 + 256, 64)

        # Define the final convolution to get the desired number of classes
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # Define the forward pass
        x_size = input.size()[2:]

        # Padding input to make it divisible by 32
        _, _, h, w = input.shape
        if h % 32 != 0 or w % 32 != 0:
            padding_h = (h // 32 + 1) * 32 - h if h % 32 != 0 else 0
            padding_w = (w // 32 + 1) * 32 - w if w % 32 != 0 else 0
            input = F.pad(input, (0, padding_w, 0, padding_h))

        # Apply the original size convolution to the input
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        x_original = self.conv_original_size3(x_original)

        # Apply the down path
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Apply the up path
        layer4_ = self.layer4_1x1(layer4)
        x = self.upsample(layer4_)
        layer3_ = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3_], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2_ = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2_], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1_ = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1_], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        # Get the final output
        out = torch.softmax(self.conv_last(x), 1)

        out = out[:, :, :x_size[0], :x_size[1]]  # Remove any padding that was added

        return out  # Return the final output
