import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

import torchvision.transforms as T

transform = nn.Sequential(
    T.Resize((228, 228)),
    T.ConvertImageDtype(torch.float32)  # Convert images to float tensors (optional normalization can be added here)
)


class InferenceDataset(Dataset):
    def __init__(self, all_img_of_obj, crop_img_of_obj, transform=None):
        self.transform = transform
        self.all_img_of_obj = all_img_of_obj
        self.crop_img_of_obj = crop_img_of_obj

    def __getitem__(self, index):
        img0 = self.all_img_of_obj[index]
        img1 = self.crop_img_of_obj

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return self.all_img_of_obj.shape[0]


"""
input: folder_path - folder of 1 or more images 
output: torch tensor of the images 
"""


def load_images_from_folder(folder_path, transform=None):
    images_list = []
    images = []
    reference_images = {}
    folder_path = Path(folder_path)
    for folder_name in os.listdir(folder_path):
        file_path = folder_path / folder_name
        # read the files in the folder
        images = []
        for file_name in os.listdir(file_path):
            image_file_name = file_path / file_name
            if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                image = Image.open(image_file_name).convert('RGB')
                if transform:
                    image = transform(image)
                images.append(image)
        if images:
            stacked_images = torch.stack(images)
            print(f"Loaded {len(images)} images for {folder_name}, Tensor Shape: {stacked_images.shape}")
            reference_images[folder_name] = stacked_images
    # return torch.stack(images_list)  # Stack images into a single tensor
    return reference_images
