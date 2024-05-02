import torch
import torch.nn as nn
from torchvision.io import read_image
import torchvision.transforms as T
from pathlib import Path
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

transform = T.Compose([
    T.Resize((228, 228)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=(0, 45), translate=(0., 0.2)),
    T.ConvertImageDtype(torch.float)
])


class SiameseDataset(Dataset):

    def __init__(self, images_folder_path, transform=None, num_samples=10, **kwargs):
        self.images_folder_path = images_folder_path
        self.transform = transform
        self.num_samples = num_samples

        self.class_names = [name for name in os.listdir(self.images_folder_path) if
                            os.path.isdir(os.path.join(self.images_folder_path, name))]
        self.all_images = [
            [self.class_names[i] + "/" + name for name in os.listdir(self.images_folder_path + self.class_names[i]) if
             "ds_store" not in name.lower()] for i in range(len(self.class_names))]

        self.data = self.create_dataset()

    def __getitem__(self, index):
        img1 = read_image(str(Path(self.images_folder_path) / self.data[0][index]))
        img2 = read_image(str(Path(self.images_folder_path) / self.data[1][index]))

        if self.transform is not None:
            img1_transformed = self.transform(img1)
            img2_transformed = self.transform(img2)

        return img1_transformed, img2_transformed, self.data[2][index], self.data[3][index], self.data[4][index]

    def __len__(self):
        return self.num_samples

    def get_sample(self):
        print(self.class_names)
        get_same_class = np.random.randint(0, 2, size=self.num_samples)
        image_one_class = np.random.randint(0, len(self.class_names), size=self.num_samples)
        image_two_class = [
            np.random.choice(
                [x for x in range(len(self.class_names)) if x != image_one_class[i]]
            )
            if get_same_class[i] == 0 else image_one_class[i] for i in range(self.num_samples)
        ]

        # to do: images 1 and 2 may be the same photo when the classes are the same
        image_one = list(map(lambda x: random.choice(self.all_images[x]), image_one_class))
        image_two = list(map(lambda x: random.choice(self.all_images[x]), image_two_class))

        return image_one, image_two, image_one_class, image_two_class, get_same_class

    def create_dataset(self):
        x1, x2, c1, c2, y = self.get_sample()
        return x1, x2, c1, c2, y
