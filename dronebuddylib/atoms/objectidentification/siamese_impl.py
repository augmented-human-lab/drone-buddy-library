import os
from pathlib import Path

import cv2
from torch import nn
from torchvision.io import read_image

from dronebuddylib.atoms.objectidentification.resources.matching.SiameseNetworkAPI import SiameseNetworkAPI
from dronebuddylib.atoms.objectidentification.resources.matching.model import SiameseModel
from dronebuddylib.atoms.objectidentification.resources.matching.tune_api import tune, loadModel

import torch
import torchvision.transforms as T


def update_memory():
    """update the memory by retraining the model (it may take several minutes)
    """
    tune()
    return


def evaluate():
    loadModel()
    return


def load_model():
    """load the model for object identification
    """
    # Load the model
    model = SiameseModel()
    model.load_state_dict(torch.load('path_to_your_model.pth'))
    model.eval()  # Set the model to evaluation mode


transform = nn.Sequential(
    T.Resize((228, 228))
)

"""
input: folder_path - folder of 1 or more images 
output: torch tensor of the images 
"""


def load_images_from_folder(folder_path, transform=transform):
    images_list = []
    for file_path in os.listdir(folder_path):
        for file_name in os.listdir(folder_path + "\\" + file_path):
            if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
                tensor = transform(read_image(str(Path(folder_path) / file_path / file_name)))
                images_list.append(tensor[:3, :, :].permute(0, 2, 1))
                tensor = transform(read_image(str(Path(folder_path) / file_path / file_name)))
                images_list.append(tensor[:3, :, :].permute(0, 2, 1))
    return torch.stack(images_list)


def inference():
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\blaaa.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_lab_tabel.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_table_with_mal_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\long_distance.jpeg"
    #
    image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\p_bottle.jpeg"

    obj_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    all_img_of_obj = load_images_from_folder(obj_folder)
    image = cv2.imread(image_path)
    api = SiameseNetworkAPI(all_img_of_obj)
    a, b = api.inference(image)
    print(a)
    print(b)


def detection():
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\blaaa.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_lab_tabel.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_table_with_mal_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\hot_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\long_distance.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\dini_bottle.jpeg"
    image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\p_bottle.jpeg"

    #
    obj_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    all_img_of_obj = load_images_from_folder(obj_folder)
    image = cv2.imread(image_path)
    api = SiameseNetworkAPI(all_img_of_obj)
    api.get_detected_objects(image)


#  add main method to run the update_memory function
if __name__ == '__main__':
    update_memory()
    # inference()
    # evaluate()
    # detection()
