import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import torch
from torchvision.transforms import functional as F
from PIL import Image
from dronebuddylib.atoms.objectidentification.resources.matching.SiameseNetworkAPI import SiameseNetworkAPI
import logging


def compare_image_with_folder_tsne(target_image_path, folder_path, api):
    target_image = cv2.imread(target_image_path)
    if target_image is None:
        logging.error(f"Failed to load the target image: {target_image_path}")
        return pd.DataFrame()

    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logging.error(f"The specified folder path is not a directory: {folder_path}")
        return pd.DataFrame()

    data = []
    labels = []
    for image_path in folder_path.glob('*.jpg'):
        try:
            comparison_image = cv2.imread(str(image_path))
            if comparison_image is None:
                logging.error(f"Failed to load comparison image: {image_path}")
                continue
            output = api.two_image_inference_difference(target_image, comparison_image).detach().numpy()
            is_same_class = folder_path.name == target_image_path.parent.name
            data.append(output.flatten())  # Flattening the output
            label_name = folder_path.name + ("_same" if is_same_class else "_diff")
            labels.append(label_name)

        except Exception as e:
            logging.error(f"Error comparing {target_image_path} and {image_path}: {str(e)}")

    return np.array(data), labels


def compare_folder_with_others_tsne(root_folder, target_folder, api):
    root_folder = Path(root_folder)
    target_folder_path = root_folder / target_folder

    if not target_folder_path.is_dir():
        logging.error(f"Specified target folder does not exist: {target_folder}")
        return

    folder_data = []
    folder_labels = []
    for target_image in target_folder_path.glob('*.jpg'):
        for folder in root_folder.iterdir():
            data, label = compare_image_with_folder_tsne(target_image, folder, api)
            folder_data.extend(data)
            folder_labels.extend(label)
    return np.array(folder_data), folder_labels


def tsne_plot_difference():
    root_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark"
    target_folder = "clock"
    api = SiameseNetworkAPI()

    root_folder = Path(root_folder)
    data, labels = compare_folder_with_others_tsne(root_folder, target_folder, api)

    # Ensure data is a 2D array
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)  # Flattening to 2D

    # Use PCA for initial dimensionality reduction
    pca = PCA(n_components=50)
    data_reduced = pca.fit_transform(data)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data_reduced)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    unique_labels = label_encoder.classes_

    # Plotting
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='tab20', edgecolor='k', alpha=0.7,
                          s=60)
    # Add a color bar with class labels
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Class')
    cbar.ax.set_yticklabels(unique_labels)
    plt.clim(-0.5, len(unique_labels) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization of image embeddings differences from Siamese for class clock', fontsize=14)

    # Generate handles and labels for the legend
    handles, _ = scatter.legend_elements()
    legend_labels = [unique_labels[i] for i in range(len(handles))]
    plt.legend(handles, legend_labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tsne_plot_difference()
