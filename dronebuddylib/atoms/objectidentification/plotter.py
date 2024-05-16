import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

    same_class_data = []
    different_class_data = []
    same_class_labels = []
    different_class_labels = []

    for image_path in folder_path.glob('*.jpg'):
        try:
            comparison_image = cv2.imread(str(image_path))
            if comparison_image is None:
                logging.error(f"Failed to load comparison image: {image_path}")
                continue
            output = api.two_image_inference_difference(target_image, comparison_image).detach().numpy()
            output_flattened = output.flatten()

            is_same_class = folder_path.name == target_image_path.parent.name
            label_name = "Same Class" if is_same_class else "Different Class"

            if is_same_class:
                same_class_data.append(output_flattened)
                same_class_labels.append(label_name)
            else:
                different_class_data.append(output_flattened)
                different_class_labels.append(label_name)

        except Exception as e:
            logging.error(f"Error comparing {target_image_path} and {image_path}: {str(e)}")

    return (np.array(same_class_data), same_class_labels), (np.array(different_class_data), different_class_labels)


def compare_folder_with_others_tsne(root_folder, target_folder, api):
    root_folder = Path(root_folder)
    target_folder_path = root_folder / target_folder

    if not target_folder_path.is_dir():
        logging.error(f"Specified target folder does not exist: {target_folder}")
        return

    same_class_data_all = []
    different_class_data_all = []
    same_class_labels_all = []
    different_class_labels_all = []

    for target_image in target_folder_path.glob('*.jpg'):
        for folder in root_folder.iterdir():
            (same_data, same_labels), (diff_data, diff_labels) = compare_image_with_folder_tsne(target_image, folder,
                                                                                                api)
            same_class_data_all.extend(same_data)
            different_class_data_all.extend(diff_data)
            same_class_labels_all.extend(same_labels)
            different_class_labels_all.extend(diff_labels)

    return (np.array(same_class_data_all), same_class_labels_all), (
    np.array(different_class_data_all), different_class_labels_all)


def tsne_plot_difference():
    root_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark"
    target_folder = "clock"
    api = SiameseNetworkAPI()

    root_folder = Path(root_folder)
    (same_class_data, same_class_labels), (diff_class_data, diff_class_labels) = compare_folder_with_others_tsne(
        root_folder, target_folder, api)

    # Combine same and different class data for PCA and t-SNE
    combined_data = np.concatenate((same_class_data, diff_class_data), axis=0)
    combined_labels = same_class_labels + diff_class_labels

    # Ensure combined_data is a 2D array
    if combined_data.ndim > 2:
        combined_data = combined_data.reshape(combined_data.shape[0], -1)  # Flattening to 2D

    # Use PCA for initial dimensionality reduction
    pca = PCA(n_components=50)
    data_reduced = pca.fit_transform(combined_data)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data_reduced)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(combined_labels)
    unique_labels = label_encoder.classes_

    # Clustering for counting
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_reduced)
    cluster_labels = kmeans.labels_

    # Counting items in clusters
    cluster_counts = np.bincount(cluster_labels)
    print(f"Cluster 0 count: {cluster_counts[0]}")
    print(f"Cluster 1 count: {cluster_counts[1]}")

    # Plotting
    plt.figure(figsize=(20, 14))  # Increased figure size
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='coolwarm', edgecolor='k', alpha=0.7,
                          s=60)
    # Add a color bar with class labels
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Class')
    cbar.ax.set_yticklabels(unique_labels, fontsize=14)  # Increased font size for better readability
    plt.clim(-0.5, len(unique_labels) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization of image embeddings differences from Siamese for class ' + target_folder,
              fontsize=18)

    # Generate handles and labels for the legend
    handles, _ = scatter.legend_elements()
    legend_labels = [unique_labels[i] for i in range(len(handles))]
    plt.legend(handles, legend_labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make space for the legend
    plt.show()

    # For debugging: print the sizes of the same and different class arrays
    print(f"Same class data size: {same_class_data.shape}")
    print(f"Different class data size: {diff_class_data.shape}")


if __name__ == '__main__':
    tsne_plot_difference()
