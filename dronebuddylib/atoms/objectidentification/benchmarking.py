import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from keras.src.applications.densenet import DenseNet121, preprocess_input
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageNet
from torchvision.models import EfficientNet_V2_S_Weights, ResNet50_Weights, resnet50, efficientnet_v2_s, resnet101, \
    ResNet101_Weights

from dronebuddylib.utils.enums import FeatureExtractors

import tensorflow as tf

from torchvision.transforms import functional as F


def benchmark_feature_extractors_for_dataset(cnn_name=FeatureExtractors.DENSENET121, dataset_path=None):
    """
    Benchmark the feature extractors for a dataset
    :param cnn_name: str, the name of the feature extractor
    :param dataset_path: str, the path to the dataset
    :return: None
    """

    if cnn_name == FeatureExtractors.DENSENET121:
        model = DenseNet121(weights='imagenet', include_top=False)
    else:
        model = None

    model.summary()

    data = []
    labels = []

    #  read the folders from the dataset path
    base_path = Path(dataset_path)
    # base_path = Path(r"C:\path_to_your_folders")  # Modify to your actual base path
    class_folders = [f for f in base_path.iterdir() if f.is_dir()]

    for folder in class_folders:
        data = []
        labels = []

        for image_path in folder.glob('*.jpg'):  # Assuming .jpg files, modify if different
            img = load_and_preprocess_img(cnn_name, image_path)
            features = extract_features(cnn_name, model, img)
            features_flattened = features.flatten()

            data.append(features_flattened)
            labels.append(folder.name)

        # start plottimg to a tsne
        data = np.array(data)
        # Encode labels to integers
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        # t-SNE transformation
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_reduced = tsne.fit_transform(data)

        # Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k',
                              alpha=0.6)
        plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Class')
        plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE visualization of image embeddings for DenseNet121 features for class : ' + folder.name)
        plt.show()

    # load the dataset from folder path


def benchmark_feature_extractors_for_dataset_all(cnn_name=FeatureExtractors.DENSENET121, dataset_path=None):
    """
    Benchmark the feature extractors for a dataset
    :param cnn_name: str, the name of the feature extractor
    :param dataset_path: str, the path to the dataset
    :return: None
    """

    if cnn_name == FeatureExtractors.DENSENET121:
        model = DenseNet121(weights='imagenet', include_top=False)
        model.summary()

    elif cnn_name == FeatureExtractors.EFFICIENTNETV2:
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.eval()

    elif cnn_name == FeatureExtractors.RESNET50:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.eval()

    elif cnn_name == FeatureExtractors.RESENT101:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        model.eval()
    else:
        model = None

    data = []
    labels = []

    # Read the folders from the dataset path
    base_path = Path(dataset_path)
    class_folders = [f for f in base_path.iterdir() if f.is_dir()]
    folder_names = [f.name for f in class_folders]
    for folder in class_folders:
        for image_path in folder.glob('*.jpg'):  # Assuming .jpg files, modify if different
            img = load_and_preprocess_img(cnn_name, image_path)
            features = extract_features(cnn_name, model, img)
            # Flatten the features to a 2D array
            features_flattened = features.flatten()
            data.append(features_flattened)
            labels.append(folder.name)

    # Convert to numpy arrays
    data = np.array(data)
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data)

    # Plotting
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k', alpha=0.6)
    # Create a legend with unique labels
    handles, _ = scatter.legend_elements()
    unique_labels = [label_encoder.inverse_transform([i])[0] for i in range(len(handles))]
    plt.legend(handles, unique_labels, title="Classes")
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Class')
    plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization of image embeddings for ' + str(cnn_name.value) + ' features of ' + str(
        folder_names) + ' classes')
    plt.show()


def load_and_preprocess_img(cnn_name, img_path):
    if cnn_name == FeatureExtractors.DENSENET121:
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    else:
        img = Image.open(img_path).convert('RGB')
        img = F.resize(img, [224, 224])
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.unsqueeze(0)
        return img


def extract_features(cnn_name, model, preprocessed_img):
    if cnn_name == FeatureExtractors.DENSENET121:
        features = model.predict(preprocessed_img)
    else:
        with torch.no_grad():
            features = model(preprocessed_img).numpy()
    return features


#  wirite main methjod to run the benchmarking
if __name__ == '__main__':
    data_path = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    # benchmark_feature_extractors_for_dataset(cnn_name=FeatureExtractors.DENSENET121, dataset_path=data_path)
    # benchmark_feature_extractors_for_dataset_all(cnn_name=FeatureExtractors.DENSENET121, dataset_path=data_path)
    # benchmark_feature_extractors_for_dataset_all(cnn_name=FeatureExtractors.EFFICIENTNETV2, dataset_path=data_path)
    # benchmark_feature_extractors_for_dataset_all(cnn_name=FeatureExtractors.RESNET50, dataset_path=data_path)
    benchmark_feature_extractors_for_dataset_all(cnn_name=FeatureExtractors.EFFICIENTNETV2, dataset_path=data_path)
