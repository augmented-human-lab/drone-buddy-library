import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from torch import nn
from torchvision.io import read_image

from dronebuddylib.atoms.objectidentification.resources.matching.SiameseNetworkAPI import SiameseNetworkAPI
from dronebuddylib.atoms.objectidentification.resources.matching.dataset import SiameseDataset
from dronebuddylib.atoms.objectidentification.resources.matching.model import SiameseModel
from dronebuddylib.atoms.objectidentification.resources.matching.tune_api import tune, loadModel

import torch
import torchvision.transforms as T
import itertools


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
    image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\new_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data\malsha_cup\malsha_cup_6.jpg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\malsha_cup\WhatsApp Image 2024-04-19 at 4.23.36 PM (2).jpeg"

    image_2 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\clock\7.jpg"
    # image_1 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\clock\6.jpg"
    image_1 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\cup\5.jpg"

    obj_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    image = cv2.imread(image_1)

    api = SiameseNetworkAPI()
    # Get the inference results
    all_similarities = api.inference(image)

    # Print similarities
    print("\nSimilarities:")
    #     create table to display the results
    table_data = []
    last_image_name = None

    for image_name, image_similarities in all_similarities.items():

        for class_name, similarities in image_similarities.items():
            # get mean similarity
            mean_similarity = sum(similarities) / len(similarities)
            # get mode similarity
            mode_similarity = max(set(similarities), key=similarities.count)
            # get median similarity
            sorted_similarities = sorted(similarities)
            n = len(sorted_similarities)
            if n % 2 == 0:
                median = (sorted_similarities[n // 2 - 1] + sorted_similarities[n // 2]) / 2

            argmax_index = max(range(len(similarities)), key=similarities.__getitem__)
            argmin_index = min(range(len(similarities)), key=similarities.__getitem__)
            argmax_value = similarities[argmax_index]
            argmin_value = similarities[argmin_index]
            recog_class = image_name.split("_")[1]
            # Only add the image name if it's different from the last one processed
            if image_name != last_image_name:
                current_val = [image_name, recog_class, class_name, f"{argmax_value:.6f}", f"{argmin_value:.6f}",
                               mean_similarity, mode_similarity, median]
                last_image_name = image_name
            else:
                # If the image name is the same, leave that field empty for better readability
                current_val = ["", "", class_name, f"{argmax_value:.6f}", f"{argmin_value:.6f}", mean_similarity,
                               mode_similarity, median]

            table_data.append(current_val)

    print(tabulate(table_data,
                   headers=["Image name", "Recognized object class", "Class", "Maximum score", "Minimum score", "Mean",
                            "Mode", "Median"], tablefmt="grid"))


def benchmark_model():
    image_2 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\clock\7.jpg"
    # image_1 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\clock\6.jpg"
    image_1 = r"C:\Users\Public\projects\temp_dronebuddy\drone-buddy-library\dronebuddylib\offline\molecules\memorized_obj_photo\cup\5.jpg"

    obj_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    all_img_of_obj = load_images_from_folder(obj_folder)
    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)
    api = SiameseNetworkAPI(all_img_of_obj)
    api.two_image_inference(image_1, image_2)
    #     load images from the folder
    #  compare iterations of the same image
    #  compare different images
    data_set = SiameseDataset(obj_folder)
    # image_sample_one is a sample of 100 images (path) taken from the training set folder
    # image_sample_two is a sample of 100 images (path) taken from the training set folder
    # image_one_class is the class of the image_sample_one
    # image_two_class is the class of the image_sample_two
    # same_class_details is a list of 100 elements, each element is 0 or 1, 0 means the two images are from different classes,
    # 1 means the two images are from the same class

    image_sample_one, image_sample_two, image_one_class, image_two_class, same_class_details = data_set.get_sample()


import logging

logging.basicConfig(level=logging.INFO, filename='process.log')


def batch_process_images(images, api, class_name, same_class, other_class_name=None):
    results = []
    for image1, image2 in itertools.combinations(images, 2):
        try:
            image_1 = cv2.imread(str(image1))
            image_2 = cv2.imread(str(image2))
            if image_1 is None or image_2 is None:
                raise ValueError(f"Failed to load one of the images: {image1}, {image2}")
            output = api.two_image_inference(image_1, image_2)
            result = {
                "Class": class_name,
                "TargetImage": image1.name,
                "ComparedImage": image2.name,
                "Same_Class": same_class,
                "Other_Class": other_class_name if not same_class else class_name,
                "Output": output
            }
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {image1} and {image2}: {str(e)}")
    return results


def compare_images_in_classes(root_folder, api, batch_size=10):
    root_folder = Path(root_folder)
    all_results = []

    for class_folder in root_folder.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob('*.jpg'))
            # Process same class comparisons in batches
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                results = batch_process_images(batch, api, class_folder.name, True)
                all_results.extend(results)

            # Process different class comparisons
            other_classes = [cf for cf in root_folder.iterdir() if cf.is_dir() and cf != class_folder]
            for other_class in other_classes:
                other_images = list(other_class.glob('*.jpg'))
                for image1 in images:
                    for j in range(0, len(other_images), batch_size):
                        other_batch = other_images[j:j + batch_size]
                        results = batch_process_images([image1] + other_batch, api, class_folder.name, False,
                                                       other_class.name)
                        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv("image_comparison_results.csv", index=False)
    return df


def detection():
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\blaaa.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_lab_tabel.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\random_table_with_mal_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\hot_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\long_distance.jpeg"
    image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\new_bottle.jpeg"
    # image_path = r"C:\Users\Public\projects\drone-buddy-library\test\object_images\p_bottle.jpeg"

    obj_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    all_img_of_obj = load_images_from_folder(obj_folder)
    image = cv2.imread(image_path)
    api = SiameseNetworkAPI(all_img_of_obj)
    api.get_detected_objects(image)


def compare_image_with_folder(target_image_path, folder_path, api):
    """
    Compare a specific image with all images in a given folder using a specified API and save results to a CSV file.

    Args:
    target_image_path (str): Path to the target image to compare.
    folder_path (str): Path to the folder containing images to compare against.
    api (object): An instance of the class with the `two_image_inference` method.

    Returns:
    DataFrame: A DataFrame containing the comparison results.
    """
    results = []
    target_image = cv2.imread(target_image_path)
    if target_image is None:
        logging.error(f"Failed to load the target image: {target_image_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the image cannot be loaded

    # Ensure the target folder exists and is a directory
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logging.error(f"The specified folder path is not a directory: {folder_path}")
        return pd.DataFrame()

    for image_path in folder_path.glob('*.jpg'):  # Adjust the glob pattern if using different file types
        try:
            comparison_image = cv2.imread(str(image_path))
            if comparison_image is None:
                logging.error(f"Failed to load comparison image: {image_path}")
                continue
            output = api.two_image_inference(target_image, comparison_image)
            is_same_class = folder_path.name == target_image_path.parent.name
            results.append({
                "TargetImage": target_image_path.name,
                "TargetClass": target_image_path.parent.name,
                "ComparedImage": image_path.name,
                "ComparedClass": image_path.parent.name,
                "IsSameClass": is_same_class,
                "Output": output.item()
            })
        except Exception as e:
            logging.error(f"Error comparing {target_image_path} and {image_path}: {str(e)}")

    df = pd.DataFrame(results)

    # Save or append results to a CSV file
    csv_file_path = 'resources/model/results/image_comparison_results_fine_tuned_new_data_set_50.csv'
    if Path(csv_file_path).exists():
        df.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file_path, mode='w', header=True, index=False)

    return df


def compare_image_with_folder_tsne(target_image_path, folder_path, api):
    """
    Compare a specific image with all images in a given folder using a specified API and save results to a CSV file.

    Args:
    target_image_path (str): Path to the target image to compare.
    folder_path (str): Path to the folder containing images to compare against.
    api (object): An instance of the class with the `two_image_inference` method.

    Returns:
    DataFrame: A DataFrame containing the comparison results.
    """
    target_image = cv2.imread(target_image_path)
    if target_image is None:
        logging.error(f"Failed to load the target image: {target_image_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the image cannot be loaded

    # Ensure the target folder exists and is a directory
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logging.error(f"The specified folder path is not a directory: {folder_path}")
        return pd.DataFrame()
    data = []
    labels = []
    for image_path in folder_path.glob('*.jpg'):  # Adjust the glob pattern if using different file types
        try:
            comparison_image = cv2.imread(str(image_path))
            if comparison_image is None:
                logging.error(f"Failed to load comparison image: {image_path}")
                continue
            output = api.two_image_inference_difference(target_image, comparison_image)
            is_same_class = folder_path.name == target_image_path.parent.name
            data.append(output)
            label_name = target_image_path.parent.name + "_" + image_path.parent.name + "_" + is_same_class
            labels.append(label_name)

        except Exception as e:
            logging.error(f"Error comparing {target_image_path} and {image_path}: {str(e)}")

    return data, labels


def compare_folder_with_others(root_folder, target_folder, api):
    """
    Compare each image in a target folder with images in all other folders under the root directory.

    Args:
    root_folder (str): The root directory containing all class folders.
    target_folder (str): The specific folder within the root to compare its images against all others.
    api (object): An instance of the class with the `two_image_inference` method.

    Returns:
    None: Results are directly saved to a CSV file.
    """
    root_folder = Path(root_folder)
    target_folder_path = root_folder / target_folder

    if not target_folder_path.is_dir():
        logging.error(f"Specified target folder does not exist: {target_folder}")
        return

    # Iterate over each image in the target folder
    for target_image in target_folder_path.glob('*.jpg'):
        for folder in root_folder.iterdir():
            compare_image_with_folder(target_image, folder, api)


def compare_folder_with_others_tsne(root_folder, target_folder, api):
    """
    Compare each image in a target folder with images in all other folders under the root directory.

    Args:
    root_folder (str): The root directory containing all class folders.
    target_folder (str): The specific folder within the root to compare its images against all others.
    api (object): An instance of the class with the `two_image_inference` method.

    Returns:
    None: Results are directly saved to a CSV file.
    """
    root_folder = Path(root_folder)
    target_folder_path = root_folder / target_folder

    if not target_folder_path.is_dir():
        logging.error(f"Specified target folder does not exist: {target_folder}")
        return

    folder_data = []
    folder_labels = []
    # Iterate over each image in the target folder
    for target_image in target_folder_path.glob('*.jpg'):
        for folder in root_folder.iterdir():
            data, label = compare_image_with_folder_tsne(target_image, folder, api)
            folder_data.append(data)
            folder_labels.append(label)
    return folder_data, folder_labels


def plot_comparison_results(csv_file_path):
    """
    Plot the comparison results from a CSV file where points are colored based on class similarity.

    Args:
    csv_file_path (str): Path to the CSV file containing comparison results.
    """
    # Load the DataFrame from CSV
    df = pd.read_csv(csv_file_path)

    # Check if the DataFrame is not empty
    if df.empty:
        print("The DataFrame is empty. No data to plot.")
        return

    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Plot data points where classes are the same
    same_class = df[df['IsSameClass'] == True]
    plt.scatter(same_class.index, same_class['Output'], color='green', label='Same Class', alpha=0.5)

    # Plot data points where classes are different
    different_class = df[df['IsSameClass'] == False]
    plt.scatter(different_class.index, different_class['Output'], color='red', label='Different Class', alpha=0.5)

    # Adding labels and title
    plt.title('Comparison of Images by Class Similarity')
    plt.xlabel('Comparison Index')
    plt.ylabel('Output')
    plt.legend()

    # Show the plot
    plt.show()


def benchmarking():
    root_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data"
    target_image = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark\clock\1.jpg"
    target_folder = "jane_bag"
    all_img_of_obj = load_images_from_folder(root_folder)

    api = SiameseNetworkAPI(all_img_of_obj)

    target_root_folder = Path(root_folder + "\\" + target_folder)
    root_folder = Path(root_folder)
    target_image_path = Path(target_image)
    # compare_image_with_folder(target_image_path, root_folder, api)
    compare_folder_with_others(root_folder, target_folder, api)


def tsne_plot_difference():
    # two_image_inference_difference
    root_folder = r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark"
    target_folder = "clock"
    all_img_of_obj = load_images_from_folder(root_folder)

    api = SiameseNetworkAPI(all_img_of_obj)

    root_folder = Path(root_folder)
    data, labels = compare_folder_with_others_tsne(root_folder, target_folder, api)
    data = np.array(data)
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Class')
    plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization of image embeddings differences from Siamese')
    plt.show()


def tsne_plotting():
    api = SiameseNetworkAPI()

    folder_path = Path(
        r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\benchmark\\" + "clock")

    data = []
    labels = []

    for image_path in folder_path.glob('*.jpg'):  # Adjust the glob pattern if using different file types
        try:
            comparison_image = cv2.imread(str(image_path))
            if comparison_image is None:
                logging.error(f"Failed to load comparison image: {image_path}")
                continue
            output = api.get_embeddings(comparison_image)
            labels.append(image_path.name)
            data.append(output)
            print(output)
        except Exception as e:
            logging.error(f"Error comparing {image_path}: {str(e)}")
    print(labels)
    digits = load_digits()
    X = digits.data
    Y = digits.target

    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    X_reduced = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap='viridis', edgecolor='k', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Digits dataset visualized using t-SNE')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()


def tsne_plotting_all():
    base_path = Path(
        r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data")
    # base_path = Path(r"C:\path_to_your_folders")  # Modify to your actual base path
    class_folders = [f for f in base_path.iterdir() if f.is_dir()]

    api = SiameseNetworkAPI()  # Assuming this is your API for getting embeddings
    data = []
    labels = []

    # Iterate over each class folder
    for folder in class_folders:
        for image_path in folder.glob('*.jpg'):  # Assuming .jpg files, modify if different
            try:
                comparison_image = cv2.imread(str(image_path))
                if comparison_image is None:
                    logging.error(f"Failed to load comparison image: {image_path}")
                    continue
                # Assuming 'get_embeddings' method returns a numpy array
                output = api.get_embeddings(comparison_image).detach().cpu().numpy()
                if output.ndim > 1:
                    output = output.squeeze()
                data.append(output)
                labels.append(folder.name)
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")

    data = np.array(data)
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data)
    # get all folder names
    folder_names = [f.name for f in class_folders]
    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k', alpha=0.6)
    # Create a legend with unique labels
    handles, _ = scatter.legend_elements()
    unique_labels = [label_encoder.inverse_transform([i])[0] for i in range(len(handles))]
    plt.legend(handles, unique_labels, title="Classes")
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Class')
    plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
    plt.xticks([])
    plt.yticks([])

    plt.title('t-SNE visualization of image embeddings for ' + str(folder_names))
    plt.show()


def tsne_plotting_given(compare_folders):
    base_path = Path(
        r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\objectidentification\resources\model\data\training_data")
    #  select the folders which are only in the compare_floders
    class_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name in compare_folders]

    print("class folders", class_folders)
    api = SiameseNetworkAPI()  # Assuming this is your API for getting embeddings
    data = []
    labels = []

    # Iterate over each class folder
    for folder in class_folders:
        for image_path in folder.glob('*.jpg'):  # Assuming .jpg files, modify if different
            try:
                comparison_image = cv2.imread(str(image_path))
                if comparison_image is None:
                    logging.error(f"Failed to load comparison image: {image_path}")
                    continue
                # Assuming 'get_embeddings' method returns a numpy array
                output = api.get_embeddings(comparison_image).detach().cpu().numpy()
                if output.ndim > 1:
                    output = output.squeeze()
                data.append(output)
                labels.append(folder.name)
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")

    data = np.array(data)
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(data)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_encoded, cmap='viridis', edgecolor='k', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)), label='Class')
    plt.clim(-0.5, len(label_encoder.classes_) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization of image embeddings for ' + str(compare_folders))
    plt.show()


# forward_difference_tsne

if __name__ == '__main__':
    # update_memory()
    # inference()
    # evaluate()
    # detection()
    # benchmarking()
    #
    # plot_comparison_results('resources/model/results/image_comparison_results_fine_tuned_normal.csv')
    # plot_comparison_results('image_comparison_results.csv')
    # plot_comparison_results('resources/model/results/image_comparison_results_fine_tuned_normal.csv')
    # plot_comparison_results('image_comparison_results_fine_tuned_changed_difference.csv')
    # plot_comparison_results('image_comparison_results_fine_tuned_changed_difference_no_sigmoid.csv')
    # plot_comparison_results('image_comparison_results_fine_tuned_changed_difference_with_sigmoid.csv')
    # tsne_plotting()
    tsne_plotting_all()
    # tsne_plot_difference()
    # tsne_plotting_given([ "malsha_cup", "dinithi_cup"])
