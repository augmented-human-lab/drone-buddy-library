import os
import pickle
import re
import threading
import time
from asyncio import Future
from pathlib import Path

import cv2
import numpy as np
import pkg_resources
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
import seaborn as sns

from dronebuddylib import PlaceRecognitionException
from dronebuddylib.atoms.objectidentification.i_object_identification import IObjectIdentification
from dronebuddylib.atoms.objectidentification.object_identification_result import IdentifiedObjectObject, \
    IdentifiedObjects
from dronebuddylib.atoms.objectidentification.resources.matching.SiameseNetworkAPI import SiameseNetworkAPI
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.models.execution_status import ExecutionStatus
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check, write_to_file_longer, read_from_file, overwrite_file
from dronebuddylib.atoms.objectidentification.resources.matching.tune_api import tune

logger = Logger()


def hook_fn(module, input, output):
    intermediate_features.append(output)


class ObjectIdentificationSiameseSiamese(IObjectIdentification):
    """
    A class to perform object identification using YOLO V8 and Siamese Network.
    """
    progress_event = threading.Event()

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the YOLO V8 object detection engine with the given engine configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations for the object detection engine.
        """
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        model_name = configs.get(AtomicEngineConfigurations.OBJECT_IDENTIFICATION_SIAMESE_YOLO_VERSION.name,
                                 configs.get(AtomicEngineConfigurations.OBJECT_IDENTIFICATION_SIAMESE_YOLO_VERSION))

        self.custom_data_set_path = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_TRAINING_DATA_SET_PATH.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_TRAINING_DATA_SET_PATH))

        if model_name is None:
            model_name = "yolov8n.pt"

        # Load YOLOv8 model
        self.yolo_model = YOLO(model_name)

        logger.log_info(self.get_class_name(), ':Initializing with model with ' + model_name + '')

        self.detector = YOLO(model_name)
        self.object_names = self.detector.names
        logger.log_debug(self.get_class_name(), 'Initialized the Siamese object identification')
        self.model = SiameseNetworkAPI()

    def remember_object(self, image=None, type=None, name=None, drone_instance=None, on_start=None,
                        on_training_set_complete=None, on_validation_set_complete=None):
        """
        Starts the process to remember an object by creating a training and validation dataset.

        Args:
            image: The image of the object to remember.
            type: The type of object.
            name: The name of the object.
            drone_instance: The instance of the drone used to capture images.
            on_start: Callback when the process starts.
            on_training_set_complete: Callback when the training set is complete.
            on_validation_set_complete: Callback when the validation set is complete.
        """
        logger.log_info(self.get_class_name(), 'Starting to remember object: type : ' + type + ' : ' + name)
        # Trigger start callback

        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "training data set creation",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create training data set for ' + name)
        self.create_dataset(type, name, 0, drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

        logger.log_info(self.get_class_name(), 'Starting to create validation data set for ' + name)
        self.create_dataset(type, name, 1, drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

    def progress_bar(self, done_event, title="Training Progress"):
        """
        Displays a progress bar for the training process.

        Args:
            done_event: Event to signal when the progress is complete.
            title (str): The title of the progress bar.
        """
        # Initialize tqdm with an unknown total duration (using total=None)
        with tqdm(total=None, desc=title, unit=" step") as pbar:
            while not done_event.is_set():
                pbar.update()  # Update the progress bar by one step
                time.sleep(0.1)  # Sleep a short time before updating again
            pbar.update()  # Ensure the progress bar completes if it hasn't already

    def create_memory(self, changes=None, drone_instance=None):
        """
        Creates and trains a KNN classifier using the collected data.

        Args:
            changes: Any changes to the model or data.

        Returns:
            dict: A dictionary containing performance metrics of the trained model.
        """
        classifier_path = "resources/model/model/object_classifier_"
        file_path = pkg_resources.resource_filename(__name__, "resources/model/model/classifier_index.txt")
        classifier_index_read = read_from_file(file_path)
        classifier_index = int(classifier_index_read) + 1
        overwrite_file(file_path, str(classifier_index))

        if self.custom_data_set_path is not None:
            training_path_name = self.custom_data_set_path
        else:
            training_path_name = pkg_resources.resource_filename(__name__, "resources/model/data/training_data")

        classifier_path_name = pkg_resources.resource_filename(__name__,
                                                               classifier_path + str(classifier_index) + ".clf")

        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "training KNN classifier",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create memory. Training KNN classifier')

        # Create a Future object
        future = Future()
        # Start the training in a separate thread
        train_thread = threading.Thread(target=self.train)

        train_thread.start()

        # Start the progress bar in a separate thread
        progress_bar_thread = threading.Thread(target=self.progress_bar, args=(self.progress_event,))
        progress_bar_thread.start()

        # Wait for the training to finish
        train_thread.join()
        # Ensure the progress bar thread completes
        progress_bar_thread.join()

        logger.log_info(self.get_class_name(),
                        'Successfully created memory. KNN classifier trained successfully. Model saved at: '
                        + classifier_path_name)
        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "training KNN classifier",
                                              "COMPLETED")

        try:
            output = future.result()  # This will also re-raise any exception set by set_exception
            logger.log_info(self.get_class_name(), 'Successfully tested the classifier')
            return output
        except Exception as e:
            print(f"Error occurred: {e}")
            return PlaceRecognitionException("Error occurred while training the classifier", 500, e)

    def train(self, feature_extractor_model="efficientnetv2", num_samples=100, emb_size=20, epochs=10, lr=1e-5,
              batch_size=4, train_val_split=0.8, num_workers=1, seed=0, output_folder_name=None, lr_scheduler=False,
              pretrained_weights=None):
        """
        Trains the model using the specified parameters.

        Args:
            feature_extractor_model (str): The model used for feature extraction (default is "efficientnetv2").
            num_samples (int): The number of samples to use for training (default is 100).
            emb_size (int): The size of the embedding (default is 20).
            epochs (int): The number of training epochs (default is 10).
            lr (float): The learning rate for training (default is 1e-5).
            batch_size (int): The batch size for training (default is 4).
            train_val_split (float): The ratio for splitting training and validation data (default is 0.8).
            num_workers (int): The number of worker threads to use for data loading (default is 1).
            seed (int): The random seed for reproducibility (default is 0).
            output_folder_name (str, optional): The folder name for saving the output.
            lr_scheduler (bool): Flag to use learning rate scheduler (default is False).
            pretrained_weights (str, optional): Path to pretrained weights for the model.

        Returns:
            dict: A dictionary containing performance metrics such as accuracy and precision of the trained model.
        """
        tune(feature_extractor_model, num_samples, emb_size, epochs, lr, batch_size, train_val_split, num_workers, seed,
             output_folder_name, lr_scheduler, pretrained_weights)

    def image_files_in_folder(self, folder):
        """
        Lists image files in a specified folder.

        Args:
            folder: Folder to search for image files.

        Returns:
            list: List of image file paths.
        """
        return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

    def recognize_objects(self, image):
        """
        Recognizes objects depicted in the given image. If the confidence of the
        predictions is below a given threshold, the object is classified as 'unknown'.

        Args:
            image: The image of the objects to recognize.

        Returns:
            RecognizedObjects: The recognized objects with their associated probabilities.
        """
        # Get the inference results
        all_similarities = self.model.inference(image)
        return all_similarities

    def create_dataset(self, object_type, object_name, data_mode, drone_instance):
        """
        Generates a dataset for a given place name, optionally using a drone for image collection.
        Different modes support training, validation, and testing data collection.

        Args:
            place_name (str): The name of the place for which to create the dataset.
            data_mode (int): The mode of dataset creation (e.g., training, validation).
            drone_instance: An optional drone instance to use for collecting images.
        """
        count = 0

        if data_mode == 0:
            data_set_size = 50
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/model/data/training_data/" + object_name)

        elif data_mode == 1:
            data_set_size = 20
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/model/data/validation_data/" + object_name)
            type = "validation"
        else:
            data_set_size = 50
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/model/data/training_data/" + object_name)

        logger.log_warning(self.get_class_name(),
                           'Data set creating. ' + str(320) +
                           ' data collected. Data set can be found at the path: ' + path_name)
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        while True:
            frame = drone_instance.get_frame_read().frame

            file_name_path = (path_name + "\\" + object_name + "_" + str(count) + ".jpg")

            results = self.yolo_model(frame)

            for result in results:
                for index, cls_tensor in enumerate(result.boxes.cls):
                    cls_index = int(cls_tensor.item())  # Convert the tensor to an integer

                    if self.object_names[int(cls_index)] == object_type:
                        # extract the bounding box coordinates of the object
                        bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[
                            0]  # Assuming it's a PyTorch tensor
                        xmin, ymin, xmax, ymax = map(int, bbox_tensor)  # Convert to integer if necessary

                        # Crop image using OpenCV (NumPy slicing)
                        cropped_image = frame[ymin:ymax, xmin:xmax]

                        # convert the cropped image to RGB
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                        # Save the cropped image using OpenCV
                        cv2.imwrite(file_name_path, cropped_image)
                        logger.log_success(self.get_class_name(), object_name + 'Object image saved.')
                        count += 1
                        break

            self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "creating data set",
                                                  "PROGRESS", str(count) + "/" + str(320))

            if cv2.waitKey(1) == 27 or count == data_set_size:  # 27 is the Esc Key
                break

        cv2.destroyAllWindows()
        logger.log_warning(self.get_class_name(),
                           'Data set created successfully. ' + str(data_set_size) +
                           ' data collected. Data set can be found at the path: ' + path_name)

    def test_image(self, image_path):
        """
        Tests an image for object recognition.

        Args:
            image_path: Path to the image file.
        """
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        for result in results:
            for index, cls_tensor in enumerate(result.boxes.cls):
                cls_index = int(cls_tensor.item())  # Convert the tensor to an integer

                print(self.object_names[int(cls_index)])

                if self.object_names[int(cls_index)] == "bottle":
                    bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]  # Assuming it's a PyTorch tensor
                    xmin, ymin, xmax, ymax = map(int, bbox_tensor)  # Convert to integer if necessary

                    # Crop image using OpenCV (NumPy slicing)
                    cropped_image = image[ymin:ymax, xmin:xmax]
                    break

    def extract_features(self, model, img, layer_index=20):
        """
        Extracts features from an image using a specific layer of the model.

        Args:
            model: The model used for feature extraction.
            img: The image to extract features from.
            layer_index (int): The layer index for feature extraction.

        Returns:
            Extracted features from the specified layer.
        """
        global intermediate_features
        intermediate_features = []
        hook = model.model.model[layer_index].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(img)
        hook.remove()
        return intermediate_features[0]  # Access the first element of the list

    def preprocess_image(self, image):
        """
        Preprocesses an image for model input.

        Args:
            image: The image to preprocess.

        Returns:
            Preprocessed image tensor.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def extract_features_from_image(self, img, layer_index=20):
        """
        Extracts features from an image using a specific layer of the model.

        Args:
            img: The image to extract features from.
            layer_index (int): The layer index for feature extraction.

        Returns:
            Extracted features from the specified layer.
        """
        img = self.preprocess_image(img)
        features = self.extract_features(self.yolo_model, img, layer_index=layer_index)
        return features

    def extract_and_plot_features(self, img, layer_index=20, channel_index=5):
        """
        Extracts and plots features from an image.

        Args:
            img: The image to extract and plot features from.
            layer_index (int): The layer index for feature extraction.
            channel_index (int): The channel index for plotting.
        """
        img = self.preprocess_image(img)
        features = self.extract_features(self.yolo_model, img, layer_index)
        print(f"Features shape for {img}: {features.shape}")

        plt.figure(figsize=(10, 5))
        sns.heatmap(features[0][channel_index].cpu().numpy(), cmap='viridis', annot=False)
        plt.title(f'Features for {img} - Layer {layer_index} - Channel {channel_index}')
        plt.show()

    def get_class_name(self) -> str:
        """
        Gets the class name of the object detection implementation.

        Returns:
            str: The class name of the object detection implementation.
        """
        return 'OBJECT_IDENTIFICATION_SIAMESE'

    def get_algorithm_name(self) -> str:
        """
        Gets the algorithm name of the object detection implementation.

        Returns:
            str: The algorithm name of the object detection implementation.
        """
        return 'Siamese Object Identification'

    def get_required_params(self) -> list:
        """
        Gets the list of required configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of required configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_IDENTIFICATION_SIAMESE_YOLO_VERSION]

    def get_optional_params(self) -> list:
        """
        Gets the list of optional configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of optional configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_IDENTIFICATION_SIAMESE_YOLO_VERSION]
