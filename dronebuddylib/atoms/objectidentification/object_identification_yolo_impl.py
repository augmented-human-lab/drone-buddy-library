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
from dronebuddylib.atoms.objectidentification.i_object_recognition import IObjectRecognition
from dronebuddylib.atoms.objectidentification.object_recognition_result import RecognizedObjects, RecognizedObjectObject
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.models.execution_status import ExecutionStatus
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check, write_to_file_longer, read_from_file, overwrite_file

logger = Logger()


def hook_fn(module, input, output):
    intermediate_features.append(output)


class ObjectRecognitionYOLOImpl(IObjectRecognition):
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
        model_name = configs.get(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION)
        self.weights_path = configs.get(AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_WEIGHTS_PATH.name,
                                        configs.get(
                                            AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_WEIGHTS_PATH))
        self.drone_instance = configs.get(AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_DRONE_INSTANCE.name,
                                          configs.get(
                                              AtomicEngineConfigurations.OBJECT_RECOGNITION_YOLO_DRONE_INSTANCE))
        self.custom_knn_algorithm_name = configs.get(
            AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_ALGORITHM_NAME.name,
            configs.get(AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_ALGORITHM_NAME))
        self.custom_knn_neighbors = configs.get(
            AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_ALGORITHM_NEIGHBORS.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_ALGORITHM_NEIGHBORS))
        self.custom_knn_model_saving_path = configs.get(
            AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_MODEL_SAVING_PATH.name,
            configs.get(AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_MODEL_SAVING_PATH))
        self.custom_knn_threshold = configs.get(AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_MODEL_THRESHOLD.name,
                                                configs.get(
                                                    AtomicEngineConfigurations.OBJECT_RECOGNITION_KNN_MODEL_THRESHOLD))
        self.custom_knn_weights = configs.get(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_V3_WEIGHTS_PATH.name,
                                              configs.get(
                                                  AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_V3_WEIGHTS_PATH))
        self.custom_data_set_path = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_TRAINING_DATA_SET_PATH.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_TRAINING_DATA_SET_PATH))

        if model_name is None:
            model_name = "yolov8n.pt"

        # Load YOLOv8 model
        weights_path = Path(self.weights_path)
        self.yolo_model = YOLO(weights_path)

        logger.log_info(self.get_class_name(), ':Initializing with model with ' + model_name + '')

        self.detector = YOLO(model_name)
        self.object_names = self.detector.names
        logger.log_debug(self.get_class_name(), 'Initialized the YOLO object recognition')

    def remember_object(self, image=None, type=None, name=None, drone_instance=None, on_start=None,
                        on_training_set_complete=None,
                        on_validation_set_complete=None):
        logger.log_info(self.get_class_name(), 'Starting to remember object: type : ' + type + ' : ' + name)
        # Trigger start callback

        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "training data set creation",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create training data set for ' + name)
        self.create_dataset(type, name, 0, self.drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

        logger.log_info(self.get_class_name(), 'Starting to create validation data set for ' + name)
        self.create_dataset(type, name, 1, self.drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

    def progress_bar(self, done_event, title="Training Progress"):
        # Initialize tqdm with an unknown total duration (using total=None)
        with tqdm(total=None, desc=title, unit=" step") as pbar:
            while not done_event.is_set():
                pbar.update()  # Update the progress bar by one step
                time.sleep(0.1)  # Sleep a short time before updating again
            pbar.update()  # Ensure the progress bar completes if it hasn't already

    def create_memory(self, changes=None):
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

        # ------------------------------------------------------------------------------------------------------------
        #  TRAINING THE CLASSIFIER
        # ------------------------------------------------------------------------------------------------------------

        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "training KNN classifier",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create memory. Training KNN classifier')

        # Create a Future object
        future = Future()
        # Start the training in a separate thread
        train_thread = threading.Thread(target=self.train, args=(training_path_name,
                                                                 classifier_path_name, self.custom_knn_neighbors,
                                                                 self.custom_knn_algorithm_name,
                                                                 self.custom_knn_weights,
                                                                 classifier_index, changes,
                                                                 False, future))

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

    def train(self, train_dir, model_save_path=None, n_neighbors=3, knn_algo='auto', weights='distance',
              classifier_index=0, changes=None, verbose=True,
              future=None):
        """
           Trains the KNN classifier on a set of labeled images stored in a directory structure.
           Each sub-directory within the train directory should represent a class (place), and contain
           images corresponding to that place.

           Args:
               train_dir (str): The directory containing the training dataset.
               model_save_path (str, optional): Path to save the trained model.
               n_neighbors (int, optional): Number of neighbors to use for k-nearest neighbors voting.
               knn_algo (str, optional): Underlying algorithm to compute the nearest neighbors.
               weights (str, optional): Weight function used in prediction.
               classifier_index (int, optional): An index to uniquely identify the classifier.
               changes (str, optional): Description of any changes or versioning info.
               verbose (bool, optional): Enables verbose output during the training process.
               future (Future, optional): A Future object for asynchronous execution.

           Returns:
               dict: A dictionary containing performance metrics such as accuracy and precision of the trained model.
           """
        X = []
        y = []

        class_count = 0
        if knn_algo is None:
            knn_algo = 'auto'
        if n_neighbors is None:
            n_neighbors = 3
        if weights is None:
            weights = 'distance'

        training_start_time = time.time()

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            class_count += 1
            if verbose:
                logger.log_debug(self.get_class_name(), "Training for class : " + class_dir)
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in self.image_files_in_folder(os.path.join(train_dir, class_dir)):
                img = cv2.imread(img_path)
                features = self.extract_features_from_image(img)
                if features is not None and features.numel() > 0:
                    flat_features = features.view(-1).cpu().numpy()  # Flatten and convert to numpy array
                    X.append(flat_features)  # Append the flat feature array
                    y.append(class_dir)
                    if verbose:
                        logger.log_debug(self.get_class_name(), "feature extraction for  : " + img_path)
                else:
                    logger.log_error(self.get_class_name(), "No features extracted for: " + img_path)

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            logger.log_error(self.get_class_name(), "No features were extracted. Aborting training.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Create and train the KNN classifier

        # Train the KNN model
        knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo,
                                       weights=weights)  # Use 3 neighbors for simplicity
        knn_clf.fit(X_train, y_train)

        logger.log_debug(self.get_class_name(), "Training completed")

        # Evaluate the model
        predictions = knn_clf.predict(X_test)

        logger.log_debug(self.get_class_name(), "Testing completed")

        # Calculate and print the accuracy
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        f1_score = precision_recall_fscore_support(y_test, predictions, average='weighted')
        report = classification_report(y_test, predictions)

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1_score,
            "report": report
        }

        # Save the trained KNN classifier
        training_end_time = time.time()

        model_string = "model name: " + str(
            classifier_index) + ".clf" + ": reason : " + changes + " : no. of classes " + str(
            class_count) + ": time taken : " + str(
            (training_end_time - training_start_time)) + " seconds  :  accuracy: " + str(
            accuracy) + " : precision: " + str(precision) + " : f1_score: " + str(f1_score) + "\n"
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        # write the model name with teh accuracy into the classifier_data.txt

        file_path = pkg_resources.resource_filename(__name__, "resources/model/model/classifier_data.txt")

        write_to_file_longer(file_path, model_string)
        logger.log_info(self.get_class_name(),
                        'Training complete. It took ' + str(
                            (training_end_time - training_start_time) / 60) + ' minutes')
        self.progress_event.set()
        if future is not None:
            future.set_result(result)  # Set the result of the future
        return result

    def image_files_in_folder(self, folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

    def recognize_objects(self, image, top_n=3) -> RecognizedObjects:
        """
        Recognizes objects depicted in the given image. If the confidence of the
        predictions is below a given threshold, the object is classified as 'unknown'.

        Args:
            image: The image of the objects to recognize.

        Returns:
            RecognizedObjects: The recognized objects with their associated probabilities.
        """
        # Preprocess the image and extract features
        # image_tensor = self.preprocess_image(image)
        features = self.extract_features_from_image(image)

        # Check if features were successfully extracted
        if features is None or features.numel() == 0:
            logger.log_info(self.get_class_name(), "No features extracted from image.")
            return RecognizedObjects([], None)

        # Flatten the feature tensor to a 1D array if it's not already 1D
        flat_features = features.view(-1).cpu().numpy()

        # Load the trained model
        if self.custom_knn_model_saving_path is not None:
            model_path = self.custom_knn_model_saving_path
        else:
            classifier_index = read_from_file(
                pkg_resources.resource_filename(__name__, "resources/model/model/classifier_index.txt"))
            model_path = pkg_resources.resource_filename(__name__,
                                                         "resources/model/model/object_classifier_" + str(
                                                             classifier_index) + ".clf")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict the class of the object using the loaded model
        predicted_label = model.predict([flat_features])[0]
        probabilities = model.predict_proba([flat_features])[0]
        max_probability = np.max(probabilities)

        logger.log_info(self.get_class_name(),
                        f"Recognized object: {predicted_label} with confidence: {max_probability}")

        probabilities = model.predict_proba([flat_features])[0]  # Probabilities of all classes
        top_n_indices = np.argsort(probabilities)[::-1][:top_n]  # Indices of top 3 classes
        top_n_probabilities = probabilities[top_n_indices]  # Probabilities of top 3 classes
        top_n_labels = model.classes_[top_n_indices]  # Labels of top 3 classes

        recognized_places = []
        most_probable_place = RecognizedObjectObject(predicted_label, max_probability)
        # Log the top three predictions

        for i, (label, prob) in enumerate(zip(top_n_labels, top_n_probabilities)):
            logger.log_info(self.get_class_name(), f"Top {i + 1} predicted place: {label} with confidence: {prob}")
            recognized_places.append(RecognizedObjectObject(label, prob))

        return RecognizedObjects(most_probable_place, recognized_places)

    def create_dataset(self, object_type, object_name, data_mode, drone_instance):
        """
                      Generates a dataset for a given place name, optionally using a drone for image collection.
                      Different modes support training, validation, and testing data collection.

                      Args:
                          place_name (str): The name of the place for which to create the dataset.
                          data_mode (int): The mode of dataset creation (e.g., training, validation).
                          drone_instance: An optional drone instance to use for collecting images.

                      """
        # if drone_instance is not None:
        if True:
            count = 00

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

                # logger.log_info(self.get_class_name(), results)
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
                            # Ensure the file_name_path is correct and includes the file extension, like '.jpg' or '.png'
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
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        # logger.log_info(self.get_class_name(), results)
        for result in results:
            for index, cls_tensor in enumerate(result.boxes.cls):
                cls_index = int(cls_tensor.item())  # Convert the tensor to an integer

                print(self.object_names[int(cls_index)])

                if self.object_names[int(cls_index)] == "bottle":
                    # extract the bounding box coordinates of the object
                    # extract the bounding box coordinates of the object
                    bbox_tensor = result.boxes[index].xyxy.cpu().numpy().tolist()[0]  # Assuming it's a PyTorch tensor
                    xmin, ymin, xmax, ymax = map(int, bbox_tensor)  # Convert to integer if necessary

                    # Crop image using OpenCV (NumPy slicing)
                    cropped_image = image[ymin:ymax, xmin:xmax]

                    # Save the cropped image using OpenCV
                    # Ensure the file_name_path is correct and includes the file extension, like '.jpg' or '.png'

                    break

    def extract_features(self, model, img, layer_index=20):  ##Choose the layer that fit your application
        global intermediate_features
        intermediate_features = []
        hook = model.model.model[layer_index].register_forward_hook(hook_fn)
        print(hook)
        with torch.no_grad():
            model(img)
        hook.remove()
        return intermediate_features[0]  # Access the first element of the list

        # Make sure to preprocess the image since the input image must be 640x640x3

    def preprocess_image(self, image):
        # Define the transformations directly suitable for your model input size
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((640, 640)),  # Resize image to the size expected by the model
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        # Convert BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        image_tensor = transform(image_rgb)

        # Add a batch dimension if needed (assuming your model expects batches)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def extract_features_from_image(self, img, layer_index=20):
        img = self.preprocess_image(img)
        features = self.extract_features(self.yolo_model, img, layer_index=layer_index)
        return features

    # Plot the Features extracted
    def extract_and_plot_features(self, img, layer_index=20, channel_index=5):
        img = self.preprocess_image(img)

        features = self.extract_features(self.yolo_model, img, layer_index)
        print(features)
        # Print the shape of the features
        print(f"Features shape for {img}: {features.shape}")

        # Plot the features as a heatmap for a specific channel
        plt.figure(figsize=(10, 5))
        sns.heatmap(features[0][channel_index].cpu().numpy(), cmap='viridis', annot=False)
        plt.title(f'Features for {img} - Layer {layer_index} - Channel {channel_index}')
        plt.show()
        # Plot the Features extracted

    def get_class_name(self) -> str:
        """
        Gets the class name of the object detection implementation.

        Returns:
            str: The class name of the object detection implementation.
        """
        return 'OBJECT_RECOGNITION_YOLO'

    def get_algorithm_name(self) -> str:
        """
        Gets the algorithm name of the object detection implementation.

        Returns:
            str: The algorithm name of the object detection implementation.
        """
        return 'YOLO V8 Object Detection'

    def get_required_params(self) -> list:
        """
        Gets the list of required configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of required configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION]

    def get_optional_params(self) -> list:
        """
        Gets the list of optional configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of optional configuration parameters.
        """
        # Additional optional parameters can be added here
        return [AtomicEngineConfigurations.OBJECT_IDENTIFICATION_YOLO_WEIGHTS_PATH]
