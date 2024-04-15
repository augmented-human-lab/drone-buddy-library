import threading
import time
from asyncio import Future

import numpy as np
import pkg_resources
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from dronebuddylib.atoms.placerecognition.i_place_recognition import IPlaceRecognition
from dronebuddylib.atoms.placerecognition.place_recognition_result import RecognizedPlaces, RecognizedPlaceObject
from dronebuddylib.exceptions.PlaceRecognitionException import PlaceRecognitionException
from dronebuddylib.models import AtomicEngineConfigurations, EngineConfigurations
from dronebuddylib.models.execution_status import ExecutionStatus

from dronebuddylib.utils.logger import Logger

import cv2
import re
from sklearn import neighbors, preprocessing
import os
import os.path
import pickle
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support, classification_report
import joblib

from dronebuddylib.utils.utils import overwrite_file, read_from_file, config_validity_check, write_to_file, \
    write_to_file_longer
from time import sleep
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = Logger()


class PlaceRecognitionRFImpl(IPlaceRecognition):
    """
        This class implements place recognition using the k-nearest-neighbors (KNN) algorithm, leveraging image features
        extracted via a pre-trained ResNet model.

        When should I use this class?
        This implementation is ideal when you need to identify specific places or landmarks within a large dataset of known
        locations efficiently. It's particularly useful for applications in robotics, drones, or any system requiring
        geographical awareness from visual cues.

        Algorithm Description:
        The KNN classifier is trained on a dataset of images labeled with their corresponding places. For an unknown image,
         the classifier predicts the place by finding the k most similar images in its training set
          (based on the closest feature vectors under Euclidean distance) and performing a majority vote on their labels.

        For instance, if k=3 and the three closest images in the training set to the given image are two images
        of the Eiffel Tower and one image of the Statue of Liberty, the result would be 'Eiffel Tower'.

        * This implementation can weight the votes according to the distance of neighbors,
         giving closer neighbors more influence on the final outcome.

        Usage:

        1. Prepare a dataset of images for the places you want to recognize.
        Organize the images so that there is a sub-directory for each place within a main directory.

        2. Use the 'train' method of this class to train the classifier on your dataset.
         You can save the trained model to disk by specifying a 'model_save_path', allowing you to reuse the model without retraining.

        3. To recognize the place depicted in a new, unknown image, call the 'recognize_place' method with the image as input.

        NOTE: This implementation requires scikit-learn, NumPy, PyTorch, torchvision,
         and PIL to be installed for machine learning operations and image processing.
         Ensure these packages are installed in your environment:

        $ pip install scikit-learn numpy torch torchvision Pillow

       A class implementing KNN-based place recognition.

       This class uses k-nearest neighbors algorithm to recognize places by comparing
       the feature vectors of images extracted using a pre-trained ResNet model. It is
       capable of training a model with labeled images of places and predicting the
       place of an unknown image.

       Attributes:
           use_drone (bool): Indicates if a drone is used to create the dataset.
           progress_event (threading.Event): Signals when the training process is done.
           current_status (ExecutionStatus): Holds the current status of the operation.
           configs (EngineConfigurations): Configuration parameters for the engine.
       """

    use_drone = False
    # Event to signal when the training is done
    progress_event = threading.Event()
    current_status = None

    classifier_path = "resources/models/classifiers/rf/trained_place_rf_model"

    def __init__(self, engine_configurations: EngineConfigurations):
        """
               Initializes the PlaceRecognitionKNNImpl class with the given engine configurations.

               Args:
                   engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # Get configurations
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        self.configs = engine_configurations

        self.use_drone = bool(
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_USE_DRONE_TO_CREATE_DATASET.name,
                        configs.get(
                            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_USE_DRONE_TO_CREATE_DATASET)))
        self.drone_instance = configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_DRONE_INSTANCE.name,
                                          configs.get(
                                              AtomicEngineConfigurations.PLACE_RECOGNITION_RF_DRONE_INSTANCE))
        self.custom_knn_algorithm_name = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NAME.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NAME))
        self.custom_knn_neighbors = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NEIGHBORS.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NEIGHBORS))
        self.custom_knn_model_saving_path = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_SAVING_PATH.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_SAVING_PATH))
        self.custom_knn_threshold = configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_THRESHOLD.name,
                                                configs.get(
                                                    AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_THRESHOLD))
        self.custom_data_set_path = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_TRAINING_DATA_SET_PATH.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_TRAINING_DATA_SET_PATH))

        self.custom_knn_weights = configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_WEIGHTS.name,
                                              configs.get(
                                                  AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_WEIGHTS))

        self.threshold = configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_THRESHOLD.name,
                                     configs.get(
                                         AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_THRESHOLD))
        self.custom_classifier_location = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_CLASSIFIER_LOCATION.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_CLASSIFIER_LOCATION))

        self.current_status = ExecutionStatus(self.get_class_name(), "INIT", "INITIALIZATION", "COMPLETED")

        # # Initialize the model
        self.extractor = configs.get(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_EXTRACTOR.name,
            configs.get(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_EXTRACTOR))

        if self.extractor is None:
            self.model = models.densenet121(pretrained=True)
            self.extractor = "Densenet121"
        elif self.extractor == "GoogLeNet":
            # Load a pre-trained ResNet model
            self.model = models.googlenet(pretrained=True)
            self.model.eval()
        elif self.extractor == "GoogLeNetPlaces365":
            from dronebuddylib.atoms.placerecognition.resources.common.googlenet.googlenet_places365 import \
                GoogLeNetPlaces365
            self.model = GoogLeNetPlaces365()
            model_path = pkg_resources.resource_filename(__name__, "resources/common/googlenet/googlenet_places365.pth")
            # Load the pre-trained weights
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        elif self.extractor == "ResNet50":
            self.model = models.resnet50(pretrained=True)
        elif self.extractor == "Densenet121":
            self.model = models.densenet121(pretrained=True)
        elif self.extractor == "ResNet18":
            self.model = models.resnet18(pretrained=True)

        # # Put the model in evaluation mode
        self.model.eval()

        # Define a transform to convert images to the format expected by ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Define a transformation pipeline with rotation and color jitter
        self.transform = transforms.Compose([
            # Convert the NumPy array to PIL Image first to use torchvision transforms
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        super().__init__(engine_configurations)

    def preprocess_image(self, image) -> np.ndarray:

        # Resize the image
        image_resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Convert the image to a float tensor and normalize
        image_tensor = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
        image_tensor /= 255.0  # Normalize to [0, 1]
        image_tensor = (image_tensor - np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))) / np.array(
            [0.229, 0.224, 0.225]).reshape((3, 1, 1))

        # Add a batch dimension
        image_tensor = np.expand_dims(image_tensor, axis=0)
        return image_tensor

    def extract_features_with_specified_models(self, image) -> str:
        img_transformed = self.transform(image)

        # Add a batch dimension since PyTorch models expect a batch of images as input
        img_batch = img_transformed.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = self.model(img_batch)
            return output

    def test_memory(self) -> dict:
        """
                Tests the memory capabilities of the classifier by evaluating its performance
                on a predefined test dataset. This method serves as a diagnostic tool to ensure
                the classifier is functioning as expected.

                Returns:
                    dict: A dictionary containing key performance metrics such as accuracy,
                    precision, and F1 score of the classifier on the test dataset.
               """
        return self.test_classifier()

    def recognize_place(self, image) -> RecognizedPlaces:
        """
            Recognizes the place depicted in the given image. If the confidence of the
            prediction is below a given threshold, the place is classified as 'unknown'.
                Args:
                    image: The image of the place to recognize.

                Returns:
                    RecognizedPlaces: The recognized place.
                """
        # Extract features from the provided image
        # This requires a feature extraction method similar to the one discussed earlier

        image_features = self.feature_extractor(image)

        # image_features = self.feature_extractor_place(image)

        # Check if features were successfully extracted
        if image_features is None:
            logger.log_info(self.get_class_name(), "No features extracted from image.")
            return None

        if self.custom_classifier_location is not None:
            model_path = self.custom_classifier_location
        else:
            classifier_index = read_from_file(
                pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt"))
            model_path = pkg_resources.resource_filename(__name__,
                                                         self.classifier_path + str(classifier_index) + ".clf")

        # Load the trained KNN model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict the place using the KNN model
        # Note: reshape image_features to match the expected input format of the model

        # ==========================================================================================================
        # In the context of a Random Forest classifier,
        # the "confidence" of a prediction is usually derived from the proportion of trees that vote for the predicted class.
        # This is a measure of how certain the ensemble is about its prediction.
        #
        # How Confidence is Calculated in Random Forest
        # --------------------------------------------
        #
        # Voting Mechanism:
        #     In classification tasks, each tree in the Random Forest makes a prediction (vote) for the class of the input sample.
        #     The final prediction of the ensemble is the class that receives the majority of the votes.
        #
        # Confidence Score:
        #     The confidence of the prediction is typically calculated as the proportion of trees that predict the winning class.
        #     For example, if you have a Random Forest consisting of 100 trees,
        #     and 70 trees predict Class A while 30 trees predict Class B for a given sample,
        #     the Random Forest will classify the sample as Class A with a confidence score of 70%.
        #
        # Understanding Confidence
        # ----------------------------
        #
        # High Confidence
        #     indicates that a large majority of the trees in the forest agree on the classification,
        #     suggesting that the model is relatively certain about its decision.
        #
        # Lower Confidence
        #     suggests that the trees are more divided on the classification, indicating less certainty in the prediction.
        #     This could happen in areas of the feature space where the classes are not well-separated or in cases of noisy data.
        # ==========================================================================================================
        #

        predicted_place = model.predict([image_features])

        probabilities = model.predict_proba([image_features])[
            0]  # Get the probabilities of the first (and only) example
        max_probability = max(probabilities)
        predicted_index = probabilities.argmax()
        predicted_label = model.classes_[predicted_index]

        # You can also retrieve the probabilities or distances if your application needs that
        # For example, to get probabilities: probabilities = model.predict_proba([image_features])

        logger.log_info(self.get_class_name(), f"Recognized place: {predicted_place}")
        logger.log_info(self.get_class_name(),
                        f"Recognized place: {predicted_label} with confidence: {max_probability}")

        # Predict the class probabilities
        probabilities = model.predict_proba([image_features])[0]  # Probabilities of all classes
        top_three_indices = np.argsort(probabilities)[::-1][:3]  # Indices of top 3 classes
        top_three_probabilities = probabilities[top_three_indices]  # Probabilities of top 3 classes
        top_three_labels = model.classes_[top_three_indices]  # Labels of top 3 classes

        recognized_places = []
        most_probable_place = RecognizedPlaceObject(predicted_label, max_probability)
        # Log the top three predictions
        for i, (label, prob) in enumerate(zip(top_three_labels, top_three_probabilities)):
            logger.log_info(self.get_class_name(), f"Top {i + 1} predicted place: {label} with confidence: {prob}")
            recognized_places.append(RecognizedPlaceObject(label, prob))

        return RecognizedPlaces(most_probable_place, recognized_places)

    def remember_place(self, image=None, name=None) -> bool:

        logger.log_info(self.get_class_name(), 'Starting to remember place: ' + name)
        # Trigger start callback

        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "training data set creation",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create training data set for ' + name)
        self.create_data_set(name, 0, self.drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

        logger.log_info(self.get_class_name(), 'Starting to create validation data set for ' + name)
        self.create_data_set(name, 1, self.drone_instance)
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)

        return True

    # Function to update the progress bar
    # deper network than KNN

    def progress_bar(self, done_event, title="Training Progress"):
        # Initialize tqdm with an unknown total duration (using total=None)
        with tqdm(total=None, desc=title, unit=" step") as pbar:
            while not done_event.is_set():
                pbar.update()  # Update the progress bar by one step
                sleep(0.1)  # Sleep a short time before updating again
            pbar.update()  # Ensure the progress bar completes if it hasn't already

    def feature_extractor(self, img):
        """
           Extracts feature vectors from the given image using a pre-defined neural network model.
            This method is crucial for converting raw images into a form that can be effectively
            used by the KNN classifier for place recognition.

            Args:
                img: The image from which to extract features, expected to be in a format compatible
                     with the pre-processing transformations.

            Returns:
                numpy.ndarray: The extracted features as a flat array.
        """

        if self.extractor == "GoogLeNetPlaces365":
            img = self.preprocess_image(img)
            #     model: The pre-trained model to use for feature extraction.
            #     preprocess: The preprocessing steps to apply to the image before feature extraction.
            img_t = self.preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                features = self.model(batch_t)

            # Convert features to a numpy array
            return features.numpy().flatten()
        else:
            img_transformed = self.transform(img)

            # Add a batch dimension since PyTorch models expect a batch of images as input
            img_batch = img_transformed.unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = self.model(img_batch)
                return output.numpy().flatten()

    def feature_extractor_place(self, img):
        """
           Extracts feature vectors from the given image using a pre-defined neural network model.
            This method is crucial for converting raw images into a form that can be effectively
            used by the KNN classifier for place recognition.

            Args:
                img: The image from which to extract features, expected to be in a format compatible
                     with the pre-processing transformations.

            Returns:
                numpy.ndarray: The extracted features as a flat array.
        """

        img_transformed = self.transform(img)

        # Add a batch dimension since PyTorch models expect a batch of images as input
        img_batch = img_transformed.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = self.model(img_batch)
            return output.numpy().flatten()

    # Assume logger and other necessary functions and imports are defined elsewhere

    #
    def train_random_forest_classifier(self, train_dir, model_save_path=None, n_estimators=100,
                                       max_depth=None, random_state=42, criterion='gini',
                                       min_samples_split=2, classifier_index=0, changes=None,
                                       verbose=False, future=None):
        """
        Trains the Random Forest classifier on a set of labeled images stored in a directory structure.
        Each sub-directory within the train directory should represent a class (place), and contain
        images corresponding to that place.

        Args:
            train_dir (str): The directory containing the training dataset.
            model_save_path (str, optional): Path to save the trained model.
            n_estimators (int, optional): The number of trees in the forest.
            max_depth (int, optional): The maximum depth of the trees.
            random_state (int, optional): Controls both the randomness of the bootstrapping of the samples
                                          used when building trees (if bootstrap=True) and the sampling of
                                          the features to consider when looking for the best split at each node.
            criterion (str, optional): The function to measure the quality of a split.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node.
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

        training_start_time = time.time()

        # Loop through each class in the training set
        for class_dir in os.listdir(train_dir):
            class_count += 1
            if verbose:
                logger.log_debug("Training for class: " + class_dir)
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current class
            for img_path in [os.path.join(train_dir, class_dir, f) for f in
                             os.listdir(os.path.join(train_dir, class_dir)) if
                             os.path.isfile(os.path.join(train_dir, class_dir, f))]:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                features = self.feature_extractor(img)  # Assuming this is defined elsewhere
                if features is not None and features.size > 0:
                    X.append(features)
                    y.append(class_dir)
                    if verbose:
                        logger.log_debug("Feature extraction for: " + img_path)
                else:
                    logger.log_error("No features extracted for: " + img_path)

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            logger.log_error("No features were extracted. Aborting training.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Create and train the Random Forest classifier
        rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
                                        criterion=criterion, min_samples_split=min_samples_split)
        rf_clf.fit(X_train, y_train)

        logger.log_debug(self.get_class_name(), "Training completed")

        # Evaluate the model
        predictions = rf_clf.predict(X_test)

        logger.log_debug(self.get_class_name(), "Testing completed")

        # Calculate and print the accuracy
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        f1 = precision_recall_fscore_support(y_test, predictions, average='weighted')
        report = classification_report(y_test, predictions)

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1,
            "report": report
        }

        # Save the trained Random Forest classifier
        training_end_time = time.time()

        # Save the trained KNN classifier
        training_end_time = time.time()

        model_string = "model name: " + str(
            classifier_index) + ".clf" + ": reason : " + changes + " : extractor : " + self.extractor + " : no. of classes " + str(
            class_count) + ": time taken : " + str(
            (training_end_time - training_start_time)) + " seconds  :  accuracy: " + str(
            accuracy) + " : precision: " + str(precision) + " : f1_score: " + str(f1) + "\n"
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(rf_clf, f)
        # write the model name with teh accuracy into the classifier_data.txt

        file_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_data.txt")

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

    def create_memory(self, changes=None):

        file_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt")
        classifier_index_read = read_from_file(file_path)
        classifier_index = int(classifier_index_read) + 1
        overwrite_file(file_path, str(classifier_index))

        if self.custom_data_set_path is not None:
            training_path_name = self.custom_data_set_path
        else:
            training_path_name = pkg_resources.resource_filename(__name__, "resources/test_data/training_data")

        # classifier_path_name = pkg_resources.resource_filename(__name__,
        #                                                        "resources/models/classifiers/"
        #                                                        + "trained_place_knn_model"
        #                                                        + str(classifier_index) + ".clf")
        classifier_path_name = pkg_resources.resource_filename(__name__,
                                                               self.classifier_path + str(classifier_index) + ".clf")

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
        # train_thread = threading.Thread(target=self.train, args=(training_path_name,
        #                                                          classifier_path_name, self.custom_knn_neighbors,
        #                                                          self.custom_knn_algorithm_name,
        #                                                          self.custom_knn_weights,
        #                                                          classifier_index, changes,
        #                                                          False, future))
        #
        # (self, train_dir, model_save_path=None, n_estimators=100,
        #                                        max_depth=None, random_state=42, criterion='gini',
        #                                        min_samples_split=2,
        #                                        verbose=False, future=None):
        #         """

        # # Start the training in a separate thread
        train_thread = threading.Thread(target=self.train_random_forest_classifier, args=(training_path_name,
                                                                                          classifier_path_name,
                                                                                          100, None, 42, 'gini', 2,
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

    def test_classifier(self, future=None):
        test_start_time = time.time()

        file_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt")

        classifier_index_read = read_from_file(file_path)
        classifier_index = int(classifier_index_read)
        classifier_path_name = pkg_resources.resource_filename(__name__,
                                                               "resources/models/classifiers/" + "trained_place_knn_model"
                                                               + str(classifier_index) + ".clf")
        testing_path_name = pkg_resources.resource_filename(__name__, "resources/test_data/testing_data")
        # Load the trained classifier
        classifier = joblib.load(classifier_path_name)

        # Load your test data
        features, true_labels, label_names = self.load_test_data(testing_path_name)

        # Predict using the classifier
        predicted_labels = classifier.predict(features)

        # Calculate and print the accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        f1_score = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        report = classification_report(true_labels, predicted_labels, target_names=label_names)

        logger.log_info(self.get_class_name(), f"Accuracy: {accuracy * 100:.2f}%")
        result = {
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1_score,
            "report": report
        }
        self.progress_event.set()
        if future is not None:
            future.set_result(result)  # Set the result of the future
        test_end_time = time.time()
        logger.log_info(self.get_class_name(),
                        'Testing complete. It took ' + str((test_end_time - test_start_time) / 60) + ' minutes')
        return result

    def preprocess_image(self, img):
        # # Load the image with OpenCV
        # img = cv2.imread(image_path)
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to PIL format to use the same preprocessing as before
        # Alternatively, you can adjust the preprocessing pipeline to work directly with the OpenCV format
        img = Image.fromarray(img)
        return img

    def load_test_data(self, test_dir):
        """
           Load images and true labels from the test directory.
           """
        features = []
        labels = []
        label_names = []

        # Iterate over each directory in test_dir
        for label_name in os.listdir(test_dir):
            label_dir = os.path.join(test_dir, label_name)
            if not os.path.isdir(label_dir):
                continue  # Skip non-directory files

            # Store the name of the label
            label_names.append(label_name)

            # Iterate over each image in the directory
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)

                # Extract features from the image (i.e., face encodings)
                img = cv2.imread(image_path)
                face_encoding = self.feature_extractor(img)
                if face_encoding is not None:
                    features.append(face_encoding)
                    labels.append(label_name)

        # Convert labels to a numerical format
        le = preprocessing.LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        return features, labels, le.classes_

    def create_data_set(self, place_name, data_mode, drone_instance=None):
        """
                Generates a dataset for a given place name, optionally using a drone for image collection.
                Different modes support training, validation, and testing data collection.

                Args:
                    place_name (str): The name of the place for which to create the dataset.
                    data_mode (int): The mode of dataset creation (e.g., training, validation).
                    drone_instance: An optional drone instance to use for collecting images.

                """
        type = "training"
        # if drone_instance is not None:
        if True:
            count = 00

            if data_mode == 0:
                path_name = pkg_resources.resource_filename(__name__,
                                                            "resources/test_data/training_data/" + place_name)
                rotation_size = 5
                height_fixer = 20

            elif data_mode == 1:
                drone_instance.move_forward(30)
                path_name = pkg_resources.resource_filename(__name__,
                                                            "resources/test_data/validation_data/" + place_name)
                rotation_size = 30
                height_fixer = 30
                type = "validation"
            else:
                path_name = pkg_resources.resource_filename(__name__,
                                                            "resources/test_data/training_data/" + place_name)
                rotation_size = 5
                height_fixer = 20

            logger.log_warning(self.get_class_name(),
                               'Data set creating. ' + str(320) +
                               ' data collected. Data set can be found at the path: ' + path_name)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
                # first check if the drone is in the air
                # if not, take off

            # first check if the drone is in the air
            # if not, take off
            #  then rotate the drone 360 degrees and take pictures every 5 degrees
            # after completing the rotation  go up 30 cm and repeat the process
            #  according to singapore building standards the height of a floor is 3.2 meters
            #  so we can take a set of pictures every 50 cm
            #  and repeat the process until we reach 1.5 meters, just to be safe
            #  overall for every place we will have 5*72= 360 pictures, which will be used to create the model

            current_height = 30
            while current_height <= 150:
                if data_mode == 0:
                    drone_instance.move_up(height_fixer)
                    logger.log_debug(self.get_class_name(),
                                     "going  up by " + str(height_fixer) + " , " + str(
                                         current_height) + " cm above the ground")
                else:
                    drone_instance.move_down(height_fixer)

                    logger.log_debug(self.get_class_name(),
                                     "going  down by   " + str(height_fixer) + " , " + str(
                                         current_height) + " cm above the ground")

                current_rotation = 0
                while current_rotation <= 360:
                    frame = drone_instance.get_frame_read().frame
                    face = cv2.resize(frame, (500, 500))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    # Save file in specified directory with unique name
                    file_name_path = (path_name + "\\" + place_name + "_" + str(count) + ".jpg")
                    cv2.imwrite(file_name_path, face)
                    # Put count on images and display live count
                    cv2.putText(face, str(count) + " " + type, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("FaceCropper", face)
                    logger.log_debug(self.get_class_name(),
                                     "turning by " + str(rotation_size) + " , " + str(current_rotation))
                    try:
                        drone_instance.rotate_clockwise(rotation_size)
                    except Exception as e:
                        logger.log_error(self.get_class_name(), "Error while rotating drone: " + str(e))
                    current_rotation += rotation_size
                    self.current_status = ExecutionStatus(self.get_class_name(), "remember_place",
                                                          "creating data set at " + str(
                                                              current_height) + " cm above the ground",
                                                          "PROGRESS", str(count) + "/" + str(320))
                    if data_mode == 1:
                        if drone_instance.get_height() <= 30:
                            drone_instance.land()
                    count += 1
                    if cv2.waitKey(1) == 27:
                        break

                current_height += 30

                if cv2.waitKey(1) == 27:
                    break

            cv2.destroyAllWindows()
            logger.log_warning(self.get_class_name(),
                               'Data set created successfully. ' + str(320) +
                               ' data collected. Data set can be found at the path: ' + path_name)

    def get_current_status(self):
        return self.current_status

    def get_required_params(self) -> list:
        """
        Get the required parameters for the FaceRecognitionImpl class.

        Returns:
            list: A list of required parameters.
        """
        pass

    def get_optional_params(self) -> list:
        """
        Get the optional parameters for the FaceRecognitionImpl class.

        Returns:
            list: A list of optional parameters.
        """
        return [AtomicEngineConfigurations.PLACE_RECOGNITION_RF_TRAINING_DATA_SET_PATH,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_TESTING_DATA_SET_PATH,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_VALIDATION_DATA_SET_PATH,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NAME,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_ALGORITHM_NEIGHBORS,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_SAVING_PATH,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_MODEL_THRESHOLD,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_USE_DRONE_TO_CREATE_DATASET,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_DRONE_INSTANCE,
                AtomicEngineConfigurations.PLACE_RECOGNITION_RF_CLASSIFIER_LOCATION,
                ]
        # return []

    def get_class_name(self) -> str:
        """
        Get the class name of the FaceRecognitionImpl class.

        Returns:
            str: The class name.
        """
        return 'PLACE_RECOGNITION_RF'

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name of the FaceRecognitionImpl class.

        Returns:
            str: The algorithm name.
        """
        return 'Place Recognition through Random Forest'
