"""
This is uses the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""
import threading
import time
from asyncio import Future

import pkg_resources

from dronebuddylib.atoms.facerecognition import IFaceRecognition
from dronebuddylib.atoms.facerecognition.face_recognition_result import RecognizedFaceObject, RecognizedFaces
from dronebuddylib.exceptions.FaceRecognitionException import FaceRecognitionException
from dronebuddylib.models import AtomicEngineConfigurations, EngineConfigurations
from dronebuddylib.models.execution_status import ExecutionStatus

from dronebuddylib.utils.logger import Logger

import math

import cv2
from sklearn import neighbors, preprocessing
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support, classification_report
import joblib

from dronebuddylib.utils.utils import overwrite_file, read_from_file, config_validity_check
from time import sleep
from tqdm import tqdm

logger = Logger()


class FaceRecognitionKNNImpl(IFaceRecognition):
    use_drone = False
    # Event to signal when the training is done
    progress_event = threading.Event()
    current_status = None

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the FaceRecognitionImpl class.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # Get configurations
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        self.configs = engine_configurations

        self.use_drone = bool(configs.get(AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET.name,
                                          configs.get(
                                              AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET)))
        self.current_status = ExecutionStatus(self.get_class_name(), "INIT", "INITIALIZATION", "COMPLETED")
        super().__init__(engine_configurations)

    def test_memory(self) -> dict:
        return self.test_classifier()

    def recognize_face(self, image) -> RecognizedFaces:
        # Find face locations and encodings in the image
        X_face_locations = face_recognition.face_locations(image)
        faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)
        distance_threshold = 0.5
        # Ensure at least one face is found
        if len(faces_encodings) == 0:
            return None
        classifier_index = read_from_file(
            pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt"))

        model_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/trained_knn_model" + str(
            classifier_index) + ".clf")
        # Ensure at least one face is found
        if len(faces_encodings) == 0:
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Use the KNN model to find the best matches for the test faces
        closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                       zip(model.predict(faces_encodings), X_face_locations, are_matches)]
        result = []
        for name, (top, right, bottom, left) in predictions:
            result.append(RecognizedFaceObject(name, [top, right, bottom, left]))
            logger.log_info(self.get_class_name(), "- Found {} at ({}, {})".format(name, left, top))
        return RecognizedFaces(result)

    def remember_face(self, image=None, name=None, drone_instance=None, on_start=None, on_training_set_complete=None,
                      on_testing_set_complete=None, on_validation_set_complete=None, on_face_not_found=None) -> bool:

        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        logger.log_info(self.get_class_name(), 'Place your face in front of the camera to create a data set.')
        logger.log_info(self.get_class_name(), 'If you see the message "Face not found, move a bit.", move a bit.')
        # Trigger start callback
        if on_start:
            on_start()
        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "training data set creation",
                                              "START")

        for i in range(5):
            logger.log_info(self.get_class_name(), 'Starting in ' + str(3 - i) + ' seconds...')
            time.sleep(1)

        logger.log_info(self.get_class_name(), 'Starting to create training data set for ' + name)
        self.create_data_set(face_classifier, name, 0, 900, drone_instance, on_face_not_found, )
        logger.log_info(self.get_class_name(), 'Successfully created training data set for ' + name)
        logger.log_info(self.get_class_name(), 'Get ready for the test set creation ' + name)

        if on_training_set_complete:
            on_training_set_complete()

        for i in range(5):
            logger.log_info(self.get_class_name(), 'Starting test set creation in ' + str(6 - i) + ' seconds...')
            time.sleep(1)
        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "testing data set creation",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create testing data set for ' + name)
        self.create_data_set(face_classifier, name, 1, 100, drone_instance, on_face_not_found)
        logger.log_info(self.get_class_name(), 'Successfully created testing data set for ' + name)

        if on_testing_set_complete:
            on_testing_set_complete()

        logger.log_info(self.get_class_name(), 'Get ready for the validation set creation ' + name)

        for i in range(5):
            logger.log_info(self.get_class_name(), 'Starting validation set creation in ' + str(6 - i) + ' seconds...')
            time.sleep(1)

        logger.log_info(self.get_class_name(), 'Starting to create validation data set for ' + name)

        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "validation data set creation",
                                              "START")

        self.create_data_set(face_classifier, name, 2, 50, drone_instance, on_face_not_found)
        logger.log_info(self.get_class_name(), 'Successfully created validation data set for ' + name)

        if on_validation_set_complete:
            on_validation_set_complete()

        return True

    # Function to update the progress bar
    def progress_bar(self, done_event, title="Training Progress"):
        # Initialize tqdm with an unknown total duration (using total=None)
        with tqdm(total=None, desc=title, unit=" step") as pbar:
            while not done_event.is_set():
                pbar.update()  # Update the progress bar by one step
                sleep(0.1)  # Sleep a short time before updating again
            pbar.update()  # Ensure the progress bar completes if it hasn't already

    def train(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param train_dir: directory that contains a sub-directory for each known person, with its name.

         (View in source code to see train_dir example tree structure)

         Structure:
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...

        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """
        X = []
        y = []

        training_start_time = time.time()

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if verbose:
                logger.log_debug(self.get_class_name(), "Training for class : " + class_dir)
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        logger.log_debug(self.get_class_name(),
                                         "Image {} not suitable for training: {}"
                                         .format(img_path,
                                                 "Didn't find a face"
                                                 if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                logger.log_debug(self.get_class_name(), ("Chose n_neighbors automatically:" + str(n_neighbors)))
        logger.log_info(self.get_class_name(), 'Start training KNN classifier')
        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        logger.log_info(self.get_class_name(), 'Fitting the classifier')
        knn_clf.fit(X, y)
        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        training_end_time = time.time()
        logger.log_info(self.get_class_name(),
                        'Training complete. It took ' + str(
                            (training_end_time - training_start_time) / 60) + ' minutes')
        self.progress_event.set()
        return knn_clf

    def create_memory(self, on_start=None, on_training_complete=None):

        file_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt")
        classifier_index_read = read_from_file(file_path)
        classifier_index = int(classifier_index_read) + 1
        overwrite_file(file_path, str(classifier_index))

        classifier_path_name = pkg_resources.resource_filename(__name__,
                                                               "resources/models/classifiers/" + "trained_knn_model"
                                                               + str(classifier_index) + ".clf")
        training_path_name = pkg_resources.resource_filename(__name__, "resources/test_data/training_data")

        # ------------------------------------------------------------------------------------------------------------
        #  TRAINING THE CLASSIFIER
        # ------------------------------------------------------------------------------------------------------------
        # Trigger start callback
        if on_start:
            on_start()

        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "training KNN classifier",
                                              "START")

        logger.log_info(self.get_class_name(), 'Starting to create memory. Training KNN classifier')

        # Start the training in a separate thread
        train_thread = threading.Thread(target=self.train, args=(training_path_name,
                                                                 classifier_path_name, 2, "ball_tree",
                                                                 True,))
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
        if on_training_complete:
            on_training_complete()
            # ------------------------------------------------------------------------------------------------------------
        #  TESTING THE CLASSIFIER
        # ------------------------------------------------------------------------------------------------------------

        logger.log_debug(self.get_class_name(), 'Starting to test the classifier')
        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "testing KNN classifier",
                                              "START")

        # Create a Future object
        future = Future()

        test_progress_bar_thread = threading.Thread(target=self.progress_bar,
                                                    args=(self.progress_event, "Testing Progress"))
        test_progress_bar_thread.start()

        testing_thread = threading.Thread(target=self.test_classifier, args=(future,))
        testing_thread.start()

        testing_thread.join()
        test_progress_bar_thread.join()
        self.current_status = ExecutionStatus(self.get_class_name(), "create_memory",
                                              "testing KNN classifier",
                                              "COMPLETED")
        # Retrieve the result from the Future
        try:
            output = future.result()  # This will also re-raise any exception set by set_exception
            logger.log_info(self.get_class_name(), 'Successfully tested the classifier')
            return output
        except Exception as e:
            print(f"Error occurred: {e}")
            return FaceRecognitionException("Error occurred while testing the classifier", 500, e)

    def test_classifier(self, future=None):
        test_start_time = time.time()

        file_path = pkg_resources.resource_filename(__name__, "resources/models/classifiers/classifier_index.txt")

        classifier_index_read = read_from_file(file_path)
        classifier_index = int(classifier_index_read)
        classifier_path_name = pkg_resources.resource_filename(__name__,
                                                               "resources/models/classifiers/" + "trained_knn_model"
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

    def extract_features(self, image_path):
        """
           Extract the face encodings from the given image path.
           """
        # Load the image file
        image = face_recognition.load_image_file(image_path)

        # Find face locations in the image
        face_locations = face_recognition.face_locations(image)

        # If no faces are found, return None
        if len(face_locations) == 0:
            return None

        # Assuming we'll use only the first face found in the image
        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

        return face_encoding

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
                face_encoding = self.extract_features(image_path)
                if face_encoding is not None:
                    features.append(face_encoding)
                    labels.append(label_name)

        # Convert labels to a numerical format
        le = preprocessing.LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        return features, labels, le.classes_

    # Load functions
    def face_extractor(self, img, face_classifier=None):
        # Load HAAR face classifier
        # Function detects faces and returns the cropped face
        # If no face detected, it returns the input image

        global cropped_face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None

        # Crop all faces found
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]

        return cropped_face

    def create_data_set(self, face_classifier, participant_name, data_mode, data_set_size=900, drone_instance=None,
                        on_face_not_found=None):

        global cap
        if self.use_drone is False:
            cap = cv2.VideoCapture(0)
        count = 00

        if data_mode == 0:
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/test_data/training_data/" + participant_name)
        elif data_mode == 1:
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/test_data/testing_data/" + participant_name)
        elif data_mode == 2:
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/test_data/validation_data/" + participant_name)
        else:
            path_name = pkg_resources.resource_filename(__name__,
                                                        "resources/test_data/training_data/" + participant_name)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        # Collect 100 samples of your face from webcam input
        while True:
            if self.use_drone:
                frame = drone_instance.get_frame_read().frame
            else:
                ret, frame = cap.read()

            if self.face_extractor(frame, face_classifier=face_classifier) is not None:
                count += 1
                face = cv2.resize(self.face_extractor(frame, face_classifier=face_classifier), (300, 300))

                # Save file in specified directory with unique name
                file_name_path = (path_name + "\\" + participant_name + "_" + str(count) + ".jpg")
                cv2.imwrite(file_name_path, face)
                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("FaceCropper", face)

            else:
                logger.log_warning(self.get_class_name(), 'Face not found, move a bit.')
                if on_face_not_found:
                    on_face_not_found()
                pass
            self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "creating data set",
                                                  "PROGRESS", str(count) + "/" + str(data_set_size))

            if cv2.waitKey(1) == 27 or count == data_set_size:  # 27 is the Esc Key
                break

        if self.use_drone is False:
            cap.release()
        cv2.destroyAllWindows()
        logger.log_warning(self.get_class_name(),
                           'Data set created successfully. ' + str(data_set_size) +
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
        return [AtomicEngineConfigurations.FACE_RECOGNITION_KNN_ALGORITHM_NEIGHBORS,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_ALGORITHM_NAME,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_MODEL_SAVING_PATH,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_TESTING_DATA_SET_SIZE,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_TESTING_DATA_SET_SIZE,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_VALIDATION_DATA_SET_SIZE,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                AtomicEngineConfigurations.FACE_RECOGNITION_KNN_DRONE_INSTANCE,
                ]

    def get_class_name(self) -> str:
        """
        Get the class name of the FaceRecognitionImpl class.

        Returns:
            str: The class name.
        """
        return 'FACE_RECOGNITION_KNN'

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name of the FaceRecognitionImpl class.

        Returns:
            str: The algorithm name.
        """
        return 'Face Recognition'
