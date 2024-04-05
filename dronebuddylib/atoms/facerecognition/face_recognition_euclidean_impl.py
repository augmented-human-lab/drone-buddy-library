import cv2
import numpy as np
import pkg_resources

from dronebuddylib.atoms.facerecognition.face_recognition_result import RecognizedFaces, RecognizedFaceObject
from dronebuddylib.atoms.facerecognition.i_face_recognition import IFaceRecognition
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.execution_status import ExecutionStatus
from dronebuddylib.utils import FileWritingException
from dronebuddylib.utils.logger import Logger

logger = Logger()

import face_recognition


class FaceRecognitionEuclideanImpl(IFaceRecognition):
    """
    Implementation of the IFaceRecognition interface using face_recognition library.
    The recognition is carried on using the Euclidean distance between the face encodings.
    """
    current_status = None

    def test_memory(self) -> dict:
        pass

    KNOWN_NAMES_FILE_PATH = pkg_resources.resource_filename(__name__, "resources/known_names.txt")
    IMAGE_PATH = "resources/images/"

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the FaceRecognitionImpl class.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        self.current_status = ExecutionStatus(self.get_class_name(), "INIT", "INITIALIZATION", "COMPLETED")

        super().__init__(engine_configurations)

    def recognize_face(self, image) -> RecognizedFaces:
        """
        Recognize faces in an image.

        Args:
            image: The image containing faces to be recognized.

        Returns:
            list: A list of recognized faces.
        """
        self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "initializing", "STARTED")

        processed_frame = self.process_frame_for_recognition(image)
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(processed_frame)
        face_encodings = face_recognition.face_encodings(processed_frame, face_locations)

        # load the user list from memory
        face_names = self.load_known_face_names()
        # load the encodings
        known_face_encodings = self.load_known_face_encodings(face_names)
        self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "load_known_face_encodings",
                                              "COMPLETED")

        recognized_faces = []
        self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "matching_face_encodings",
                                              "STARTED")

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = face_names[best_match_index]

            recognized_faces.append(RecognizedFaceObject(name, face_encoding))

        return RecognizedFaces(recognized_faces)

    def process_frame_for_recognition(self, frame):
        """
        Pre-process the frame for face recognition.

        Args:
            frame: The frame to be processed.

        Returns:
            The processed frame.
        """
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        logger.log_debug(self.get_class_name(),
                         "Face Recognition : shape of the frame : " + rgb_small_frame.shape.__str__())

        self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "process_frame_for_recognition",
                                              "COMPLETED")

        return rgb_small_frame

    def load_known_face_names(self):
        """
        Load the known face names from a file.

        Returns:
            list: A list of known face names.
        """
        path = self.KNOWN_NAMES_FILE_PATH
        known_face_names = self.read_file_into_list(path)
        self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "load_known_face_names",
                                              "COMPLETED")
        return known_face_names

    def load_known_face_encodings(self, known_face_names):
        """
        Load the known face encodings from files.

        Args:
            known_face_names: A list of known face names.

        Returns:
            list: A list of known face encodings.
        """
        known_face_encodings = []
        for name in known_face_names:
            face_path = pkg_resources.resource_filename(__name__, self.IMAGE_PATH + name + ".jpg")
            face_image = face_recognition.load_image_file(face_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)

            self.current_status = ExecutionStatus(self.get_class_name(), "recognize_face", "load_known_face_encodings",
                                                  "COMPLETED")
        return known_face_encodings

    def read_file_into_list(self, filename):
        """
        Read a file into a list.

        Args:
            filename: The path to the file.

        Returns:
            list: A list of lines from the file.
        """
        try:
            with open(filename, "r") as file:
                lines = file.readlines()
                lines_without_newline = [line.rstrip('\n') for line in lines]
                return [line for line in lines_without_newline if line]

        except FileNotFoundError as e:
            raise FileNotFoundError("The specified file is not found.", e) from e

    def remember_face(self, image_path=None, name=None) -> bool:
        """
        Associate a name with a face in an image.

        Args:
            image_path: The path to the image containing the face.
            name: The name to be associated with the face.

        Returns:
            bool: True if the association was successful, False otherwise.
        """

        self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "remember_face",
                                              "STARTED")
        try:
            text_file_path = self.KNOWN_NAMES_FILE_PATH
            with open(text_file_path, 'a') as file:
                file.write(name + '\n')
        except IOError:
            logger.log_error(self.get_class_name(), "Error while writing to the file : " + name)
            raise FileWritingException("Error while writing to the file : " + name)

        try:
            new_file_name = pkg_resources.resource_filename(__name__,
                                                            self.IMAGE_PATH + name + ".jpg")
            loaded_image = cv2.imread(image_path)
            cv2.imwrite(new_file_name, loaded_image)
            self.current_status = ExecutionStatus(self.get_class_name(), "remember_face", "remember_face",
                                                  "COMPLETED")
        except IOError:
            raise FileWritingException("Error while writing to the file : ", new_file_name)
        return True

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
        pass

    def get_class_name(self) -> str:
        """
        Get the class name of the FaceRecognitionImpl class.

        Returns:
            str: The class name.
        """
        return 'FACE_RECOGNITION_EUCLIDEAN'

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name of the FaceRecognitionImpl class.

        Returns:
            str: The algorithm name.
        """
        return 'Face Recognition'

    def get_current_status(self) -> dict:
        """
        Get the current status of the FaceRecognitionImpl class.

        Returns:
            dict: The current status.
        """
        pass
