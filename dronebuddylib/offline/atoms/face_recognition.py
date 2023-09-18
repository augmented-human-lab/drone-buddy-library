import cv2
import face_recognition
import numpy as np
import pkg_resources

from dronebuddylib.utils.exceptions import FileWritingException
from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()

'''
in order to get the trained json file follow the https://snips-nlu.readthedocs.io/en/latest/tutorial.html
'''


def process_frame_for_recognition(frame):
    """
     Preprocesses a video frame for faster face recognition processing.

     Args:
         frame (numpy.ndarray): The input frame to be processed.

     Returns:
         numpy.ndarray: The processed frame ready for face recognition.
     """

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    logger.debug("Face Recognition : shape of the frame : ", rgb_small_frame.shape)
    return rgb_small_frame


def load_known_face_names():
    """
      Loads a list of known face names from a text file.

      Returns:
          list: List of known face names.
      """
    # load the user list from known_faces.txt
    path = pkg_resources.resource_filename(__name__, "resources/facerecognition/known_names.txt")
    known_face_names = read_file_into_list(path)
    return known_face_names


def load_known_face_encodings(known_face_names):
    """
       Loads known face encodings corresponding to the provided face names.

       Args:
           known_face_names (list): List of known face names.

       Returns:
           list: List of known face encodings.
       """
    known_face_encodings = []
    for name in known_face_names:
        face_path = pkg_resources.resource_filename(__name__, "resources/facerecognition/images/" + name + ".jpg")
        face_image = face_recognition.load_image_file(face_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings


def read_file_into_list(filename):
    """
     Reads lines from a file and returns them as a list.

     Args:
         filename (str): Path to the file to be read.

     Returns:
         list: List of lines read from the file.
     Raises:
         FileNotFoundError: If the specified file is not found.
     """
    field_list = []
    # Open the file in read mode
    try:
        with open(filename, "r") as file:
            # Read the contents of the file line by line
            lines = file.readlines()
            lines_without_newline = [line.rstrip('\n') for line in lines]
            return [line for line in lines_without_newline if line]

    except FileNotFoundError as e:
        raise FileNotFoundError("The specified file is not found.", e) from e
    # Return the list of fields


def get_video_feed(frame, face_locations, face_names):
    """
      Displays the video feed with annotated face boxes and names.

      Args:
          frame (numpy.ndarray): The frame to display.
          face_locations (list): List of face locations.
          face_names (list): List of recognized face names.

      Returns:
          None
      """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)


def find_all_the_faces(frame, show_feed=False):
    """
        Detects and recognizes faces in a given video frame.

        Args:
            frame (numpy.ndarray): The input frame to process.
            show_feed (bool): Whether to display the annotated video feed.

        Returns:
            list: List of recognized face names.
        """
    processed_frame = process_frame_for_recognition(frame)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(processed_frame)
    face_encodings = face_recognition.face_encodings(processed_frame, face_locations)

    # load the user list from memory
    face_names = load_known_face_names()
    # load the encodings
    known_face_encodings = load_known_face_encodings(face_names)

    recognized_faces = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
        #    name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = face_names[best_match_index]

        recognized_faces.append(name)

    if show_feed:
        get_video_feed(frame, face_locations, recognized_faces)

    return recognized_faces


def add_people_to_memory(name, image_path):
    """
      Adds a new person's name and image to the recognition memory.

      Args:
          name (str): Name of the new person.
          image_path (str): Path to the image of the new person.

      Returns:
          bool: True if the addition was successful, False otherwise.
      """

    # add name to the known_names.txt
    try:
        text_file_path = pkg_resources.resource_filename(__name__, "resources/facerecognition/known_names.txt")
        with open(text_file_path, 'a') as file:
            file.write(name + '\n')
    except IOError:
        logger.error("Error while writing to the file : ", name)
        raise FileWritingException("Error while writing to the file : ", name)
    # add file to the images folder

    try:
        new_file_name = pkg_resources.resource_filename(__name__, "resources/facerecognition/images/" + name + ".jpg")
        loaded_image = cv2.imread(image_path)
        cv2.imwrite(new_file_name, loaded_image)

    except IOError:
        raise FileWritingException("Error while writing to the file : ", new_file_name)
    return True
