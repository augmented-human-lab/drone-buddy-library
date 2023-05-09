import cv2
import numpy as np
import pkg_resources
import face_recognition

from dronebuddylib.utils.exceptions import FileWritingException
from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()


def process_frame_for_recognition(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    logger.debug("Face Recognition : shape of the frame : ", rgb_small_frame.shape)
    return rgb_small_frame


def load_known_face_names():
    # load the user list from known_faces.txt
    path = pkg_resources.resource_filename(__name__, "resources/facerecognition/know_names.txt")
    known_face_names = read_file_into_list(path)
    return known_face_names


def load_known_face_encodings(known_face_names):
    known_face_encodings = []
    for name in known_face_names:
        face_path = pkg_resources.resource_filename(__name__, "resources/facerecognition/images/" + name + ".jpg")
        face_image = face_recognition.load_image_file(face_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings


def read_file_into_list(filename):
    field_list = []
    # Open the file in read mode
    try:
        with open(filename, "r") as file:
            # Read the contents of the file line by line
            for line in file:
                # Split the line into fields based on a delimiter (e.g., comma)
                if line.strip() and len(line.strip().split(",")) != 0:
                    field_list.append(line.strip().split(","))
    except FileNotFoundError as e:
        raise FileNotFoundError("The specified file is not found.", e) from e
    # Return the list of fields
    return field_list


def find_all_the_faces(frame):
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

    return recognized_faces


def add_people_to_memory(file_name, folder_path, name, image):
    # add name to the known_names.txt
    try:
        with open(file_name, 'a') as file:
            file.write(name + ",")
    except IOError:
        logger.error("Error while writing to the file : ", file_name)
        raise FileWritingException("Error while writing to the file : ", file_name)
    # add file to the images folder

    try:
        with open(image, 'rb') as f:
            file_data = f.read()
        new_file_name = pkg_resources.resource_filename(__name__, "resources/facerecognition/images/" + name + ".jpg")

        with open(new_file_name, 'wb') as f:
            f.write(file_data)
    except IOError:
        raise FileWritingException("Error while writing to the file : ", new_file_name)
    return True