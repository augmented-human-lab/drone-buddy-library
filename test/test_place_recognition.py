import os
import re
import threading
import time
import unittest
from collections import Counter

import cv2
import pkg_resources
import pyttsx3

from djitellopy import Tello

from dronebuddylib.atoms.placerecognition import PlaceRecognitionKNNImpl
from dronebuddylib.atoms.placerecognition.place_recognition_engine import PlaceRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations, logger
from dronebuddylib.models.enums import FaceRecognitionAlgorithm, AtomicEngineConfigurations, PlaceRecognitionAlgorithm

from collections import defaultdict

from dronebuddylib.utils.utils import write_to_file_longer

voice_gen_engine = pyttsx3.init()


class TestFaceRecognition(unittest.TestCase):
    is_drone_in_air = False

    def test_face_rec_knn_add_place(self):

        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        tello.takeoff()
        self.is_drone_in_air = True
        print("battery: ", tello.get_battery())
        print("temperature: ", tello.get_temperature())
        thread1 = threading.Thread(target=self.add_place, args=("com3_meeting_room", tello,))
        thread3 = threading.Thread(target=self.keep_drone_in_air, args=(tello,))

        # Start threads
        thread3.start()
        thread1.start()

        thread1.join()
        thread3.join()

    def test_face_rec_knn_recognize_place_by_video(self):
        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        tello.takeoff()
        self.is_drone_in_air = True
        print("battery: ", tello.get_battery())
        print("temperature: ", tello.get_temperature())

        # self.test_determine_place_on_video(tello)
        thread3 = threading.Thread(target=self.keep_drone_in_air, args=(tello,))
        thread_recognize = threading.Thread(target=self.test_determine_place_on_video, args=(tello,))

        # Start threads
        thread3.start()
        thread_recognize.start()

        thread3.join()
        thread_recognize.join()

    def keep_drone_in_air(self, drone_instance):
        moving_dir = -1
        while True:
            logger.log_info("Executing functions: ", "Drone is in the air")

            if drone_instance is not None and self.is_drone_in_air:
                print("battery: ", drone_instance.get_battery())
                print("temperature: ", drone_instance.get_temperature())
                if drone_instance.get_battery() < 20:
                    drone_instance.land()
                    logger.log_info("Executing functions: ", "Low battery, landing")

                if drone_instance.get_temperature() > 90:
                    drone_instance.land()
                    logger.log_info("Executing functions: ", "High temperature, landing")
                drone_instance.send_rc_control(0, 0, moving_dir, 0)
                moving_dir = moving_dir * -1

            # break
            time.sleep(4)  # Sleep to simulate work and prevent a tight loop

    def add_place(self, name, tello):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_DRONE_INSTANCE,
                                         tello)
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         'True')
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.remember_place(None, name, tello)
        print(result)
        tello.land()

    def test_face_rec_knn_create_memory_KNN(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         "True")
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
                                         "GoogLeNetPlaces365")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.create_memory("New KNN with GoogLeNetPlaces365")
        print(result)

    def test_face_rec_knn_create_memory_RF(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_USE_DRONE_TO_CREATE_DATASET,
                                         "True")
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_RF_EXTRACTOR,
                                         "GoogLeNetPlaces365")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_RF, engine_configs)
        result = engine.create_memory("with color changed images and RF ")
        print(result)

    #
    def test_face_rec_knn_recognize_place(self):
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_pace.jpeg')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_bedroom_1.png')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_bedroom.png')
        # image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_forest.png')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_lab.jpeg')
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_meeting_room.jpeg')
        # image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_meeting_1.jpeg')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\office.jpeg')
        # image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_kitchen.jpeg')
        # image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_kitchen_1.jpeg')
        # image = cv2.imread( r'C:\Users\augme\Downloads\archive\data\20095.jpg')

        engine_configs = EngineConfigurations({})
        # engine_configs.add_configuration(
        # AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_CLASSIFIER_LOCATION,
        # r"C:\Users\Public\projects\drone-buddy-library\dronebuddylib\atoms\placerecognition\resources\models\classifiers\trained_place_knn_model24.clf")

        # engine_configs.add_configuration(
        #     AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
        #     "ResNet18")
        engine_configs.add_configuration(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
            "GoogLeNetPlaces365")
        # engine_configs.add_configuration(
        #     AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
        #     "Densenet121")

        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.recognize_place(image)
        # result = engine.recognize_place(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\place_2.jpg')
        # result = engine.recognize_place(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_pace.jpeg')
        print(result)
        #
        # def test_classifier(self):
        #     engine_configs = EngineConfigurations({})
        #     # engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
        #     engine_configs = EngineConfigurations({})
        #
        #     engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        #     result = engine.test_classifier()
        #     print(result)

    def test_feature_extraction(self):
        engine_configs = EngineConfigurations({})
        img_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_bedroom_1.png'
        image = cv2.imread(img_path)
        engine = PlaceRecognitionKNNImpl(engine_configs)
        # result = engine.extract_features_with_specified_models(img_path)
        result = engine.extract_features_with_specified_models(image)
        print(result)

    def test_face_rec_knn(self):
        tello = Tello()
        tello.connect()
        tello.streamon()
        session_number = "4"
        image = tello.get_frame_read().frame
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
            "GoogLeNetPlaces365")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.recognize_place(image)
        counter = 0
        while True:
            counter += 1
            image = tello.get_frame_read().frame
            result = engine.recognize_place(image)

            identified_place = "None"

            print(result)
            if result is not None and result.recognized_places is not None:
                print(result)
                path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number
                file_name_path = (
                        path + "\\" + result.recognized_places + "_" + str(
                    counter) + ".jpg")
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(file_name_path, image)

        cv2.imshow("Image" + identified_place + " _ ", image)
        cv2.waitKey(1)

    def test_determine_current_location(self):
        # List to store filenames
        filenames = []
        session_number = "demo_session_lab"
        folder_path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number

        # Regex to match the prefix, excluding numbers
        prefix_pattern = re.compile(r'(\D+)_')

        # Function to extract the prefix from a filename
        def get_prefix(filename):
            match = prefix_pattern.search(filename)
            if match:
                return match.group(1)
            return ""

        # List to store prefixes
        prefixes = []

        # Walk through the given folder and extract prefixes from file names
        for _, _, files in os.walk(folder_path):
            for filename in files:
                prefix = get_prefix(filename)
                if prefix:  # Only add if a prefix was found
                    prefixes.append(prefix)
            break  # Ensures we don't dive into subdirectories

        # Count occurrences of each prefix
        prefix_counts = Counter(prefixes)
        print(prefix_counts)
        if prefix_counts is not None:
            print("Most common prefix:", prefix_counts.most_common(1)[0][0])
            if prefix_counts.most_common(1)[0][0] == "unknown":
                could_be = prefix_counts.most_common(2)
                print("this could be the : ", could_be)

    def test_determine_place_on_video(self, tello):
        session_number = "unknown_session"
        image = tello.get_frame_read().frame
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
            "GoogLeNetPlaces365")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        counter = 0
        tello.move_up(50)
        predictions = []

        current_rotation = 0
        while True:
            if current_rotation > 360:
                tello.land()
                break
            counter += 1
            image = tello.get_frame_read().frame
            result = engine.recognize_place(image)

            identified_place = "None"

            print(result)
            if result is not None and result.most_likely is not None:
                print(result)
                predictions.append(result.most_likely.name)
                path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number
                file_name_path = (
                        path + "\\" + result.most_likely.name + "_" + str(
                    counter) + ".jpg")
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(file_name_path, image)
            current_rotation += 10
            tello.rotate_clockwise(10)

        cv2.imshow("Image" + identified_place + " _ ", image)
        cv2.waitKey(1)
        # Count occurrences of each prefix
        prefix_counts = Counter(predictions)
        print("possiblities", prefix_counts)
        if prefix_counts is not None:
            print("Most common prefix:", prefix_counts.most_common(1)[0][0])
            if prefix_counts.most_common(1)[0][0] == "unknown":
                could_be = prefix_counts.most_common(2)
                print("this could be the : ", could_be)
            else:
                print("this is the place: ", prefix_counts.most_common(1)[0][0])

    def test_folder_for_max_files(self):
        extractor_name = "GoogLeNetPlaces365"
        classifier_name = "RF  48"
        changes = "Testing with i4 lab that are changed colors"
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(
            AtomicEngineConfigurations.PLACE_RECOGNITION_RF_EXTRACTOR,
            extractor_name)
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_RF, engine_configs)
        # # #
        # engine_configs.add_configuration(
        #     AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
        #     extractor_name)
        # engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)

        session_number = "unknown_session"
        folder_path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number

        # Dictionary to store the sum of confidences for each place
        confidence_sums = defaultdict(float)

        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  # Check if it's a file
                # Call the dummy method (or your actual method) with the file path

                cv2_image = cv2.imread(file_path)
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

                recognized_places = engine.recognize_place(cv2_image)
                print(recognized_places)
                # Sum up the confidences for each recognized place
                for place in recognized_places.recognition_possibilities:
                    confidence_sums[place.name] += place.confidence

        # Print the summed confidences
        confidence_string = ""
        for place, total_confidence in confidence_sums.items():
            literal = f"{place}: Total Confidence = {total_confidence}  \n "
            print(literal)
            confidence_string += literal

        # Determine the place with the highest summed confidence
        most_confident_place = max(confidence_sums, key=confidence_sums.get)
        highest_confidence_sum = confidence_sums[most_confident_place]

        model_string = (
                " Model : " + classifier_name + " :  Extractor : " + extractor_name + " :  Changes : " + changes +
                " :  Most probable place : " + most_confident_place + " -  " + str(highest_confidence_sum)
                + ": possibilities : " + confidence_string + " \n ")

        file_path = pkg_resources.resource_filename(__name__, "/classifier_test_data.txt")

        write_to_file_longer(file_path, model_string)

        print(f"Place with the highest confidence sum: {most_confident_place} ({highest_confidence_sum})")

    def test_say_out_loud(self):
        text = "this is nice"
        voice_gen_engine.say(text)

    def test_test(self):
        path_name = pkg_resources.resource_filename(__name__,
                                                    "resources/test_data/training_data/")
        print(path_name)

    def test_drone_images(self):
        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        self.is_drone_in_air = True
        print("battery: ", tello.get_battery())
        print("temperature: ", tello.get_temperature())
        counter = 0
        while True:
            image = tello.get_frame_read().frame
            #          show the images on a cv2 window
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
            counter += 1
            time.sleep(2)


if __name__ == '__main__':
    unittest.main()
