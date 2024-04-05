import unittest

import cv2
import numpy as np

from dronebuddylib.atoms.bodyfeatureextraction import BodyFeatureExtractionImpl, HandFeatureExtractionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
import mediapipe as mp


class TestFeatureExtraction(unittest.TestCase):
    def test_body_feature_extraction_get_detected_pose(self):
        # Load the image
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\pose_detection_test.png'
        image = cv2.imread(image_path)

        # Ensure the image is loaded
        if image is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verify that image_rgb is a NumPy array of an appropriate type (e.g., numpy.uint8)
        if not (isinstance(image_rgb, np.ndarray) and image_rgb.dtype == np.uint8):
            raise TypeError("Image data is not in the expected format (numpy.uint8)")

        # Create a MediaPipe Image object
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Continue with your existing processing
        engine_configs = EngineConfigurations({})
        engine = BodyFeatureExtractionImpl(engine_configs)
        result = engine.get_detected_pose(image_rgb)
        drawn_image = engine.draw_landmarks_on_image(image_rgb, result)
        cv2.imshow("final_image", drawn_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self.assertIsNot(len(result.pose_landmarks), 0)  # add assertion here

    def test_hand_feature_extraction_get_detected_pose(self):
        # Load the image
        # image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\thumbs_up.jpeg'
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\frame_with_person456.jpg'
        image = cv2.imread(image_path)

        # Ensure the image is loaded
        if image is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verify that image_rgb is a NumPy array of an appropriate type (e.g., numpy.uint8)
        if not (isinstance(image_rgb, np.ndarray) and image_rgb.dtype == np.uint8):
            raise TypeError("Image data is not in the expected format (numpy.uint8)")

        # Create a MediaPipe Image object
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Continue with your existing processing
        engine_configs = EngineConfigurations({})
        engine = HandFeatureExtractionImpl(engine_configs)
        result = engine.get_gesture(image_rgb)

        self.assertIsNot(len(result.hand_landmarks), 0)  # add assertion here
        self.assertEqual(result.gestures[0][0].category_name, 'Thumb_Up')  # add assertion here


    def test_hand_feature_extraction_get_detected_pose_different_pose(self):
        # Load the image
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\victory-hand-gesture.jpg'
        image = cv2.imread(image_path)

        # Ensure the image is loaded
        if image is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verify that image_rgb is a NumPy array of an appropriate type (e.g., numpy.uint8)
        if not (isinstance(image_rgb, np.ndarray) and image_rgb.dtype == np.uint8):
            raise TypeError("Image data is not in the expected format (numpy.uint8)")

        # Create a MediaPipe Image object
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Continue with your existing processing
        engine_configs = EngineConfigurations({})
        engine = HandFeatureExtractionImpl(engine_configs)
        result = engine.get_gesture(image_rgb)

        self.assertIsNot(len(result.hand_landmarks), 0)  # add assertion here
        self.assertEqual(result.gestures[0][0].category_name, 'Victory')  # add assertion here

    def test_hand_feature_extraction_count_fingers(self):
        # Load the image
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\3_fingers.jpg'
        image = cv2.imread(image_path)

        # Ensure the image is loaded
        if image is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verify that image_rgb is a NumPy array of an appropriate type (e.g., numpy.uint8)
        if not (isinstance(image_rgb, np.ndarray) and image_rgb.dtype == np.uint8):
            raise TypeError("Image data is not in the expected format (numpy.uint8)")

        # Create a MediaPipe Image object
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Continue with your existing processing
        engine_configs = EngineConfigurations({})
        engine = HandFeatureExtractionImpl(engine_configs)
        result = engine.count_raised_fingers(image_rgb)

        self.assertEqual(result, 2)  # add assertion here
        # self.assertEqual(result.gestures[0][0].category_name, 'Thumb_Up')  # add assertion here


if __name__ == '__main__':
    unittest.main()
