import unittest

import cv2

from dblib.enums.vision_enums import ObjectDetectionReturnTypes
from dblib.objectdetection import detect_common_objects, detect_common_object_labels


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')

        labels = detect_common_objects(image)
        assert len(labels) > 0 and 'chair' in labels

        print(labels)

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')

        labels = detect_common_object_labels(image, detection_type=ObjectDetectionReturnTypes.LABELS)
        print(labels)
