import unittest

import cv2

import dronebuddylib as dbl


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')

        labels = dbl.detect_common_objects(image)
        assert len(labels) > 0 and 'chair' in labels

        print(labels)

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')

        labels = dbl.detect_common_object_labels(image, detection_type=dbl.ObjectDetectionReturnTypes.LABELS)
        print(labels)

    def test_me(self):
        print('test_me')


if __name__ == '__main__':
    unittest.main()
