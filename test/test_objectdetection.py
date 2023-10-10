import unittest

import cv2

import dronebuddylib.offline.atoms as dbl



# read input image


class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')
        # image = cv2.imread('group.jpg')
        configs = VisionConfigs(r"C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\Test\yolov3.weights")
        labels = dbl.detect_objects(VisionAlgorithm.YOLO_V8, configs, image)
        # assert len(labels) > 0 and 'chair' in labels

        print("labels", labels)

    def test_me(self):
        print('test_me')


if __name__ == '__main__':
    unittest.main()
