import unittest

import cv2

import dronebuddylib as dbl


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image_engine = dbl.init_yolo_engine(
            r"C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\Test\resources\objectdetection\yolov3.weights")
        image = cv2.imread('test_image.jpg')

        labels = dbl.get_label_yolo(image_engine, image)
        assert len(labels) > 0 and 'chair' in labels

        print(labels)

    def test_me(self):
        print('test_me')


if __name__ == '__main__':
    unittest.main()
