import unittest

import cv2

import offline.atoms as dbl



class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image = cv2.imread('test_image_clear.jpg')

        labels = dbl.get_label_yolo(image)
        assert len(labels) > 0 and 'chair' in labels

        print(labels)

    def test_me(self):
        print('test_me')


if __name__ == '__main__':
    unittest.main()
