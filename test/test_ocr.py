import unittest

import dronebuddylib.atoms as dbl_atoms


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_ocr_with_clear_text(self):

        vision_engine = dbl_online.()
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_image_clear.jpg'
        recognized = dbl_online.detect_text(vision_engine, image_path)
        print(recognized)
        # recognized.text_annotations.pb[0].description

    def test_ocr_with_unclear_text(self):
        vision_engine = dbl_online.init_google_vision_engine()
        image_path = r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\dronebuddylib\online\atoms\test_image_ocr_unclear.png'
        recognized = dbl_online.detect_text(vision_engine, image_path)
        print(recognized)

    def test_ocr_with_billboard_text(self):
        vision_engine = dbl_online.init_google_vision_engine()
        image_path = r'C:\Users\malshadz\projects\DroneBuddy\drone-buddy-library\Test\test_ocr_image_billboard.jpg'
        recognized = dbl_online.detect_text(vision_engine, image_path)
        print(recognized)
