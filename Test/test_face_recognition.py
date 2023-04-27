import unittest

import cv2

import dronebuddylib as dbl


class TestFaceRecognition(unittest.TestCase):

    def test_add_to_the_memory(self):
        dbl.add_people_to_memory('malsha.jpg', 'malsha')
        self.assertEqual(True, False)  # add assertion here

    def test_video_capture(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            people_names = dbl.find_all_the_faces(frame)
            print(people_names)

        self.assertEqual(True, False)  # add assertion here

    def test_face_rec_with_image(self):
        image = cv2.imread('group.jpg')
        people_names = dbl.find_all_the_faces(image)
        print(people_names)


if __name__ == '__main__':
    unittest.main()
