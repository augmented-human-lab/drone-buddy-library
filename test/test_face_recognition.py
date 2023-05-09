import unittest

import cv2

import dronebuddylib.atoms as dbl


class TestFaceRecognition(unittest.TestCase):

    def test_add_to_the_memory(self):
        image = cv2.imread('test_image.jpg')
        file_path = r"D:\projects\drone-buddy-library\test\123.png"
        # with open(file_path, 'r') as file:
        #     image_file = file.read()
        result = dbl.add_people_to_memory('yasith.png', 'yasith', file_path)
        self.assertEqual(True, result)  # add assertion here

    def test_video_capture(self):
        video_capture = cv2.VideoCapture(0)

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            people_names = dbl.find_all_the_faces(frame)
            print(people_names)

    def test_face_rec_with_image(self):
        image = cv2.imread('group.jpg')
        people_names = dbl.find_all_the_faces(image)
        print(people_names)


if __name__ == '__main__':
    unittest.main()
