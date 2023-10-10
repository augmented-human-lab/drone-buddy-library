import cv2
import mediapipe as mp
from djitellopy import Tello


def get_head_bounding_box(tello: Tello):
    """
    Get the bounding box of the head in front of the drone.

    Args:
        tello (Tello)

    Returns:
        list: image,
        [int: x coordinate of the left top corner of the bounding box,
        int: y coordinate of the left top corner of the bounding box,
        int: width,
        int: height]
    """
    while True:
        mp_face_detection = mp.solutions.face_detection
        image = tello.get_frame_read().frame
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(max(bboxC.ymin - 0.1, 0) * ih), int(bboxC.width * iw), int(
                        (bboxC.ymin + bboxC.height - max(bboxC.ymin - 0.1, 0)) * ih)
                    return image, [x, y, w, h]
