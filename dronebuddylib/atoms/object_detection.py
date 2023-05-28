import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

    
def detect_objects(img):
  """
  Detect the bounding boxes of the objects in a frame.

  Args:
      img (list): the frame to be detected

  Returns:
      list: the list of the bounding boxes of all objects detected
  """
  work_path = os.path.dirname(os.path.realpath(__file__))
  model_path = os.path.join(work_path, 'resources', 'objectdetection', 'efficientdet_lite0_uint8.tflite')
  base_options = python.BaseOptions(model_asset_path=model_path)
  options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.2)
  detector = vision.ObjectDetector.create_from_options(options)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
  # with ObjectDetector.create_from_options(options) as detector:
  detection_result = detector.detect(mp_image)
  bounding_boxes = []
  if detection_result is not None:
      for detection in detection_result.detections:
          bounding_box = detection.bounding_box
          origin_x = bounding_box.origin_x
          origin_y = bounding_box.origin_y
          width = bounding_box.width
          height = bounding_box.height
          bounding_boxes.append([origin_x, origin_y, width, height])
  return bounding_boxes