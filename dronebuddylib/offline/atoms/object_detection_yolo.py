import cv2
import numpy as np
import pkg_resources

from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()


# Please put the three files in the same working directory.
class YoloEngine:

    def __init__(self, weights_path: str):
        """
         Initialize a YOLO (You Only Look Once) object detection engine.

         Args:
             weights_path: The file path to the pre-trained weights file.

         Returns:
             None.

         Raises:
             FileNotFoundError: If the specified configuration or labels file is not found.

         """
        config = pkg_resources.resource_filename(__name__, "resources/objectdetection/yolov3.cfg")
        labels = pkg_resources.resource_filename(__name__, "resources/objectdetection/yolov3.txt")
        weights = weights_path
        try:
            self.net = cv2.dnn.readNet(weights, config)
            with open(labels, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        except FileNotFoundError as e:
            raise FileNotFoundError("The specified weights is not found.", e) from e


def init_yolo_engine(weights_path: str):
    """
     Initialize a YOLO (You Only Look Once) object detection engine.

     Args:
         weights_path: The file path to the pre-trained weights file.

     Returns:
         None.

     Raises:
         FileNotFoundError: If the specified configuration or labels file is not found.

     """
    logger.info('Object Detection YOLO : Initialized YOLO model')

    # Return the YoloEngine object for further use in the program
    return YoloEngine(weights_path)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def get_label_yolo(yolo_engine: YoloEngine, image):
    """
       Get the labels of objects detected in an image using a YOLO (You Only Look Once) object detection engine.

       Args:
           yolo_engine: The YoloEngine object used for object detection.
           image: The image to detect objects in.

       Returns:
           A list of labels corresponding to the objects detected in the image.

       """

    # Get the classes list from the YoloEngine object
    classes = yolo_engine.classes

    # Get the width and height of the image
    try:
        width = image.shape[1]
        height = image.shape[0]
    except:
        raise ValueError("The specified image is not valid.")

    # Initialize lists for the class IDs, confidences, boxes, and labels of detected objects
    class_ids = []
    confidences = []
    boxes = []
    labels = []

    # Set the non-maximum suppression (NMS) and confidence thresholds
    nms_threshold = 0.4
    conf_threshold = 0.5

    # Get the pre-trained weights and configuration from the YoloEngine object
    net = yolo_engine.net

    # Preprocess the image using the OpenCV deep neural network (dnn) module
    blob = cv2.dnn.blobFromImage(image, 0.004, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output layers of the YOLO model
    outs = net.forward(get_output_layers(net))

    # Process each detected object and its corresponding confidence level and class ID
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the confidence level of the detected object is above the specified threshold,
            # add it to the list of detected objects
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                confidences.append(float(confidence))
                class_ids.append(class_id)
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Get the label of each detected object and add it to the list of labels
    for i in indices:
        labels.append(str(classes[class_ids[i]]))

    logger.debug('Object Detection YOLO : detected objects: %s', labels)

    # Return the list of labels
    return labels


def get_boxes_yolo(yolo_engine: YoloEngine, image):
    """
       Get the bounding boxes of objects detected in an image using a YOLO (You Only Look Once) object detection engine.

       Args:
           yolo_engine: The YoloEngine object used for object detection.
           image: The image to detect objects in.

       Returns:
           A list of bounding boxes corresponding to the objects detected in the image.

       """

    # Get the width and height of the image
    try:
        width = image.shape[1]
        height = image.shape[0]
    except:
        raise ValueError("The specified image is not valid.")

    # Initialize lists for the class IDs, confidences, boxes, and labels of detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Get the pre-trained weights and configuration from the YoloEngine object
    net = yolo_engine.net

    # Preprocess the image using the OpenCV deep neural network (dnn) module
    blob = cv2.dnn.blobFromImage(image, 0.004, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output layers of the YOLO model
    outs = net.forward(get_output_layers(net))

    # Process each detected object and its corresponding confidence level and class ID
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the confidence level of the detected object is above the specified threshold,
            # add it to the list of detected objects
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                confidences.append(float(confidence))
                class_ids.append(class_id)
                boxes.append([x, y, w, h])
    return boxes
