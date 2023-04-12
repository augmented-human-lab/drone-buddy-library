import cv2
import numpy as np


# Please put the three files in the same working directory.


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def get_label_yolo(image):
    config = "./resources/objectdetection/yolov3.cfg"
    labels = "./resources/objectdetection/yolov3.txt"
    weights = "./resources/objectdetection/yolov3.weights"

    classes = None
    with open(labels, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    width = image.shape[1]
    height = image.shape[0]

    class_ids = []
    confidences = []
    boxes = []
    labels = []
    nms_threshold = 0.4
    conf_threshold = 0.5

    net = cv2.dnn.readNet(weights, config)
    blob = cv2.dnn.blobFromImage(image, 0.004, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        labels.append(str(classes[class_ids[i]]))
    return labels
