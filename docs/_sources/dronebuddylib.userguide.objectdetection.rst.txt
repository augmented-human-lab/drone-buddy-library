Supported models
==========

YOLO
~~~~~~~~~~~~~~~~~~~~~~~


YOLO (You Only Look Once) is a popular object detection algorithm known for its fast and real-time performance. It stands out for its ability to simultaneously predict object classes and bounding box coordinates in a single forward pass through a deep neural network.

Please refer to the [YOLOv5](https://docs.ultralytics.com/) documentation for more details.
Here's a simplified explanation of how YOLO works:

Dividing the Image into Grid: YOLO divides the input image into a grid of cells. Each cell is responsible for predicting objects located within its boundaries.

#. Anchor Boxes: YOLO uses pre-defined anchor boxes, which are a set of bounding box shapes with different aspect ratios. These anchor boxes are initially defined based on the characteristics of the dataset being used.

#. Prediction: The neural network is designed to simultaneously predict multiple bounding boxes and their corresponding class probabilities within each grid cell. For each anchor box, the network predicts the coordinates (x, y, width, height) of the bounding box and the confidence score representing the likelihood of containing an object. It also predicts class probabilities for each object class.

#. Non-Maximum Suppression: YOLO applies a post-processing step called non-maximum suppression (NMS) to remove duplicate or overlapping bounding box predictions. NMS selects the most confident detection among overlapping boxes based on a defined threshold.

#. Output: The final output of the YOLO algorithm is a set of bounding boxes along with their class labels and confidence scores, representing the detected objects in the image.

YOLO's key advantages lie in its speed and efficiency. Since it performs object detection in a single pass through the neural network, it avoids the need for region proposals or sliding windows, resulting in faster inference times. This makes YOLO suitable for real-time applications like video analysis, robotics, and autonomous vehicles.

YOLO has evolved over time, and different versions such as YOLOv1, YOLOv2 (also known as YOLO9000), YOLOv3, YOLOv4, YOLOv8 have been introduced. These iterations have incorporated various improvements, including network architecture changes, feature extraction enhancements, and the use of more advanced techniques like skip connections and feature pyramid networks.

YOLO models are typically trained on large labeled datasets, such as COCO (Common Objects in Context), to learn to detect objects across multiple classes effectively. The training process involves optimizing the neural network parameters using techniques like backpropagation and gradient descent.

Keep in mind that while YOLO offers fast inference times, it may sacrifice some accuracy compared to slower, more complex object detection algorithms. The choice of object detection algorithm depends on the specific requirements of the application, balancing factors like accuracy, speed, and available computational resources.

How to use
--------------------

.. code-block:: Python

     image = cv2.imread('test_image.jpg')

     engine_configs = EngineConfigurations({})
     engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
     engine = ObjectDetectionEngine(VisionAlgorithm.YOLO, engine_configs)
     objects = engine.get_detected_objects(image)


MediaPipe
~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/object_detector) documentation for more details.


DroneBuddy utilizes MediaPipe for object detection, leveraging its advanced computer vision capabilities. This section provides an overview of how MediaPipe’s object detection is integrated and employed in DroneBuddy, enhancing its functionalities. Detailed information about MediaPipe can be found on its official website or documentation.

#. Computer Vision Technology: MediaPipe offers state-of-the-art computer vision technology, enabling DroneBuddy to detect and identify objects in real-time. It uses machine learning models to recognize various objects within the camera’s field of view.

#. Real-Time Processing: MediaPipe’s object detection in DroneBuddy is designed for real-time application. It efficiently processes video frames to detect objects quickly and accurately, which is crucial for dynamic drone operations.

#. Robust and Versatile: MediaPipe's models are trained on a diverse set of data, making them robust and versatile for different environments and scenarios. This versatility is beneficial for DroneBuddy, which may operate in various settings.

#. Lightweight and Efficient: The algorithms used in MediaPipe are optimized for performance, ensuring that they are lightweight and efficient. This is particularly important for DroneBuddy, as it allows for faster processing without overburdening the drone's computational resources.

Integration in DroneBuddy
-------------------------

#. Installation and Setup: To integrate MediaPipe in DroneBuddy, first, install the MediaPipe library in the development environment. This might involve adding MediaPipe as a dependency in the project.

#. Model Selection: Choose the appropriate MediaPipe object detection model based on DroneBuddy’s requirements. MediaPipe offers several pre-trained models that vary in accuracy and performance.

#. Video Stream Processing: In DroneBuddy, integrate the MediaPipe object detection into the video stream processing pipeline. This involves capturing video frames from the drone’s camera and feeding them into the MediaPipe model for object detection.

#. Handling Detection Output: Once objects are detected, DroneBuddy processes this information. This could involve marking objects on a display, logging information, or making real-time decisions based on the detected objects.

#. Tuning and Customization: Depending on the specific use case of DroneBuddy, you might need to fine-tune the MediaPipe models or customize the detection parameters for optimal performance.

Example of MediaPipe Integration in DroneBuddy
------------------------------------------------

Here’s a hypothetical code example showing how MediaPipe’s object detection could be integrated into DroneBuddy:

.. code-block:: python

     image = cv2.imread('test_image.jpg')

     engine_configs = EngineConfigurations({})
     engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
     engine = ObjectDetectionEngine(VisionAlgorithm.MEDIA_PIPE, engine_configs)


     objects = engine.get_detected_objects(image)

This example demonstrates capturing video frames from DroneBuddy’s camera, processing them through MediaPipe for object detection, and then handling the detection results.


Important Considerations
------------------------
While MediaPipe offers sophisticated object detection capabilities, it's important to note that its performance can vary based on environmental conditions, object sizes, and camera quality. Regular testing and adjustments may be necessary to ensure MediaPipe operates effectively within the specific context of DroneBuddy.

