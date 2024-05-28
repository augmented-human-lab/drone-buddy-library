YOLO Object Detection
========================

The official documentation for YOLO can be found `here <(https://docs.ultralytics.com/>`_.

Installation
-------------

To install YOLO Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[OBJECT_DETECTION_YOLO]


Usage
-------------

The YOLO integration module requires the following configurations to function

#.  OBJECT_DETECTION_YOLO_VERSION - This refers to the model that you want to use for the detection purposes. The list of versions can be found `here <https://docs.ultralytics.com/`_

Code Example
-------------

.. code-block:: python

    image = cv2.imread('test_image.jpg')

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
    engine = ObjectDetectionEngine(VisionAlgorithm.YOLO, engine_configs)
    objects = engine.get_detected_objects(image)




Output
-------------

The output will be given in the following json format

.. code-block:: json

    {
      "message": "",
      "result": {
        "object_names": [
          ""
        ],
        "detected_objects": [
          {
            "detected_categories": [
              {
                "category_name": "",
                "confidence": 0
              }
            ],
            "bounding_box": {
              "origin_x": 0,
              "origin_y": 0,
              "width": 0,
              "height": 0
            }
          }
        ]
      }
    }



