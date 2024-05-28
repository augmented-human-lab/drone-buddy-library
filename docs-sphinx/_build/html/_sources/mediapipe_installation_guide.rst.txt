Mediapipe Object Detection
========================

The official documentation for vosk can be found `here <https://developers.google.com/mediapipe>`_.

Installation
-------------

To install Mediapipe Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[OBJECT_DETECTION_MP]


Usage
-------------

The Mediapipe integration module requires the no configurations to function.



Code Example
-------------

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine = MPObjectDetectionImpl(EngineConfigurations({}))
    detected_objects = engine.get_detected_objects(mp_image)





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

