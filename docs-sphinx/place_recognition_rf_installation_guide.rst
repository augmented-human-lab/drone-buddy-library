Place Recognition using Random Forests
======================================

The `PlaceRecognitionRFImpl` class implements place recognition using the Random Forest algorithm, leveraging image features extracted via pre-trained convolutional neural networks (CNNs). This implementation is ideal for applications requiring geographical awareness from visual cues, such as in robotics and drones.

Installation
------------

To install the Place Recognition module, run the following snippet, which will install the required dependencies:

.. code-block::

    pip install dronebuddylib[PLACE_RECOGNITION_RF]

Usage
-----

The Place Recognition module requires the following configurations to function:

- `PLACE_RECOGNITION_RF_TRAINING_DATA_SET_PATH`: Path to the training dataset - not required, if not used will be using the default location.
- `PLACE_RECOGNITION_RF_MODEL_SAVING_PATH`: Path to save the trained model - not required, if not used will be using the default location.
- `PLACE_RECOGNITION_RF_CLASSIFIER_LOCATION`: Path to the saved model for recognition purposes - not required, if not used will be using the default location.

Code Example
------------

1. **Training the Model**

   .. code-block:: python

        engine_configs = EngineConfigurations({})

         # Mark this false, if you are using the mobile to create the dataset
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         'True')
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_DRONE_INSTANCE,
                                         tello)

        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.remember_place(None, name, tello)


2. **Recognizing a Place**

   .. code-block:: python

       import cv2
       from place_recognition_rf import PlaceRecognitionRFImpl

       # Load your image
       image = cv2.imread('path/to/new/image.jpg')


       # Recognize the place
       recognized_places = engine.recognize_place(image)

Output
------

The output will be given in the following JSON format:

.. code-block:: json

    {
      "most_probable_place": {
        "place_name": "John's office",
        "confidence": 0.85
      },
      "recognized_places": [
        {
          "place_name": "John's office",
          "confidence": 0.85
        },
        {
          "place_name": "Jane's office",
          "confidence": 0.10
        },
        {
          "place_name": "NUS librar",
          "confidence": 0.05
        }
      ]
    }

Working example
---------------


.. code-block:: python

    session_number = "unknown_session"
        image = tello.get_frame_read().frame
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(
            AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
            "GoogLeNetPlaces365")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        counter = 0
        tello.move_up(50)
        predictions = []

        current_rotation = 0
        while True:
            if current_rotation > 360:
                tello.land()
                break
            counter += 1
            image = tello.get_frame_read().frame
            result = engine.recognize_place(image)

            identified_place = "None"

            print(result)
            if result is not None and result.most_likely is not None:
                print(result)
                predictions.append(result.most_likely.name)
                path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number
                file_name_path = (
                        path + "\\" + result.most_likely.name + "_" + str(
                    counter) + ".jpg")
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(file_name_path, image)
            current_rotation += 10
            tello.rotate_clockwise(10)

        cv2.imshow("Image" + identified_place + " _ ", image)
        cv2.waitKey(1)
        # Count occurrences of each prefix
        prefix_counts = Counter(predictions)
        print("possiblities", prefix_counts)
        if prefix_counts is not None:
            print("Most common prefix:", prefix_counts.most_common(1)[0][0])
            if prefix_counts.most_common(1)[0][0] == "unknown":
                could_be = prefix_counts.most_common(2)
                print("this could be the : ", could_be)
            else:                print("this is the place: ", prefix_counts.most_common(1)[0][0])


Algorithm Description
---------------------

The `PlaceRecognitionRFImpl` class uses the Random Forest algorithm for place recognition by leveraging image features extracted from pre-trained CNN models such as ResNet, DenseNet, and GoogLeNet. The following steps outline the algorithm and approaches used:

1. Feature Extraction
   ~~~~~~~~~~~~~~~~~~~

   - **Pre-trained CNN Models**: The class utilizes pre-trained CNN models (e.g., ResNet, DenseNet) to extract feature vectors from images. The algorithm currently uses model trained on Google Places 365.
   - **Image Preprocessing**: Images are preprocessed using transformations like resizing, center cropping, normalization, and conversion to tensors suitable for input to the CNN models.

2. Random Forest Classifier
   ~~~~~~~~~~~~~~~~~~~~~~~~

   - **Training**:
     - The extracted features from the training images are used to train the Random Forest classifier.
     - The classifier is configured with parameters such as the number of trees (`n_estimators`), maximum depth (`max_depth`), and splitting criterion (`criterion`).
     - The training dataset is split into training and validation sets using `train_test_split`.
     - The classifier is trained on the training set and evaluated on the validation set to measure accuracy, precision, and other performance metrics.

   - **Recognition**:
     - For recognizing places in new images, the same preprocessing and feature extraction steps are applied.
     - The extracted feature vector is then passed to the trained Random Forest classifier to predict the place.
     - The classifier provides probabilities for each possible place, and the place with the highest probability is selected as the recognized place.

3. Additional Features
   ~~~~~~~~~~~~~~~~~~~

   - **Confidence Scores**: The class computes confidence scores based on the proportion of trees in the Random Forest that vote for each class.
   - **Progress Monitoring**: The training and recognition processes can be monitored using progress bars and status updates.
   - **Drone Integration**: Optionally, the class can use a drone to capture images for dataset creation, providing a method for creating comprehensive datasets from various heights and angles.

This guide provides a comprehensive overview of the installation, usage, and algorithmic approach of the `PlaceRecognitionRFImpl` class for place recognition using Random Forests and CNNs.
