Sample Program
=============

A sample programs that uses dronebuddylib to interact with a drone is given below. This program uses the library to
recognize speech, recognize intent, and execute drone functions based on the recognized intent. The program also uses
the library to recognize faces, objects, and text. The program also uses the library to generate speech and read aloud
the recognized intent and the results of the face, object, and text recognition.



.. code-block:: python

        import asyncio
        import datetime
        import threading
        import time
        from typing import List

        import cv2
        import requests
        import speech_recognition
        from djitellopy import Tello
        from dronebuddylib import SpeechRecognitionEngine, IntentRecognitionEngine, FaceRecognitionEngine, \
            ObjectDetectionEngine, TextRecognitionEngine, SpeechGenerationEngine
        from dronebuddylib.models import EngineConfigurations
        from dronebuddylib.models.enums import AtomicEngineConfigurations, IntentRecognitionAlgorithm, DroneCommands, \
            FaceRecognitionAlgorithm, VisionAlgorithm, TextRecognitionAlgorithm, SpeechGenerationAlgorithm
        from dronebuddylib.utils.enums import SpeechRecognitionAlgorithm, SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms
        from dronebuddylib.utils.utils import Logger

        engine_configs = EngineConfigurations({})

        logger = Logger()

        is_drone_in_air = False


        def open_mic_operations(drone_instance, on_recognized_callback):
            speech_microphone = speech_recognition.Microphone()

            engine_configs.add_configuration(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                             SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)
            engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)

            intent_engine = init_intent_rec_engine()
            face_recognition_engine = init_face_rec_engine()
            object_recognition_engine = init_object_rec_engine()
            text_recognition_engine = init_text_rec_engine()
            voice_engine = init_voice_generation_engine()

            while True:
                with speech_microphone as source:
                    logger.log_info("TEST",
                                    "Recognizing voice *********************************************************************************************************")

                    print(
                        "*********************************************************************************************************")
                    print("Say something...")
                    print(
                        "*********************************************************************************************************")
                    print(time.time())
                    try:
                        logger.log_info("TEST",
                                        "Recognizing voice *********************************************************************************************************")
                        result = engine.recognize_speech(source)
                        if result.recognized_speech is not None:
                            logger.log_info("TEST", "Recognized: " + result.recognized_speech)
                            intent = recognize_intent_gpt(intent_engine, result.recognized_speech)
                            read_aloud_text = execute_drone_functions(intent, drone_instance, face_recognition_engine,
                                                                      object_recognition_engine,
                                                                      text_recognition_engine, voice_engine)
                            on_recognized_callback(read_aloud_text, voice_engine)

                        else:
                            logger.log_warning("TEST", "Not Recognized: voice ")

                    except speech_recognition.WaitTimeoutError:
                        engine.recognize_speech(source)

                    time.sleep(1)  # Sleep to simulate work and prevent a tight loop


        def generate_voice_response(voice_engine, text):
            try:
                voice_engine.read_phrase(text)
            except Exception as e:
                logger.log_error("Error in voice generation:", str(e))


        def recognize_intent_snips(recognized_text):
            engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)
            engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.SNIPS_NLU, engine_configs)
            recognized_intent = engine.recognize_intent(recognized_text)
            logger.log_info("Recognized intent: ", recognized_intent.intent)
            return recognized_intent.intent


        def init_intent_rec_engine():
            engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
            engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL, "gpt-3.5-turbo-0613")
            engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                                             "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\stats\\")
            engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                                             "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                                             "https://api.openai.com/v1/chat/completions")
            engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, engine_configs)
            return engine


        def init_object_rec_engine():
            engine_configs.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
            engine = ObjectDetectionEngine(VisionAlgorithm.YOLO, engine_configs)
            return engine


        def init_voice_generation_engine():
            engine = SpeechGenerationEngine(SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE.name, engine_configs)
            return engine


        def init_face_rec_engine():
            engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
            return engine


        def recognize_intent_gpt(engine, recognized_text):
            try:
                recognized_intent = engine.recognize_intent(recognized_text)
                logger.log_info("Recognized intent: ", recognized_intent.intent)
                return recognized_intent.intent
            except:
                logger.log_error("Recognized intent: ", 'NONE')
                return "NONE"


        def execute_drone_functions(intent: str, drone_instance, face_engine, object_engine, text_engine, voice_engine):
            global is_drone_in_air

            if intent == DroneCommands.TAKE_OFF.name:
                is_drone_in_air = True
                take_off(drone_instance)
                return "I'm taking off"
            elif intent == DroneCommands.LAND.name:
                is_drone_in_air = False
                land(drone_instance)
                return "I'm landing"
            elif intent == DroneCommands.ROTATE_CLOCKWISE.name:
                rotate_clockwise(drone_instance)
                return "I'm rotating clockwise"
            elif intent == DroneCommands.ROTATE_COUNTER_CLOCKWISE.name:
                rotate_counter_clockwise(drone_instance)
                return "I'm rotating counter clockwise"
            elif intent == DroneCommands.FORWARD.name:
                move_forward(drone_instance)
                return "I'm moving forward"
            elif intent == DroneCommands.BACKWARD.name:
                move_backward(drone_instance)
                return "I'm moving backward"
            elif intent == DroneCommands.LEFT.name:
                move_left(drone_instance)
                return "I'm moving to the left"
            elif intent == DroneCommands.RIGHT.name:
                move_right(drone_instance)
                return "I'm moving to the right"
            elif intent == DroneCommands.UP.name:
                move_up(drone_instance)
                return "I'm moving up"
            elif intent == DroneCommands.DOWN.name:
                move_down(drone_instance)
                return "I'm moving down"
            elif intent == DroneCommands.FLIP.name:
                flip_forward(drone_instance)  # Assuming flip_forward is the desired flip command
                return "I'm flipping"
            elif intent == DroneCommands.RECOGNIZE_TEXT.name:
                text = recognize_text(text_engine, drone_instance)
                return "I read the text as " + text
            elif intent == DroneCommands.RECOGNIZE_PEOPLE.name:
                return recognize_people(face_engine, drone_instance)
                # return "I'm trying to recognize people"
            elif intent == DroneCommands.RECOGNIZE_OBJECTS.name:
                detected = recognize_objects(object_engine, drone_instance)
                return detected
            elif intent == DroneCommands.STOP.name:
                land(drone_instance)
                is_drone_in_air = False
                return "I'm stopping"


        def init_drone():
            drone_instance = Tello()
            drone_instance.connect()
            drone_instance.streamon()
            return drone_instance


        def take_off(drone_instance):
            logger.log_info("Executing functions: ", "Drone is taking off")
            if drone_instance is not None:
                drone_instance.takeoff()


        def land(drone_instance):
            logger.log_info("Executing functions: ", "Drone is landing")
            if drone_instance is not None:
                drone_instance.land()


        def rotate_clockwise(drone_instance):
            logger.log_info("Executing functions: ", "Drone is rotating clockwise")
            if drone_instance is not None:
                drone_instance.rotate_clockwise(90)


        def rotate_counter_clockwise(drone_instance):
            logger.log_info("Executing functions: ", "Drone is rotating counter clockwise")
            if drone_instance is not None:
                drone_instance.rotate_counter_clockwise(90)


        def move_forward(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving forward")
            if drone_instance is not None:
                drone_instance.move_forward(30)


        def move_backward(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving backward")
            if drone_instance is not None:
                drone_instance.move_backward(30)


        def move_left(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving left")
            if drone_instance is not None:
                drone_instance.move_left(30)


        def move_right(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving right")
            if drone_instance is not None:
                drone_instance.move_right(30)


        def move_up(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving up")
            if drone_instance is not None:
                drone_instance.move_up(30)


        def move_down(drone_instance):
            logger.log_info("Executing functions: ", "Drone is moving down")
            if drone_instance is not None:
                drone_instance.move_down(30)


        def flip_forward(drone_instance):
            logger.log_info("Executing functions: ", "Drone is flipping forward")
            if drone_instance is not None:
                drone_instance.flip_forward()


        def flip_backward(drone_instance):
            logger.log_info("Executing functions: ", "Drone is flipping backward")
            if drone_instance is not None:
                drone_instance.flip_backward()


        def flip_left(drone_instance):
            logger.log_info("Executing functions: ", "Drone is flipping left")
            if drone_instance is not None:
                drone_instance.flip_left()


        def recognize_people(engine, drone_instance):
            logger.log_info("Executing functions: ", "Drone is recognizing people")
            current_resized_image = get_image_with_cv2(drone_instance)
            result = engine.recognize_face(current_resized_image)
            return describe_face_rec_results(result)


        def get_image_with_cv2(drone_instance):
            current_frame = drone_instance.get_frame_read().frame
            current_resized_image = cv2.resize(current_frame, (500, 500))
            return current_resized_image


        def recognize_objects(engine, drone_instance):
            logger.log_info("Executing functions: ", "Drone is recognizing objects")
            current_resized_image = get_image_with_cv2(drone_instance)

            detected_objects = engine.get_detected_objects(current_resized_image)
            return describe_object_rec_results(detected_objects.object_names)


        def format_list(object_list: list):
            # creates a string from the object list, if there are duplicates count the number of duplicates and add it to the string.
            # for example if there are duplicates of chair in te list, add to the string 2 chairs
            # add a comma after each object and before the last item and a 'and' before the last item

            formatted_list = ""
            for object in object_list:
                if object_list.count(object) > 1:
                    formatted_list += str(object_list.count(object)) + " " + object + ", "
                else:
                    formatted_list += object + ", "
            formatted_list = formatted_list[:-2]
            formatted_list = formatted_list[::-1].replace(",", "and ", 1)[::-1]

            return formatted_list


        def describe_face_rec_results(labels):
            global init_position
            # remove duplicates from the labels
            labels = list(dict.fromkeys(labels))

            read_aloud_text = "I see "
            if len(labels) == 0:
                read_aloud_text = "I don't see anyone I recognize"
            elif len(labels) == 1:
                read_aloud_text = " I see  " + get_describing_phrase(labels[0])
            elif len(labels) >= 2:
                read_aloud_text = " I see "
                for i in range(0, len(labels) - 2):
                    read_aloud_text += get_describing_phrase(labels[i]) + " ,  "
                read_aloud_text = read_aloud_text + " and  " + get_describing_phrase((labels[len(labels) - 1]))
            return read_aloud_text

        def get_describing_phrase(name):
            if name.lower() == 'unknown':
                return "someone I don't recognize"
            else:
                return name


        def describe_object_rec_results(labels):
            # remove duplicates from the labels
            labels = list(dict.fromkeys(labels))

            read_aloud_text = "I see "
            if len(labels) == 0:
                read_aloud_text = "I don't see anything in the front, you are safe to move forward"
            elif len(labels) == 1:
                read_aloud_text = " I see a " + labels[0]
            elif len(labels) >= 2:
                read_aloud_text = " I see a "
                for i in range(0, len(labels) - 1):
                    read_aloud_text += labels[i] + " , "
                read_aloud_text = read_aloud_text + " and a " + labels[len(labels) - 1]
            else:
                read_aloud_text = "I don't see anything in the front, you are safe to move forward"
            logger.log_success("Recognized objects: ", read_aloud_text)

            return read_aloud_text


        def init_text_rec_engine():
            engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
            return engine


        def recognize_text(engine, drone_instance):
            logger.log_info("Executing functions: ", "Drone is recognizing text")
            image_path = save_frame(drone_instance, "text_rec_images")
            result = engine.recognize_text(image_path)
            return result.text


        def save_frame(drone_instance, type):
            # Assuming 'frame' is your frame from the drone
            frame = drone_instance.get_frame_read().frame
            # Specify the path where you want to save the image
            output_path = r"C:\Users\Public\projects\drone-buddy-launcher\resources\\" + type + "\\" + str(
                datetime.datetime.now().timestamp()) + ".jpg"

            # Save the frame as a JPEG image
            cv2.imwrite(output_path, frame)
            return output_path


        def keep_drone_in_air(drone_instance):
            moving_dir = -1
            voice = init_voice_generation_engine()
            global is_drone_in_air
            while True:
                logger.log_info("Executing functions: ", "Drone is in the air")
                if drone_instance is not None and is_drone_in_air:
                    print("battery: ", drone_instance.get_battery())
                    if drone_instance.get_battery() < 20:
                        land(drone_instance)
                        voice.read_phrase("I'm running out of battery, I'm landing")
                    if drone_instance.get_temperature() > 90:
                        land(drone_instance)
                        voice.read_phrase("I'm getting overheated, I'm landing")

                    drone_instance.send_rc_control(0, 0, moving_dir, 0)
                    moving_dir = moving_dir * -1
                    # break
                time.sleep(4)  # Sleep to simulate work and prevent a tight loop


        def find_person(face_engine, drone_instance):
            logger.log_info("Executing functions: ", "Drone is recognizing people")
            current_resized_image = get_image_with_cv2(drone_instance)
            result = face_engine.recognize_face(current_resized_image)
            return result


        def on_voice_recognized(recognized_text, voice_engine):
            logger.log_info("Recognized text:", recognized_text)
            # Call the voice generation function with the recognized text
            generate_voice_response(voice_engine, recognized_text)


        if __name__ == '__main__':
            # drone_instance = None
            drone_instance = init_drone()
            # Create threads
            thread1 = threading.Thread(target=open_mic_operations, args=(drone_instance, on_voice_recognized,))
            thread2 = threading.Thread(target=keep_drone_in_air, args=(drone_instance,))

            # Start threads
            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()
