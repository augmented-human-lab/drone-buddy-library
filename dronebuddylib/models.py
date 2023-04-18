import dronebuddylib.atoms as dbl_atoms


class EngineBank:
    def __init__(self):
        self.speech_to_text_engine = None
        self.text_to_speech_engine = None
        self.object_detection_yolo_engine = None
        self.drone_instance = None

    def set_speech_to_text_engine(self, speech_model):
        speech = dbl_atoms.init_speech_to_text_engine(speech_model)
        self.speech_to_text_engine = speech

    def set_text_to_speech_engine(self, rate=150, volume=1, voice_id='TTS_MS_EN-US_ZIRA_11.0'):
        text = dbl_atoms.init_text_to_speech_engine(rate, volume, voice_id)
        self.text_to_speech_engine = text

    def set_object_detection_yolo_engine(self, weights_path):
        image = dbl_atoms.init_yolo_engine(weights_path)
        self.object_detection_yolo_engine = image

    def set_drone_instance(self, drone_instance):
        self.drone_instance = drone_instance

    def get_speech_to_text_engine(self):
        return self.speech_to_text_engine

    def get_text_to_speech_engine(self):
        return self.text_to_speech_engine

    def get_object_detection_yolo_engine(self):
        return self.object_detection_yolo_engine

    def get_drone_instance(self):
        return self.drone_instance
