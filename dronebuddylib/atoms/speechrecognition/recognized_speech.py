class RecognizedSpeechResult:
    def __init__(self, text, total_billed_time):
        self.recognized_speech = text
        self.total_billed_time = total_billed_time