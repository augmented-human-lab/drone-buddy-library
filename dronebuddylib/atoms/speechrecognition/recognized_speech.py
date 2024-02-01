class RecognizedSpeechResult:
    def __init__(self, text, total_billed_time):
        self.recognized_speech = text
        self.total_billed_time = total_billed_time

    def to_json(self):
        return {
            'recognized_speech': self.recognized_speech,
            'total_billed_time': self.total_billed_time
        }