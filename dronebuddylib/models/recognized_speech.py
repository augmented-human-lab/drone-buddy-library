class RecognizedSpeechResult:
    def __init__(self, text, total_billed_time):
        self._recognized_speech = text
        self._total_billed_time = total_billed_time