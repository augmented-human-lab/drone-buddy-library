import pyttsx3

''''This is a wrapper for ttx. '''


class Voice:

    def __init__(self, r, v):
        self.engine = pyttsx3.init()
        self.rate = self.engine.setProperty('rate', r)
        self.volume = self.engine.setProperty("volume", v)

    def get_rate(self):
        return self.rate

    def get_volume(self):
        return self.volume

    def set_rate(self, new_rate):
        self.engine.setProperty('rate', new_rate)
        return

    # This function is to set the volume of the voice. The volume should between 0 and 1.
    def set_volume(self, new_volume):
        self.engine.setProperty('volume', new_volume)
        return

    # This function is to set the texture of the voice, such as language, gender.
    # For more voice_ids, please see the documentation.
    def set_voice_id(self, new_voice_id):
        self.engine.setProperty('voice', new_voice_id)
        return

    # The input is the text. The output is the audio.
    def play_audio(self, text):
        print(text)
        self.engine.say(text)
        print("Done")
        self.engine.runAndWait()
        self.engine.stop()
        return
