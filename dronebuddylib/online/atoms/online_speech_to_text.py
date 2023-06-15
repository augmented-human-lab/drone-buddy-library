from google.cloud import speech


def init_google_speech_engine():
    return speech.SpeechClient()


def recognize_speech(client, audio_steam) -> speech.RecognizeResponse:
    audio = speech.RecognitionAudio(content=audio_steam)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response
