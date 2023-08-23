from google.cloud import speech


def init_google_speech_engine():
    """
    Initializes the Google Cloud Speech-to-Text client.

    Returns:
        speech.SpeechClient: The initialized Speech-to-Text client.

    Example:
        speech_client = init_google_speech_engine()
    """
    return speech.SpeechClient()


def recognize_speech(client, audio_steam) -> speech.RecognizeResponse:
    """
    Recognizes speech from an audio stream using the Google Cloud Speech-to-Text client.

    Args:
        client (speech.SpeechClient): The Speech-to-Text client instance.
        audio_stream (bytes): The audio stream content to be recognized.

    Returns:
        speech.RecognizeResponse: The response containing recognized speech results.

    Example:
        audio_content = get_audio_stream_from_somewhere()
        response = recognize_speech(speech_client, audio_content)
        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")
    """
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
