import unittest

from dronebuddylib.atoms.textrecognition.text_recognition_engine import TextRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import TextRecognitionAlgorithm


# read input image


class TestTextRecognition(unittest.TestCase):

    def test_ocr_with_clear_text(self):
        engine_configs = EngineConfigurations({})

        engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image_clear.jpg'
        result = engine.recognize_text(image_path)
        print(result)
        assert result.text == 'MERRY\nCHRISTMAS\nAND HAPPY NEW YEAR'  # Assuming TextRecognitionResult stores detected text in a 'text' attribute
        assert result.locale == 'en'

    def test_ocr_with_noisy_background(self):
        # Similar to test_ocr_with_clear_text but with a different mock response simulating a noisy background
        engine_configs = EngineConfigurations({})

        engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\noisy_background.png'
        result = engine.recognize_text(image_path)
        print(result.text)
        print(result.locale)
        assert "Lorem ipsum dolor sit" in result.text  # Check if the specific substring is present in the recognized text
        assert result.locale == "la"  # Assuming the locale is English

    def test_ocr_with_unclear_text(self):
        # Similar to test_ocr_with_clear_text but with a different mock response simulating a noisy background
        engine_configs = EngineConfigurations({})

        engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image_ocr_unclear.png'

        result = engine.recognize_text(image_path)
        print(result.text)
        print(result.locale)
        assert result.text != ""  # Assuming some text is still recognized despite the noise

    def test_ocr_with_non_english_text(self):
        # Test with non-English text. Adjust the mock response accordingly.
        engine_configs = EngineConfigurations({})

        engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\non_english.png'
        result = engine.recognize_text(image_path)
        print(result.text)
        print(result.locale)
        assert "星4手小牛" in result.text  # Check if the specific substring is present in the recognized text
        assert result.locale != "en"  # Assuming the locale is not English

    def test_ocr_with_no_text_image(self):
        # Test with an image that has no text. Adjust the mock response to return empty text.
        engine_configs = EngineConfigurations({})

        engine = TextRecognitionEngine(TextRecognitionAlgorithm.GOOGLE_VISION, engine_configs)
        image_path = r'C:\Users\Public\projects\drone-buddy-library\test\test_images\no_text_image.png'
        result = engine.recognize_text(image_path)
        print(result.text)
        print(result.locale)
        assert result.text == ""  # Expecting no text to be detected
